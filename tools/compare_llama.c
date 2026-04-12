#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef MODEL_PATH
#define MODEL_PATH "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
#endif
#ifndef LLAMA_CLI_DEFAULT
#define LLAMA_CLI_DEFAULT "llama-cli"
#endif

#define N_THREADS   4
#define N_GEN       32
#define MAX_OUTPUT  65536

static const char *g_model_path;
static const char *g_llama_cli;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int run_capture(const char *cmd, char *buf, int bufsz) {
    FILE *fp = popen(cmd, "r");
    if (!fp) return -1;
    int total = 0;
    while (total < bufsz - 1) {
        int n = (int)fread(buf + total, 1, (size_t)(bufsz - 1 - total), fp);
        if (n <= 0) break;
        total += n;
    }
    buf[total] = '\0';
    pclose(fp);
    return total;
}

/* Extract generation tokens/second from llama.cpp stderr.
 * Looks for the last "eval time =" line and parses "N tokens per second)". */
static double extract_llama_speed(const char *output, const char *label) {
    const char *search = label;  /* "llama_perf_sampler_print" or "eval time =" */
    const char *p = strstr(output, search);
    if (!p) return 0.0;
    /* Find last occurrence */
    const char *p2 = strstr(p + strlen(search), search);
    if (p2) p = p2;
    const char *tps = strstr(p, "tokens per second)");
    if (!tps) return 0.0;
    const char *e = tps - 1;
    while (e > p && *e == ' ') e--;
    const char *s = e;
    while (s > p && s[-1] != ' ' && s[-1] != ',') s--;
    return atof(s);
}

/* Extract prompt eval speed from llama.cpp output.
 * Looks for "prompt eval time =" ... "tokens per second)" */
static double extract_llama_prompt_speed(const char *output) {
    const char *p = strstr(output, "prompt eval time =");
    if (!p) return 0.0;
    const char *tps = strstr(p, "tokens per second)");
    if (!tps) return 0.0;
    const char *e = tps - 1;
    while (e > p && *e == ' ') e--;
    const char *s = e;
    while (s > p && s[-1] != ' ' && s[-1] != ',') s--;
    return atof(s);
}

typedef struct {
    int   n_tokens;
    char  text[4096];
    int   text_len;
} gen_result_t;

/* Escape a string for JSON output (handles quotes and backslashes). */
static void json_print_string(const char *s) {
    putchar('"');
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') putchar('\\');
        if (*s == '\n') { putchar('\\'); putchar('n'); continue; }
        if (*s == '\r') { putchar('\\'); putchar('r'); continue; }
        if (*s == '\t') { putchar('\\'); putchar('t'); continue; }
        putchar(*s);
    }
    putchar('"');
}

int main(void) {
    g_model_path = getenv("BITNET_MODEL");
    if (!g_model_path) g_model_path = MODEL_PATH;
    g_llama_cli = getenv("LLAMA_CLI");
    if (!g_llama_cli) g_llama_cli = LLAMA_CLI_DEFAULT;

    /* --- Setup checks --- */

    /* Check model file exists */
    if (access(g_model_path, R_OK) != 0) {
        fprintf(stderr, "ERROR: model file not found: %s\n", g_model_path);
        fprintf(stderr, "Set BITNET_MODEL to the path of ggml-model-i2_s.gguf\n");
        return 1;
    }

    /* Check llama-cli is available */
    {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "command -v '%s' >/dev/null 2>&1", g_llama_cli);
        if (system(cmd) != 0) {
            fprintf(stderr, "ERROR: llama-cli not found: %s\n", g_llama_cli);
            fprintf(stderr, "Set LLAMA_CLI to the path of llama-cli, "
                            "or ensure it is on $PATH\n");
            return 1;
        }
    }

    fprintf(stderr, "=== bitnet-c11 vs llama.cpp Comparison ===\n");
    fprintf(stderr, "Model:     %s\n", g_model_path);
    fprintf(stderr, "llama-cli: %s\n", g_llama_cli);
    fprintf(stderr, "Threads:   %d\n", N_THREADS);

    /* --- Phase 1: bitnet-c11 benchmark --- */

    fprintf(stderr, "\n--- bitnet-c11 inference ---\n");

    bitnet_model_t *model = bitnet_model_load(g_model_path);
    if (!model) {
        fprintf(stderr, "ERROR: failed to load model\n");
        return 1;
    }

    bitnet_params_t params = bitnet_params_default();
    params.n_threads = N_THREADS;
    params.temperature = 0.0f;
    params.top_k = 1;

    bitnet_ctx_t *ctx = bitnet_ctx_new(model, params);
    if (!ctx) {
        fprintf(stderr, "ERROR: failed to create context\n");
        bitnet_model_free(model);
        return 1;
    }
    bn_sampler_seed(&ctx->sampler, 42);

    const char *prompt = "The meaning of life is";

    /* Prompt processing */
    int prompt_tokens[256];
    int n_prompt = bitnet_tokenize(ctx, prompt, prompt_tokens, 256);

    double pp_t0 = now_sec();
    float *logits = NULL;
    for (int i = 0; i < n_prompt; i++) {
        logits = bitnet_forward(ctx, &prompt_tokens[i], 1, (i == n_prompt - 1));
    }
    double pp_t1 = now_sec();
    double c11_pp_time = pp_t1 - pp_t0;
    double c11_pp_tps = (double)n_prompt / c11_pp_time;

    fprintf(stderr, "Prompt eval: %d tokens in %.2f ms (%.1f tok/s)\n",
            n_prompt, c11_pp_time * 1000, c11_pp_tps);

    /* Token generation */
    gen_result_t c11_result;
    memset(&c11_result, 0, sizeof(c11_result));

    double tg_t0 = now_sec();
    for (int i = 0; i < N_GEN; i++) {
        int token = bitnet_sample_token(ctx, logits);
        c11_result.n_tokens++;
        const char *text = bn_token_text(ctx->tokenizer, token);
        if (text) {
            int len = (int)strlen(text);
            if (c11_result.text_len + len < 4095) {
                memcpy(c11_result.text + c11_result.text_len, text, (size_t)len);
                c11_result.text_len += len;
                c11_result.text[c11_result.text_len] = '\0';
            }
        }
        logits = bitnet_forward(ctx, &token, 1, true);
        if (!logits) break;
    }
    double tg_t1 = now_sec();
    double c11_tg_time = tg_t1 - tg_t0;
    double c11_tg_tps = (double)c11_result.n_tokens / c11_tg_time;

    fprintf(stderr, "Generation:  %d tokens in %.2f ms (%.2f tok/s)\n",
            c11_result.n_tokens, c11_tg_time * 1000, c11_tg_tps);

    bitnet_ctx_free(ctx);
    bitnet_model_free(model);

    /* --- Phase 2: llama.cpp benchmark --- */

    fprintf(stderr, "\n--- llama.cpp inference ---\n");

    char cmd[2048];
    char ref_output[MAX_OUTPUT];

    snprintf(cmd, sizeof(cmd),
             "%s -m '%s' -p '%s' -n %d -t %d --temp 0.0 -b 1 -ngl 0 "
             "--no-display-prompt 2>&1",
             g_llama_cli, g_model_path, prompt, N_GEN, N_THREADS);

    int rc = run_capture(cmd, ref_output, MAX_OUTPUT);
    double ref_pp_tps = 0.0;
    double ref_tg_tps = 0.0;

    if (rc > 0) {
        ref_pp_tps = extract_llama_prompt_speed(ref_output);
        ref_tg_tps = extract_llama_speed(ref_output, "eval time =");
        /* If "eval time" didn't work, the last occurrence is generation speed */
        if (ref_tg_tps <= 0.0)
            ref_tg_tps = extract_llama_speed(ref_output, "tokens per second");

        if (ref_pp_tps > 0.0)
            fprintf(stderr, "Prompt eval: %.1f tok/s\n", ref_pp_tps);
        else
            fprintf(stderr, "Prompt eval: (could not parse)\n");

        if (ref_tg_tps > 0.0)
            fprintf(stderr, "Generation:  %.2f tok/s\n", ref_tg_tps);
        else
            fprintf(stderr, "Generation:  (could not parse)\n");
    } else {
        fprintf(stderr, "WARNING: llama-cli produced no output\n");
    }

    /* --- Phase 3: structured JSON output --- */

    fprintf(stderr, "\n--- Results ---\n");

    printf("{\n");
    printf("  \"model\": "); json_print_string(g_model_path); printf(",\n");
    printf("  \"prompt\": "); json_print_string(prompt); printf(",\n");
    printf("  \"threads\": %d,\n", N_THREADS);
    printf("  \"n_gen\": %d,\n", N_GEN);
    printf("  \"bitnet_c11\": {\n");
    printf("    \"prompt_eval_tok_s\": %.2f,\n", c11_pp_tps);
    printf("    \"generation_tok_s\": %.2f,\n", c11_tg_tps);
    printf("    \"generation_tokens\": %d,\n", c11_result.n_tokens);
    printf("    \"generation_time_ms\": %.2f\n", c11_tg_time * 1000);
    printf("  },\n");
    printf("  \"llama_cpp\": {\n");
    printf("    \"prompt_eval_tok_s\": %.2f,\n", ref_pp_tps);
    printf("    \"generation_tok_s\": %.2f\n", ref_tg_tps);
    printf("  }");

    if (ref_tg_tps > 0.0) {
        printf(",\n  \"speedup\": %.2f", c11_tg_tps / ref_tg_tps);
    }
    printf("\n}\n");

    /* Print human-readable summary to stderr */
    if (ref_tg_tps > 0.0) {
        double ratio = c11_tg_tps / ref_tg_tps;
        fprintf(stderr, "Generation speedup: %.2fx (c11=%.2f, llama=%.2f tok/s)\n",
                ratio, c11_tg_tps, ref_tg_tps);
    }
    if (ref_pp_tps > 0.0) {
        double ratio = c11_pp_tps / ref_pp_tps;
        fprintf(stderr, "Prompt eval speedup: %.2fx (c11=%.1f, llama=%.1f tok/s)\n",
                ratio, c11_pp_tps, ref_pp_tps);
    }

    return 0;
}
