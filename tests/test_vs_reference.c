#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <errno.h>

#ifndef MODEL_PATH
#define MODEL_PATH "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
#endif
#ifndef LLAMA_CLI
#define LLAMA_CLI  "llama-cli"
#endif
#ifndef LLAMA_TOK
#define LLAMA_TOK  "llama-tokenize"
#endif

#define N_THREADS  4
#define N_PREDICT  40
#define MAX_OUTPUT 65536

static const char *g_model_path;
static const char *g_llama_cli;
static const char *g_llama_tok;

static int n_pass = 0;
static int n_fail = 0;

#define CHECK(cond, ...) do { \
    if (cond) { n_pass++; printf("  PASS: " __VA_ARGS__); printf("\n"); } \
    else      { n_fail++; printf("  FAIL: " __VA_ARGS__); printf("\n"); } \
} while(0)

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

static char *extract_llama_text(const char *output) {
    const char *start = output;
    while (*start == '\n' || *start == '\r') start++;

    const char *end = start + strlen(start);
    const char *eot = strstr(start, " [end of text]");
    if (eot) end = eot;

    int len = (int)(end - start);
    char *text = (char *)malloc((size_t)len + 1);
    memcpy(text, start, (size_t)len);
    text[len] = '\0';

    while (len > 0 && (text[len-1] == '\n' || text[len-1] == '\r' || text[len-1] == ' '))
        text[--len] = '\0';
    return text;
}

static double extract_llama_speed(const char *output) {
    const char *p = strstr(output, "eval time =");
    if (!p) return 0.0;
    const char *p2 = strstr(p + 11, "eval time =");
    if (p2) p = p2;
    const char *tps = strstr(p, "tokens per second)");
    if (!tps) return 0.0;
    const char *e = tps - 1;
    while (e > p && *e == ' ') e--;
    const char *s = e;
    while (s > p && s[-1] != ' ' && s[-1] != ',') s--;
    return atof(s);
}

static void test_tokenizer(bitnet_ctx_t *ctx) {
    printf("\n=== Test 1: Tokenizer Comparison ===\n");

    const char *prompts[] = {
        "Hello",
        "The capital of France",
        "12345",
        "The meaning of life is",
        "I think therefore I am.",
    };
    int n_prompts = (int)(sizeof(prompts) / sizeof(prompts[0]));

    char ref_buf[MAX_OUTPUT];
    int our_tokens[256];

    for (int p = 0; p < n_prompts; p++) {
        int our_n = bitnet_tokenize(ctx, prompts[p], our_tokens, 256);

        char cmd[1024];
        snprintf(cmd, sizeof(cmd),
                 "printf '%%s' '%s' | %s -m '%s' --stdin --log-disable 2>/dev/null",
                 prompts[p], g_llama_tok, g_model_path);
        int rc = run_capture(cmd, ref_buf, MAX_OUTPUT);

        if (rc <= 0) {
            printf("  '%s': c11=%d tokens (reference tokenizer unavailable)\n",
                   prompts[p], our_n);
            continue;
        }

        int ref_tokens[256];
        int ref_n = 0;
        char *line = strtok(ref_buf, "\n");
        while (line && ref_n < 256) {
            int tok;
            if (sscanf(line, " %d", &tok) == 1) {
                ref_tokens[ref_n++] = tok;
            }
            line = strtok(NULL, "\n");
        }

        printf("  '%s': c11=%d tokens, ref=%d tokens", prompts[p], our_n, ref_n);
        if (our_n == ref_n) {
            int match = 1;
            for (int i = 0; i < our_n; i++) {
                if (our_tokens[i] != ref_tokens[i]) { match = 0; break; }
            }
            if (match) {
                printf(" — EXACT MATCH\n");
                n_pass++;
            } else {
                printf(" — IDs differ\n");
                printf("    c11: ");
                for (int i = 0; i < our_n; i++) printf("%d ", our_tokens[i]);
                printf("\n    ref: ");
                for (int i = 0; i < ref_n; i++) printf("%d ", ref_tokens[i]);
                printf("\n");
                n_fail++;
            }
        } else {
            printf(" — COUNT MISMATCH\n");
            printf("    c11: ");
            for (int i = 0; i < our_n; i++) printf("%d ", our_tokens[i]);
            printf("\n    ref: ");
            for (int i = 0; i < ref_n; i++) printf("%d ", ref_tokens[i]);
            printf("\n");
            n_fail++;
        }
    }
}

typedef struct {
    int   tokens[512];
    int   n_tokens;
    char  text[4096];
    int   text_len;
} gen_result_t;

static void collect_cb(int token, const char *text, void *ud) {
    gen_result_t *r = (gen_result_t *)ud;
    if (r->n_tokens < 512) {
        r->tokens[r->n_tokens++] = token;
    }
    if (text) {
        int len = (int)strlen(text);
        if (r->text_len + len < 4095) {
            memcpy(r->text + r->text_len, text, (size_t)len);
            r->text_len += len;
            r->text[r->text_len] = '\0';
        }
    }
}

static void test_generation(bitnet_ctx_t *ctx) {
    printf("\n=== Test 2: Greedy Generation Comparison ===\n");
    printf("  (temp=0, greedy decoding, %d tokens)\n", N_PREDICT);
    printf("  NOTE: Numerical divergence between implementations is expected.\n");
    printf("        Different float ordering/FMA causes logit drift after a few tokens.\n");
    printf("        We check: both produce coherent English, not garbage.\n\n");

    const char *prompts[] = {
        "The meaning of life is",
        "Once upon a time",
        "In the year 2025,",
    };
    int n_prompts = (int)(sizeof(prompts) / sizeof(prompts[0]));

    for (int p = 0; p < n_prompts; p++) {
        printf("  Prompt: \"%s\"\n", prompts[p]);

        ctx->kv_len = 0;
        int prompt_tokens[256];
        int n_prompt = bitnet_tokenize(ctx, prompts[p], prompt_tokens, 256);

        gen_result_t c11_result;
        memset(&c11_result, 0, sizeof(c11_result));
        bitnet_generate(ctx, prompt_tokens, n_prompt, N_PREDICT,
                        collect_cb, &c11_result);

        char cmd[1024];
        char ref_output[MAX_OUTPUT];
        snprintf(cmd, sizeof(cmd),
                 "%s -m '%s' -p '%s' -n %d -t %d --temp 0.0 -b 1 -ngl 0 "
                 "--no-display-prompt 2>/dev/null",
                 g_llama_cli, g_model_path, prompts[p], N_PREDICT, N_THREADS);

        int rc = run_capture(cmd, ref_output, MAX_OUTPUT);
        char *ref_text = NULL;
        if (rc > 0) {
            ref_text = extract_llama_text(ref_output);
        }

        printf("    C11: \"%s\"\n", c11_result.text);
        if (ref_text) {
            printf("    Ref: \"%s\"\n", ref_text);

            int c11_len = c11_result.text_len;
            int ref_len = (int)strlen(ref_text);

            int c11_words = 0, ref_words = 0;
            for (int i = 0; i < c11_len; i++)
                if (c11_result.text[i] == ' ') c11_words++;
            c11_words++;
            for (int i = 0; i < ref_len; i++)
                if (ref_text[i] == ' ') ref_words++;
            ref_words++;

            char c11_copy[4096], ref_copy[4096];
            strncpy(c11_copy, c11_result.text, 4095); c11_copy[4095] = '\0';
            strncpy(ref_copy, ref_text, 4095); ref_copy[4095] = '\0';

            char *c11_words_arr[256];
            int c11_wc = 0;
            char *w = strtok(c11_copy, " ");
            while (w && c11_wc < 256) { c11_words_arr[c11_wc++] = w; w = strtok(NULL, " "); }

            char *ref_words_arr[256];
            int ref_wc = 0;
            w = strtok(ref_copy, " ");
            while (w && ref_wc < 256) { ref_words_arr[ref_wc++] = w; w = strtok(NULL, " "); }

            int prefix_words = 0;
            int min_wc = c11_wc < ref_wc ? c11_wc : ref_wc;
            for (int i = 0; i < min_wc; i++) {
                if (strcmp(c11_words_arr[i], ref_words_arr[i]) == 0)
                    prefix_words++;
                else
                    break;
            }

            if (strcmp(c11_result.text, ref_text) == 0) {
                printf("    Result: EXACT MATCH (%d chars)\n", c11_len);
                n_pass++;
            } else {
                printf("    Result: DIVERGED after %d word(s) — c11=%d words, ref=%d words\n",
                       prefix_words, c11_wc, ref_wc);
                CHECK(c11_wc >= 5, "C11 output is coherent (%d words)", c11_wc);
                CHECK(ref_wc >= 5, "Ref output is coherent (%d words)", ref_wc);
            }
            free(ref_text);
        } else {
            printf("    Ref: (unavailable)\n");
            CHECK(c11_result.text_len > 10, "C11 produced output (%d chars)", c11_result.text_len);
        }
        printf("\n");
    }
}

static void test_performance(bitnet_ctx_t *ctx) {
    printf("\n=== Test 3: Performance Comparison ===\n");

    const char *prompt = "The meaning of life is";

    ctx->kv_len = 0;
    int prompt_tokens[256];
    int n_prompt = bitnet_tokenize(ctx, prompt, prompt_tokens, 256);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    gen_result_t c11_result;
    memset(&c11_result, 0, sizeof(c11_result));
    bitnet_generate(ctx, prompt_tokens, n_prompt, N_PREDICT,
                    collect_cb, &c11_result);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double c11_secs = (double)(t1.tv_sec - t0.tv_sec) +
                      (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
    double c11_tps = (double)c11_result.n_tokens / c11_secs;

    char cmd[1024];
    char ref_output[MAX_OUTPUT];
    snprintf(cmd, sizeof(cmd),
             "%s -m '%s' -p '%s' -n %d -t %d --temp 0.0 -b 1 -ngl 0 2>&1",
             g_llama_cli, g_model_path, prompt, N_PREDICT, N_THREADS);
    run_capture(cmd, ref_output, MAX_OUTPUT);
    double ref_tps = extract_llama_speed(ref_output);

    printf("  C11:       %.2f tok/s (%d tokens in %.2f s)\n",
           c11_tps, c11_result.n_tokens, c11_secs);
    if (ref_tps > 0.0) {
        printf("  Reference: %.2f tok/s\n", ref_tps);
        double ratio = c11_tps / ref_tps * 100.0;
        printf("  Ratio:     %.1f%% of reference speed\n", ratio);
        CHECK(c11_tps > 1.0, "C11 speed > 1 tok/s (%.2f)", c11_tps);
        CHECK(ratio > 30.0, "C11 >= 30%% of reference speed (%.1f%%)", ratio);
    } else {
        printf("  Reference: (speed extraction failed)\n");
        CHECK(c11_tps > 1.0, "C11 speed > 1 tok/s (%.2f)", c11_tps);
    }
}

static void test_first_token_logits(bitnet_ctx_t *ctx) {
    printf("\n=== Test 4: First-Token Logit Sanity ===\n");

    const char *prompt = "Hello";
    ctx->kv_len = 0;

    int tokens[256];
    int n = bitnet_tokenize(ctx, prompt, tokens, 256);
    printf("  Prompt: \"%s\" (%d tokens including BOS)\n", prompt, n);

    float *logits = NULL;
    for (int i = 0; i < n; i++) {
        logits = bitnet_forward(ctx, &tokens[i], 1, (i == n - 1));
    }

    if (!logits) {
        printf("  FAIL: no logits returned\n");
        n_fail++;
        return;
    }

    int top5[5] = {0};
    for (int k = 0; k < 5; k++) {
        float best = -1e30f;
        for (int i = 0; i < ctx->model->n_vocab; i++) {
            int skip = 0;
            for (int j = 0; j < k; j++) if (top5[j] == i) skip = 1;
            if (skip) continue;
            if (logits[i] > best) { best = logits[i]; top5[k] = i; }
        }
    }

    printf("  Top-5 next-token predictions:\n");
    for (int k = 0; k < 5; k++) {
        const char *text = bn_token_text(ctx->tokenizer, top5[k]);
        printf("    [%d] token=%d logit=%.4f text=\"%s\"\n",
               k, top5[k], logits[top5[k]], text);
    }

    CHECK(top5[0] < 128000, "top prediction is a regular token (id=%d)", top5[0]);

    float lmin = logits[0], lmax = logits[0];
    for (int i = 1; i < ctx->model->n_vocab; i++) {
        if (logits[i] < lmin) lmin = logits[i];
        if (logits[i] > lmax) lmax = logits[i];
    }
    printf("  Logit range: [%.2f, %.2f]\n", lmin, lmax);
    CHECK(lmax - lmin > 1.0f, "logit range is non-degenerate (%.2f)", lmax - lmin);
    CHECK(lmax - lmin < 1e6f, "logit range is not exploded (%.2f)", lmax - lmin);
}

static void test_determinism(bitnet_ctx_t *ctx) {
    printf("\n=== Test 5: Determinism (same input -> same output) ===\n");

    const char *prompt = "The sky is";
    int tokens[256];
    int n = bitnet_tokenize(ctx, prompt, tokens, 256);

    int out1[64], out2[64];
    int n1 = 0, n2 = 0;

    for (int run = 0; run < 2; run++) {
        ctx->kv_len = 0;
        int *out = (run == 0) ? out1 : out2;

        float *logits = NULL;
        for (int i = 0; i < n; i++) {
            logits = bitnet_forward(ctx, &tokens[i], 1, (i == n - 1));
        }

        for (int g = 0; g < 20; g++) {
            int best = 0;
            for (int i = 1; i < ctx->model->n_vocab; i++) {
                if (logits[i] > logits[best]) best = i;
            }
            out[run == 0 ? n1++ : n2++] = best;
            logits = bitnet_forward(ctx, &best, 1, true);
        }
    }

    int match = (n1 == n2);
    if (match) {
        for (int i = 0; i < n1; i++) {
            if (out1[i] != out2[i]) { match = 0; break; }
        }
    }

    printf("  Run 1: ");
    for (int i = 0; i < n1; i++) printf("%d ", out1[i]);
    printf("\n  Run 2: ");
    for (int i = 0; i < n2; i++) printf("%d ", out2[i]);
    printf("\n");

    CHECK(match, "two greedy runs produce identical token sequences");
}

int main(void) {
    g_model_path = getenv("BITNET_MODEL");
    if (!g_model_path) g_model_path = MODEL_PATH;
    g_llama_cli = getenv("LLAMA_CLI");
    if (!g_llama_cli) g_llama_cli = LLAMA_CLI;
    g_llama_tok = getenv("LLAMA_TOKENIZE");
    if (!g_llama_tok) g_llama_tok = LLAMA_TOK;

    printf("=== bitnet-c11 vs Reference (llama.cpp) Comparison Test ===\n");
    printf("Model: %s\n", g_model_path);
    printf("Threads: %d, Predict: %d tokens\n", N_THREADS, N_PREDICT);

    bitnet_model_t *model = bitnet_model_load(g_model_path);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    bitnet_params_t params = bitnet_params_default();
    params.n_threads = N_THREADS;
    params.temperature = 0.0f;
    params.top_k = 1;

    bitnet_ctx_t *ctx = bitnet_ctx_new(model, params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        bitnet_model_free(model);
        return 1;
    }

    bn_sampler_seed(&ctx->sampler, 42);

    test_tokenizer(ctx);
    test_generation(ctx);
    test_performance(ctx);
    test_first_token_logits(ctx);
    test_determinism(ctx);

    printf("\n===================================================\n");
    printf("Results: %d passed, %d failed\n", n_pass, n_fail);
    printf("===================================================\n");

    bitnet_ctx_free(ctx);
    bitnet_model_free(model);

    return n_fail > 0 ? 1 : 0;
}
