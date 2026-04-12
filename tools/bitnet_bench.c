#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s -m <model.gguf> [-t threads]\n", prog);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int parse_positive_int_option(const char *opt, const char *value,
                                     int *out) {
    char *end = NULL;
    long parsed;

    errno = 0;
    parsed = strtol(value, &end, 10);
    if (errno == ERANGE || end == value || *end != '\0' ||
        parsed <= 0 || parsed > INT_MAX) {
        fprintf(stderr, "Invalid value for %s: '%s' (must be a positive integer)\n",
                opt, value);
        return 0;
    }

    *out = (int)parsed;
    return 1;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    int n_threads = 4;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Option -m requires a value\n");
                print_usage(argv[0]);
                return 1;
            }
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Option -t requires a value\n");
                print_usage(argv[0]);
                return 1;
            }
            if (!parse_positive_int_option("-t", argv[++i], &n_threads)) {
                return 1;
            }
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path) {
        print_usage(argv[0]);
        return 1;
    }

    fprintf(stderr, "Loading model: %s\n", model_path);
    bitnet_model_t *model = bitnet_model_load(model_path);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    bitnet_params_t params = bitnet_params_default();
    params.n_threads = n_threads;
    params.temperature = 0.0f;

    bitnet_ctx_t *ctx = bitnet_ctx_new(model, params);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    float *logits = NULL;

    fprintf(stderr, "\n--- Prompt Processing Benchmark ---\n");
    double pp_attn_time = 0.0;
    {
        const char *prompt = "The quick brown fox jumps over the lazy dog. "
                             "This is a benchmark prompt designed to test "
                             "the speed of prompt processing in the bitnet-c11 "
                             "inference engine. We want to measure how quickly "
                             "the model can process a reasonable length input.";
        int tokens[512];
        int n = bitnet_tokenize(ctx, prompt, tokens, 512);
        fprintf(stderr, "Prompt length: %d tokens\n", n);

        bitnet_attn_time_reset(ctx);
        double t0 = now_sec();
        for (int i = 0; i < n; i++) {
            bool need_logits = (i == n - 1);
            int kv_before = ctx->kv_len;
            float *ret = bitnet_forward(ctx, &tokens[i], 1, need_logits);
            if ((need_logits && !ret) ||
                (!need_logits && ctx->kv_len == kv_before)) {
                fprintf(stderr,
                        "Prompt processing benchmark failed at token %d\n",
                        i);
                bitnet_ctx_free(ctx);
                bitnet_model_free(model);
                return 1;
            }
            if (ret) logits = ret;
        }
        double t1 = now_sec();
        double pp_time = t1 - t0;
        pp_attn_time = bitnet_attn_time_reset(ctx);
        fprintf(stderr, "Prompt processing: %.2f ms (%.1f tokens/sec)\n",
                pp_time * 1000, (double)n / pp_time);
        fprintf(stderr, "  attention time:  %.2f ms (%.1f%% of total)\n",
                pp_attn_time * 1000,
                pp_time > 0 ? pp_attn_time / pp_time * 100.0 : 0.0);
    }

    fprintf(stderr, "\n--- Token Generation Benchmark ---\n");
    double tg_attn_time = 0.0;
    {
        int gen_count = 0;
        int n_gen = 32;

        if (!logits) {
            fprintf(stderr, "Token generation benchmark has no prompt state\n");
            bitnet_ctx_free(ctx);
            bitnet_model_free(model);
            return 1;
        }

        bitnet_attn_time_reset(ctx);
        double t0 = now_sec();
        for (int i = 0; i < n_gen; i++) {
            int token = bitnet_sample_token(ctx, logits);
            gen_count++;
            logits = bitnet_forward(ctx, &token, 1, true);
            if (!logits) {
                fprintf(stderr,
                        "Token generation benchmark failed after %d tokens\n",
                        gen_count);
                bitnet_ctx_free(ctx);
                bitnet_model_free(model);
                return 1;
            }
        }

        double t1 = now_sec();
        double tg_time = t1 - t0;
        tg_attn_time = bitnet_attn_time_reset(ctx);
        fprintf(stderr, "Token generation: %d tokens in %.2f ms (%.2f tokens/sec)\n",
                gen_count, tg_time * 1000, (double)gen_count / tg_time);
        fprintf(stderr, "  attention time:  %.2f ms (%.1f%% of total)\n",
                tg_attn_time * 1000,
                tg_time > 0 ? tg_attn_time / tg_time * 100.0 : 0.0);
    }

    printf("{\n");
    printf("  \"model\": \"%s\",\n", model_path);
    printf("  \"threads\": %d,\n", n_threads);
    printf("  \"n_params_m\": %d,\n",
           (int)((long)model->n_embd * model->n_vocab / 1000000 +
                  (long)model->n_layer * (4 * (long)model->n_embd * model->n_embd +
                   3 * (long)model->n_ff * model->n_embd) / 1000000));
    printf("  \"pp_attn_ms\": %.2f,\n", pp_attn_time * 1000);
    printf("  \"tg_attn_ms\": %.2f,\n", tg_attn_time * 1000);
    printf("  \"engine\": \"bitnet-c11\"\n");
    printf("}\n");

    bitnet_ctx_free(ctx);
    bitnet_model_free(model);
    return 0;
}
