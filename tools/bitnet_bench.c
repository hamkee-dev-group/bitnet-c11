#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s -m <model.gguf> [-t threads]\n"
            "       %s --micro [-t threads]\n",
            prog, prog);
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

/* ---- Microbenchmark helpers ---- */

static uint8_t *alloc_i2s_weights(int n_rows, int n_cols) {
    int row_bytes = bn_i2s_row_stride(n_cols);
    uint8_t *w = (uint8_t *)calloc(1, (size_t)n_rows * row_bytes);
    if (!w) return NULL;
    /* Fill with a repeating pattern so the kernel does real work */
    for (size_t i = 0; i < (size_t)n_rows * row_bytes; i++)
        w[i] = (uint8_t)(0x6A); /* encodes a mix of -1/0/+1 */
    return w;
}

static int8_t *alloc_i8_acts(int n) {
    int8_t *a = (int8_t *)malloc((size_t)n * sizeof(int8_t));
    if (!a) return NULL;
    for (int i = 0; i < n; i++)
        a[i] = (int8_t)(1 + (i % 5));
    return a;
}

static float *alloc_f32_matrix(int n_rows, int n_cols) {
    float *m = (float *)malloc((size_t)n_rows * n_cols * sizeof(float));
    if (!m) return NULL;
    for (size_t i = 0; i < (size_t)n_rows * n_cols; i++)
        m[i] = 0.01f * (float)(i % 97);
    return m;
}

static float *alloc_f32_vec(int n) {
    float *v = (float *)malloc((size_t)n * sizeof(float));
    if (!v) return NULL;
    for (int i = 0; i < n; i++)
        v[i] = 0.02f * (float)(i % 53);
    return v;
}

typedef struct {
    const char *label;
    int n_rows;
    int n_cols;
} bench_shape_t;

static void bench_i2s_gemv(const bench_shape_t *shape, int n_threads) {
    int n_rows = shape->n_rows;
    int n_cols = shape->n_cols;
    uint8_t *weights = alloc_i2s_weights(n_rows, n_cols);
    int8_t  *acts    = alloc_i8_acts(n_cols);
    float   *out     = (float *)calloc((size_t)n_rows, sizeof(float));
    if (!weights || !acts || !out) {
        fprintf(stderr, "  %s: allocation failed\n", shape->label);
        free(weights); free(acts); free(out);
        return;
    }

    bn_i2s_gemv_fn gemv;
#if defined(__AVX2__)
    gemv = bn_i2s_gemv_avx2;
#else
    gemv = bn_i2s_gemv_scalar;
#endif

    int warmup = 3;
    int iters  = 20;

    /* 1-thread timing */
    for (int i = 0; i < warmup; i++)
        gemv(weights, acts, out, n_rows, n_cols);

    double t0 = now_sec();
    for (int i = 0; i < iters; i++)
        gemv(weights, acts, out, n_rows, n_cols);
    double t1 = now_sec();
    double us_1t = (t1 - t0) / iters * 1e6;

    /* N-thread timing */
    double us_mt = us_1t;
    if (n_threads > 1) {
        bn_worker_pool_t *pool = bn_pool_create(n_threads);
        if (pool) {
            for (int i = 0; i < warmup; i++)
                bn_gemv_mt(weights, acts, out, n_rows, n_cols,
                           gemv, n_threads, pool);
            t0 = now_sec();
            for (int i = 0; i < iters; i++)
                bn_gemv_mt(weights, acts, out, n_rows, n_cols,
                           gemv, n_threads, pool);
            t1 = now_sec();
            us_mt = (t1 - t0) / iters * 1e6;
            bn_pool_free(pool);
        }
    }

    printf("  %-24s %4dx%-5d  1T: %8.1f us  %dT: %8.1f us  speedup: %.2fx\n",
           shape->label, n_rows, n_cols,
           us_1t, n_threads, us_mt, us_1t / us_mt);

    free(weights); free(acts); free(out);
}

static void bench_f32_matmul(const bench_shape_t *shape, int n_threads) {
    int n_rows = shape->n_rows;
    int n_cols = shape->n_cols;
    float *w   = alloc_f32_matrix(n_rows, n_cols);
    float *x   = alloc_f32_vec(n_cols);
    float *out = (float *)calloc((size_t)n_rows, sizeof(float));
    if (!w || !x || !out) {
        fprintf(stderr, "  %s: allocation failed\n", shape->label);
        free(w); free(x); free(out);
        return;
    }

    int warmup = 3;
    int iters  = 20;

    /* 1-thread timing */
    for (int i = 0; i < warmup; i++)
        bn_matmul_f32(out, w, x, n_rows, n_cols, 1, NULL);

    double t0 = now_sec();
    for (int i = 0; i < iters; i++)
        bn_matmul_f32(out, w, x, n_rows, n_cols, 1, NULL);
    double t1 = now_sec();
    double us_1t = (t1 - t0) / iters * 1e6;

    /* N-thread timing */
    double us_mt = us_1t;
    if (n_threads > 1) {
        bn_worker_pool_t *pool = bn_pool_create(n_threads);
        if (pool) {
            for (int i = 0; i < warmup; i++)
                bn_matmul_f32(out, w, x, n_rows, n_cols, n_threads, pool);
            t0 = now_sec();
            for (int i = 0; i < iters; i++)
                bn_matmul_f32(out, w, x, n_rows, n_cols, n_threads, pool);
            t1 = now_sec();
            us_mt = (t1 - t0) / iters * 1e6;
            bn_pool_free(pool);
        }
    }

    printf("  %-24s %4dx%-5d  1T: %8.1f us  %dT: %8.1f us  speedup: %.2fx\n",
           shape->label, n_rows, n_cols,
           us_1t, n_threads, us_mt, us_1t / us_mt);

    free(w); free(x); free(out);
}

static int run_micro(int n_threads) {
    printf("=== Kernel Microbenchmarks (threads=%d) ===\n\n", n_threads);

    /* Representative shapes from bitnet_forward():
     *   Q/K/V projection:  embd x embd   (768x768)
     *   FFN gate/up:       ff x embd     (2048x768)
     *   FFN down:          embd x ff     (768x2048)
     *   Output projection: vocab x embd  (32000x768)
     */
    bench_shape_t i2s_shapes[] = {
        { "attn_qkv (embd*embd)", 768,   768  },
        { "ffn_gate (ff*embd)",   2048,  768  },
        { "ffn_down (embd*ff)",   768,   2048 },
        { "output (vocab*embd)",  32000, 768  },
    };
    int n_i2s = (int)(sizeof(i2s_shapes) / sizeof(i2s_shapes[0]));

    printf("I2_S GEMV:\n");
    for (int i = 0; i < n_i2s; i++)
        bench_i2s_gemv(&i2s_shapes[i], n_threads);

    bench_shape_t f32_shapes[] = {
        { "output_f32 (vocab*embd)", 32000, 768 },
    };
    int n_f32 = (int)(sizeof(f32_shapes) / sizeof(f32_shapes[0]));

    printf("\nF32 matmul:\n");
    for (int i = 0; i < n_f32; i++)
        bench_f32_matmul(&f32_shapes[i], n_threads);

    printf("\n");
    return 0;
}

/* ---- Full-model benchmark ---- */

static int run_model_bench(const char *model_path, int n_threads) {
    fprintf(stderr, "Loading model: %s\n", model_path);
    bitnet_model_t *model = bitnet_model_load(model_path);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    bitnet_params_t params = bitnet_params_default();
    params.n_threads = n_threads;
    params.temperature = 0.0f;

    bitnet_ctx_t *ctx = bitnet_ctx_new(model, params);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }

    float *logits = NULL;
    int pp_tokens = 0;
    double pp_time = 0.0;
    int tg_tokens = 0;
    double tg_time = 0.0;

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
        pp_time = t1 - t0;
        pp_tokens = n;
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
        tg_time = t1 - t0;
        tg_tokens = gen_count;
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
    printf("  \"pp_tokens\": %d,\n", pp_tokens);
    printf("  \"pp_tok_s\": %.2f,\n", pp_time > 0 ? (double)pp_tokens / pp_time : 0.0);
    printf("  \"pp_attn_ms\": %.2f,\n", pp_attn_time * 1000);
    printf("  \"tg_tokens\": %d,\n", tg_tokens);
    printf("  \"tg_tok_s\": %.2f,\n", tg_time > 0 ? (double)tg_tokens / tg_time : 0.0);
    printf("  \"tg_attn_ms\": %.2f,\n", tg_attn_time * 1000);
    printf("  \"engine\": \"bitnet-c11\"\n");
    printf("}\n");

    bitnet_ctx_free(ctx);
    bitnet_model_free(model);
    return 0;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    int n_threads = 4;
    int micro = 0;

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
        } else if (strcmp(argv[i], "--micro") == 0) {
            micro = 1;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (micro) {
        return run_micro(n_threads);
    }

    if (!model_path) {
        print_usage(argv[0]);
        return 1;
    }

    return run_model_bench(model_path, n_threads);
}
