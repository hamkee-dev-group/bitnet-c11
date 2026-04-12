#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * Micro-benchmark: bn_rope with precomputed cos/sin tables vs.
 * the original per-head powf/cosf/sinf implementation.
 */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* Original (slow) implementation — recomputes powf/cosf/sinf per head. */
static void rope_original(float *q, int dim, int head_dim, int pos,
                          float freq_base) {
    int n_heads = dim / head_dim;
    for (int h = 0; h < n_heads; h++) {
        float *v = q + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq  = 1.0f / powf(freq_base, (float)i / (float)head_dim);
            float theta = (float)pos * freq;
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);
            float v0 = v[i];
            float v1 = v[i + 1];
            v[i]     = v0 * cos_t - v1 * sin_t;
            v[i + 1] = v0 * sin_t + v1 * cos_t;
        }
    }
}

/* New (fast) implementation — precomputes tables once per call. */
static void rope_precomputed(float *q, int dim, int head_dim, int pos,
                             float freq_base) {
    int n_heads = dim / head_dim;
    int half = head_dim / 2;

    float cos_tab[half];
    float sin_tab[half];
    for (int j = 0; j < half; j++) {
        float freq  = 1.0f / powf(freq_base, (float)(2 * j) / (float)head_dim);
        float theta = (float)pos * freq;
        cos_tab[j] = cosf(theta);
        sin_tab[j] = sinf(theta);
    }

    for (int h = 0; h < n_heads; h++) {
        float *v = q + h * head_dim;
        for (int j = 0; j < half; j++) {
            float v0 = v[2 * j];
            float v1 = v[2 * j + 1];
            v[2 * j]     = v0 * cos_tab[j] - v1 * sin_tab[j];
            v[2 * j + 1] = v0 * sin_tab[j] + v1 * cos_tab[j];
        }
    }
}

int main(void) {
    /* Typical BitNet b1.58 700M parameters. */
    const int n_heads  = 20;
    const int head_dim = 128;
    const int dim      = n_heads * head_dim;
    const float freq_base = 10000.0f;
    const int iters    = 100000;

    float *buf_orig = malloc((size_t)dim * sizeof(float));
    float *buf_pre  = malloc((size_t)dim * sizeof(float));
    if (!buf_orig || !buf_pre) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    /* Fill with deterministic data. */
    for (int i = 0; i < dim; i++) {
        buf_orig[i] = (float)(i % 37) * 0.01f;
        buf_pre[i]  = buf_orig[i];
    }

    /* Correctness check: both must produce the same output. */
    rope_original(buf_orig, dim, head_dim, 42, freq_base);
    rope_precomputed(buf_pre, dim, head_dim, 42, freq_base);
    float max_err = 0.0f;
    for (int i = 0; i < dim; i++) {
        float err = fabsf(buf_orig[i] - buf_pre[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max absolute error: %.2e (should be ~0)\n", (double)max_err);
    if (max_err > 1e-5f) {
        fprintf(stderr, "ERROR: implementations diverge!\n");
        free(buf_orig);
        free(buf_pre);
        return 1;
    }

    /* Benchmark original. */
    for (int i = 0; i < dim; i++) buf_orig[i] = (float)(i % 37) * 0.01f;
    double t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        rope_original(buf_orig, dim, head_dim, it, freq_base);
    }
    double t_orig = now_sec() - t0;

    /* Benchmark precomputed. */
    for (int i = 0; i < dim; i++) buf_pre[i] = (float)(i % 37) * 0.01f;
    t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        rope_precomputed(buf_pre, dim, head_dim, it, freq_base);
    }
    double t_pre = now_sec() - t0;

    printf("Original:     %.2f ms (%d iters)\n", t_orig * 1000.0, iters);
    printf("Precomputed:  %.2f ms (%d iters)\n", t_pre * 1000.0, iters);
    printf("Speedup:      %.2fx\n", t_orig / t_pre);

    free(buf_orig);
    free(buf_pre);
    return 0;
}
