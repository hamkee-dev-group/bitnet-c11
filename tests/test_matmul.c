#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static uint8_t *pack_test_weights(const int8_t *vals, int n_rows, int n_cols) {
    int row_bytes = n_cols / 4;
    uint8_t *packed = (uint8_t *)calloc(1, (size_t)n_rows * row_bytes + 32);

    for (int row = 0; row < n_rows; row++) {
        uint8_t *out = packed + (size_t)row * row_bytes;
        int col = 0;
        while (col < n_cols) {
            int blk = (n_cols - col >= 128) ? 128 : (n_cols - col);
            for (int j = 0; j < blk; j++) {
                int group_idx = j / 32;
                int group_pos = j % 32;
                int8_t v = vals[row * n_cols + col + j];
                uint8_t q;
                if (v == -1)     q = 0;
                else if (v == 0) q = 1;
                else             q = 2;
                out[group_pos] |= (q << (6 - 2 * group_idx));
            }
            out += 32;
            col += blk;
        }
    }
    return packed;
}

static void test_basic(void) {
    printf("Test: basic dot product (all +1 weights)...\n");
    int N = 128;
    int8_t *wvals = (int8_t *)calloc((size_t)N, sizeof(int8_t));
    int8_t *acts  = (int8_t *)calloc((size_t)N, sizeof(int8_t));

    for (int i = 0; i < N; i++) { wvals[i] = 1; acts[i] = 1; }

    uint8_t *packed = pack_test_weights(wvals, 1, N);

    float out_s, out_a;
    bn_i2s_gemv_scalar(packed, acts, &out_s, 1, N);
#if defined(__AVX2__)
    bn_i2s_gemv_avx2(packed, acts, &out_a, 1, N);
#else
    out_a = out_s;
#endif

    printf("  Scalar: raw=%.0f (expected %d), real=%.0f - %d = %.0f (expected %d)\n",
           out_s, 2 * N, out_s, N, out_s - (float)N, N);
    assert(fabsf(out_s - (float)(2 * N)) < 0.5f);
#if defined(__AVX2__)
    printf("  AVX2:   raw=%.0f (expected %d)\n", out_a, 2 * N);
    assert(fabsf(out_a - (float)(2 * N)) < 0.5f);
#endif

    free(wvals); free(acts); free(packed);
    printf("  OK\n");
}

static void test_mixed(void) {
    printf("Test: mixed weights {-1, 0, +1}...\n");
    int N = 256;
    int8_t *wvals = (int8_t *)calloc((size_t)N, sizeof(int8_t));
    int8_t *acts  = (int8_t *)calloc((size_t)N, sizeof(int8_t));

    for (int i = 0; i < N; i++) {
        wvals[i] = (int8_t)((i % 3) - 1);
        acts[i] = 3;
    }

    uint8_t *packed = pack_test_weights(wvals, 1, N);

    int expected_raw = 0;
    int expected_real = 0;
    for (int i = 0; i < N; i++) {
        int q;
        if (wvals[i] == -1) q = 0;
        else if (wvals[i] == 0) q = 1;
        else q = 2;
        expected_raw += q * 3;
        expected_real += wvals[i] * 3;
    }

    float out_s;
    bn_i2s_gemv_scalar(packed, acts, &out_s, 1, N);
    printf("  Scalar: raw=%.0f (expected %d), real=%.0f (expected %d)\n",
           out_s, expected_raw, out_s - (float)(N * 3), expected_real);
    assert(fabsf(out_s - (float)expected_raw) < 0.5f);

#if defined(__AVX2__)
    float out_a;
    bn_i2s_gemv_avx2(packed, acts, &out_a, 1, N);
    printf("  AVX2:   raw=%.0f (expected %d)\n", out_a, expected_raw);
    assert(fabsf(out_a - (float)expected_raw) < 0.5f);
#endif

    free(wvals); free(acts); free(packed);
    printf("  OK\n");
}

static void test_multirow(void) {
    printf("Test: multi-row GEMV...\n");
    int ROWS = 8, COLS = 256;
    int8_t *wvals = (int8_t *)calloc((size_t)ROWS * COLS, sizeof(int8_t));
    int8_t *acts  = (int8_t *)calloc((size_t)COLS, sizeof(int8_t));

    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < (r + 1) * 32 && c < COLS; c++) {
            wvals[r * COLS + c] = 1;
        }
    }
    for (int i = 0; i < COLS; i++) acts[i] = 2;

    uint8_t *packed = pack_test_weights(wvals, ROWS, COLS);
    float out_s[8], out_a[8];
    bn_i2s_gemv_scalar(packed, acts, out_s, ROWS, COLS);
#if defined(__AVX2__)
    bn_i2s_gemv_avx2(packed, acts, out_a, ROWS, COLS);
#endif

    for (int r = 0; r < ROWS; r++) {
        int n_active = (r + 1) * 32;
        if (n_active > COLS) n_active = COLS;
        int expected = n_active * 4 + (COLS - n_active) * 2;
        printf("  Row %d: scalar=%.0f avx2=%.0f expected=%d\n",
               r, out_s[r],
#if defined(__AVX2__)
               out_a[r],
#else
               out_s[r],
#endif
               expected);
        assert(fabsf(out_s[r] - (float)expected) < 0.5f);
#if defined(__AVX2__)
        assert(fabsf(out_a[r] - (float)expected) < 0.5f);
#endif
    }

    free(wvals); free(acts); free(packed);
    printf("  OK\n");
}

static void test_quantize(void) {
    printf("Test: activation quantization...\n");
    float input[128];
    int8_t output[128];
    float scale;
    int32_t sum;

    for (int i = 0; i < 128; i++) {
        input[i] = -1.0f + 2.0f * (float)i / 127.0f;
    }

    bn_quantize_acts(input, output, 128, &scale, &sum);

    printf("  scale=%.4f (expected ~127.0)\n", scale);
    assert(fabsf(scale - 127.0f) < 1.0f);
    assert(output[0] == -127);
    assert(output[127] == 127);
    printf("  q[0]=%d (expected -127), q[127]=%d (expected 127)\n",
           output[0], output[127]);
    printf("  OK\n");
}

static void test_i16_overflow(void) {
    printf("Test: i16 accumulator overflow (n_cols=4096, all +1 weights, acts=127)...\n");
    int N = 4096;
    int8_t *wvals = (int8_t *)malloc((size_t)N * sizeof(int8_t));
    int8_t *acts  = (int8_t *)malloc((size_t)N * sizeof(int8_t));

    for (int i = 0; i < N; i++) { wvals[i] = 1; acts[i] = 127; }

    uint8_t *packed = pack_test_weights(wvals, 1, N);

    float out_s;
    bn_i2s_gemv_scalar(packed, acts, &out_s, 1, N);
    printf("  Scalar: raw=%.0f\n", out_s);

#if defined(__AVX2__)
    float out_a;
    bn_i2s_gemv_avx2(packed, acts, &out_a, 1, N);
    printf("  AVX2:   raw=%.0f\n", out_a);
    assert(fabsf(out_a - out_s) < 0.5f);
#endif

    free(wvals); free(acts); free(packed);
    printf("  OK\n");
}

int main(void) {
    printf("=== Matmul Kernel Tests ===\n\n");
    test_basic();
    test_mixed();
    test_multirow();
    test_quantize();
    test_i16_overflow();
    printf("\n=== All matmul tests passed ===\n");
    return 0;
}
