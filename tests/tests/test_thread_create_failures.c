#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_create_call_count = 0;
static int g_fail_on_call = -1;

static int test_pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                               void *(*start_routine)(void *), void *arg) {
    g_create_call_count++;
    if (g_create_call_count == g_fail_on_call) return EAGAIN;
    return pthread_create(thread, attr, start_routine, arg);
}

#define BN_PTHREAD_CREATE test_pthread_create
#include "../src/bitnet_core.c"

static uint8_t *pack_test_weights(const int8_t *vals, int n_rows, int n_cols) {
    int row_bytes = n_cols / 4;
    uint8_t *packed = (uint8_t *)calloc(1, (size_t)n_rows * row_bytes + 32);
    assert(packed != NULL);

    for (int row = 0; row < n_rows; row++) {
        uint8_t *out = packed + (size_t)row * row_bytes;
        int col = 0;
        while (col < n_cols) {
            int blk = (n_cols - col >= 128) ? 128 : (n_cols - col);
            for (int j = 0; j < blk; j++) {
                int group_idx = j / 32;
                int group_pos = j % 32;
                int8_t v = vals[row * n_cols + col + j];
                uint8_t q = (v == -1) ? 0 : (v == 0 ? 1 : 2);
                out[group_pos] |= (uint8_t)(q << (6 - 2 * group_idx));
            }
            out += 32;
            col += blk;
        }
    }
    return packed;
}

static char *capture_stderr(void (*fn)(void *), void *arg) {
    int saved_stderr = dup(STDERR_FILENO);
    assert(saved_stderr >= 0);

    FILE *tmp = tmpfile();
    assert(tmp != NULL);
    assert(dup2(fileno(tmp), STDERR_FILENO) >= 0);

    fn(arg);

    fflush(stderr);
    assert(dup2(saved_stderr, STDERR_FILENO) >= 0);
    close(saved_stderr);

    assert(fseek(tmp, 0, SEEK_END) == 0);
    long len = ftell(tmp);
    assert(len >= 0);
    assert(fseek(tmp, 0, SEEK_SET) == 0);

    char *buf = (char *)calloc((size_t)len + 1, 1);
    assert(buf != NULL);
    if (len > 0) {
        size_t nread = fread(buf, 1, (size_t)len, tmp);
        assert(nread == (size_t)len);
    }
    fclose(tmp);
    return buf;
}

typedef struct {
    const uint8_t *weights;
    const int8_t *acts;
    float *out;
    int n_rows;
    int n_cols;
    int n_threads;
} gemv_case_t;

static void run_gemv_case(void *arg) {
    gemv_case_t *tc = (gemv_case_t *)arg;
    g_create_call_count = 0;
    g_fail_on_call = 2;
    bn_gemv_mt(tc->weights, tc->acts, tc->out, tc->n_rows, tc->n_cols,
               bn_i2s_gemv_scalar, tc->n_threads);
    g_fail_on_call = -1;
}

static void test_gemv_thread_create_failure(void) {
    printf("Test: GEMV falls back when pthread_create fails...\n");

    enum { ROWS = 16, COLS = 256 };
    int8_t wvals[ROWS * COLS];
    int8_t acts[COLS];
    float expected[ROWS];
    float actual[ROWS];

    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            wvals[r * COLS + c] = (int8_t)(((r + c) % 3) - 1);
        }
    }
    for (int c = 0; c < COLS; c++) acts[c] = (int8_t)((c % 5) - 2);

    uint8_t *packed = pack_test_weights(wvals, ROWS, COLS);
    bn_i2s_gemv_scalar(packed, acts, expected, ROWS, COLS);

    gemv_case_t tc = { packed, acts, actual, ROWS, COLS, 4 };
    char *log = capture_stderr(run_gemv_case, &tc);

    assert(strstr(log, "bn_gemv_mt: pthread_create failed") != NULL);
    for (int i = 0; i < ROWS; i++) assert(actual[i] == expected[i]);

    free(log);
    free(packed);
    printf("  OK\n");
}

typedef struct {
    float *out;
    const float *w;
    const float *x;
    int n_out;
    int n_in;
    int n_threads;
} f32_case_t;

static void run_f32_case(void *arg) {
    f32_case_t *tc = (f32_case_t *)arg;
    g_create_call_count = 0;
    g_fail_on_call = 3;
    bn_matmul_f32(tc->out, tc->w, tc->x, tc->n_out, tc->n_in, tc->n_threads);
    g_fail_on_call = -1;
}

static void test_f32_thread_create_failure(void) {
    printf("Test: F32 matmul falls back when pthread_create fails...\n");

    enum { ROWS = 16, COLS = 16 };
    float w[ROWS * COLS];
    float x[COLS];
    float expected[ROWS];
    float actual[ROWS];

    for (int i = 0; i < ROWS * COLS; i++) w[i] = (float)((i % 7) - 3) * 0.25f;
    for (int i = 0; i < COLS; i++) x[i] = (float)(i - 4) * 0.5f;

    bn_matmul_f32(expected, w, x, ROWS, COLS, 1);

    f32_case_t tc = { actual, w, x, ROWS, COLS, 4 };
    char *log = capture_stderr(run_f32_case, &tc);

    assert(strstr(log, "bn_matmul_f32: pthread_create failed") != NULL);
    for (int i = 0; i < ROWS; i++) assert(actual[i] == expected[i]);

    free(log);
    printf("  OK\n");
}

int main(void) {
    printf("=== Thread Creation Failure Tests ===\n\n");
    test_gemv_thread_create_failure();
    test_f32_thread_create_failure();
    printf("\n=== All thread creation failure tests passed ===\n");
    return 0;
}
