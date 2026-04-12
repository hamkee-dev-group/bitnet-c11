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
static bool g_fail_all = false;

static int test_pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                               void *(*start_routine)(void *), void *arg) {
    g_create_call_count++;
    if (g_fail_all || g_create_call_count == g_fail_on_call) return EAGAIN;
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

/* ---- Test: GEMV with NULL pool (sequential fallback) ---- */

static void test_gemv_null_pool(void) {
    printf("Test: GEMV falls back to sequential with NULL pool...\n");

    enum { ROWS = 16, COLS = 256 };
    int8_t wvals[ROWS * COLS];
    int8_t acts[COLS];
    float expected[ROWS];
    float actual[ROWS];

    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            wvals[r * COLS + c] = (int8_t)(((r + c) % 3) - 1);
    for (int c = 0; c < COLS; c++) acts[c] = (int8_t)((c % 5) - 2);

    uint8_t *packed = pack_test_weights(wvals, ROWS, COLS);
    bn_i2s_gemv_scalar(packed, acts, expected, ROWS, COLS);

    bn_gemv_mt(packed, acts, actual, ROWS, COLS, bn_i2s_gemv_scalar, 4, NULL);

    for (int i = 0; i < ROWS; i++) assert(actual[i] == expected[i]);

    free(packed);
    printf("  OK\n");
}

/* ---- Test: F32 matmul with NULL pool (sequential fallback) ---- */

static void test_f32_null_pool(void) {
    printf("Test: F32 matmul falls back to sequential with NULL pool...\n");

    enum { ROWS = 16, COLS = 16 };
    float w[ROWS * COLS];
    float x[COLS];
    float expected[ROWS];
    float actual[ROWS];

    for (int i = 0; i < ROWS * COLS; i++) w[i] = (float)((i % 7) - 3) * 0.25f;
    for (int i = 0; i < COLS; i++) x[i] = (float)(i - 4) * 0.5f;

    bn_matmul_f32(expected, w, x, ROWS, COLS, 1, NULL);
    bn_matmul_f32(actual, w, x, ROWS, COLS, 4, NULL);

    for (int i = 0; i < ROWS; i++) assert(actual[i] == expected[i]);

    printf("  OK\n");
}

/* ---- Test: pool creation total failure → NULL ---- */

typedef struct {
    bn_worker_pool_t *pool;
} pool_create_ctx_t;

static void run_pool_create_fail_all(void *arg) {
    pool_create_ctx_t *pc = (pool_create_ctx_t *)arg;
    g_create_call_count = 0;
    g_fail_all = true;
    pc->pool = bn_pool_create(4);
    g_fail_all = false;
}

static void test_pool_create_total_failure(void) {
    printf("Test: Pool creation returns NULL when all threads fail...\n");

    pool_create_ctx_t pc = { NULL };
    char *log = capture_stderr(run_pool_create_fail_all, &pc);

    assert(pc.pool == NULL);
    assert(strstr(log, "bn_pool_create: pthread_create failed") != NULL);

    free(log);
    printf("  OK\n");
}

/* ---- Test: pool creation partial failure → reduced pool works ---- */

static void run_pool_create_partial(void *arg) {
    pool_create_ctx_t *pc = (pool_create_ctx_t *)arg;
    g_create_call_count = 0;
    g_fail_on_call = 2;
    pc->pool = bn_pool_create(4);
    g_fail_on_call = -1;
}

static void test_pool_partial_failure(void) {
    printf("Test: Pool works with partial thread creation failure...\n");

    pool_create_ctx_t pc = { NULL };
    char *log = capture_stderr(run_pool_create_partial, &pc);

    assert(pc.pool != NULL);
    assert(pc.pool->n_workers < 3);
    assert(strstr(log, "bn_pool_create: pthread_create failed") != NULL);

    /* Use the degraded pool for GEMV and verify correctness */
    enum { ROWS = 16, COLS = 256 };
    int8_t wvals[ROWS * COLS];
    int8_t acts[COLS];
    float expected[ROWS];
    float actual[ROWS];

    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            wvals[r * COLS + c] = (int8_t)(((r + c) % 3) - 1);
    for (int c = 0; c < COLS; c++) acts[c] = (int8_t)((c % 5) - 2);

    uint8_t *packed = pack_test_weights(wvals, ROWS, COLS);
    bn_i2s_gemv_scalar(packed, acts, expected, ROWS, COLS);

    bn_gemv_mt(packed, acts, actual, ROWS, COLS,
               bn_i2s_gemv_scalar, 4, pc.pool);

    for (int i = 0; i < ROWS; i++) assert(actual[i] == expected[i]);

    free(packed);
    bn_pool_free(pc.pool);
    free(log);
    printf("  OK\n");
}

/* ---- Test: pool lifecycle + GEMV correctness ---- */

static void test_pool_gemv(void) {
    printf("Test: GEMV works correctly with worker pool...\n");

    g_create_call_count = 0;
    g_fail_on_call = -1;
    g_fail_all = false;
    bn_worker_pool_t *pool = bn_pool_create(4);
    assert(pool != NULL);
    assert(pool->n_workers == 3);

    enum { ROWS = 16, COLS = 256 };
    int8_t wvals[ROWS * COLS];
    int8_t acts[COLS];
    float expected[ROWS];
    float actual[ROWS];

    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++)
            wvals[r * COLS + c] = (int8_t)(((r + c) % 3) - 1);
    for (int c = 0; c < COLS; c++) acts[c] = (int8_t)((c % 5) - 2);

    uint8_t *packed = pack_test_weights(wvals, ROWS, COLS);
    bn_i2s_gemv_scalar(packed, acts, expected, ROWS, COLS);

    /* Run twice to verify pool reuse */
    for (int iter = 0; iter < 2; iter++) {
        memset(actual, 0, sizeof(actual));
        bn_gemv_mt(packed, acts, actual, ROWS, COLS,
                   bn_i2s_gemv_scalar, 4, pool);
        for (int i = 0; i < ROWS; i++) assert(actual[i] == expected[i]);
    }

    free(packed);
    bn_pool_free(pool);
    printf("  OK\n");
}

/* ---- Test: pool lifecycle + F32 matmul correctness ---- */

static void test_pool_f32(void) {
    printf("Test: F32 matmul works correctly with worker pool...\n");

    g_create_call_count = 0;
    g_fail_on_call = -1;
    g_fail_all = false;
    bn_worker_pool_t *pool = bn_pool_create(4);
    assert(pool != NULL);

    enum { ROWS = 16, COLS = 16 };
    float w[ROWS * COLS];
    float x[COLS];
    float expected[ROWS];
    float actual[ROWS];

    for (int i = 0; i < ROWS * COLS; i++) w[i] = (float)((i % 7) - 3) * 0.25f;
    for (int i = 0; i < COLS; i++) x[i] = (float)(i - 4) * 0.5f;

    bn_matmul_f32(expected, w, x, ROWS, COLS, 1, NULL);

    /* Run twice to verify pool reuse */
    for (int iter = 0; iter < 2; iter++) {
        memset(actual, 0, sizeof(actual));
        bn_matmul_f32(actual, w, x, ROWS, COLS, 4, pool);
        for (int i = 0; i < ROWS; i++) assert(actual[i] == expected[i]);
    }

    bn_pool_free(pool);
    printf("  OK\n");
}

int main(void) {
    printf("=== Thread Creation Failure Tests ===\n\n");
    test_gemv_null_pool();
    test_f32_null_pool();
    test_pool_create_total_failure();
    test_pool_partial_failure();
    test_pool_gemv();
    test_pool_f32();
    printf("\n=== All thread creation failure tests passed ===\n");
    return 0;
}
