#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TEST_VOCAB_SIZE = 256,
    TEST_THREAD_COUNT = 8,
    TEST_ITERATIONS = 200,
};

static uint32_t test_byte_to_codepoint(int byte) {
    if ((byte >= 0x21 && byte <= 0x7E) ||
        (byte >= 0xA1 && byte <= 0xAC) ||
        (byte >= 0xAE && byte <= 0xFF)) {
        return (uint32_t)byte;
    }

    int n = 0;
    for (int b = 0; b < byte; b++) {
        if (!((b >= 0x21 && b <= 0x7E) ||
              (b >= 0xA1 && b <= 0xAC) ||
              (b >= 0xAE && b <= 0xFF))) {
            n++;
        }
    }
    return (uint32_t)(256 + n);
}

static int test_codepoint_to_utf8(uint32_t cp, char *out) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }

    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

static char *test_make_byte_token(int byte) {
    char utf8[5];
    int len = test_codepoint_to_utf8(test_byte_to_codepoint(byte), utf8);
    char *token = (char *)malloc((size_t)len + 1);
    assert(token != NULL);
    memcpy(token, utf8, (size_t)len);
    token[len] = '\0';
    return token;
}

static bn_gguf_t *test_make_tokenizer_gguf(void) {
    bn_gguf_t *g = (bn_gguf_t *)calloc(1, sizeof(*g));
    assert(g != NULL);

    g->fd = -1;
    g->n_kv = 4;
    g->kvs = (bn_gguf_kv_t *)calloc((size_t)g->n_kv, sizeof(g->kvs[0]));
    assert(g->kvs != NULL);

    char **tokens = (char **)calloc(TEST_VOCAB_SIZE, sizeof(tokens[0]));
    char **merges = (char **)calloc(1, sizeof(merges[0]));
    assert(tokens != NULL);
    assert(merges != NULL);
    for (int i = 0; i < TEST_VOCAB_SIZE; i++) {
        tokens[i] = test_make_byte_token(i);
    }
    merges[0] = strdup("A A");
    assert(merges[0] != NULL);

    g->kvs[0].key = strdup("tokenizer.ggml.tokens");
    g->kvs[0].type = BN_GGUF_TYPE_ARRAY;
    g->kvs[0].val.arr.type = BN_GGUF_TYPE_STRING;
    g->kvs[0].val.arr.len = TEST_VOCAB_SIZE;
    g->kvs[0].val.arr.data = tokens;

    g->kvs[1].key = strdup("tokenizer.ggml.merges");
    g->kvs[1].type = BN_GGUF_TYPE_ARRAY;
    g->kvs[1].val.arr.type = BN_GGUF_TYPE_STRING;
    g->kvs[1].val.arr.len = 1;
    g->kvs[1].val.arr.data = merges;

    g->kvs[2].key = strdup("tokenizer.ggml.bos_token_id");
    g->kvs[2].type = BN_GGUF_TYPE_UINT32;
    g->kvs[2].val.u32 = TEST_VOCAB_SIZE;

    g->kvs[3].key = strdup("tokenizer.ggml.eos_token_id");
    g->kvs[3].type = BN_GGUF_TYPE_UINT32;
    g->kvs[3].val.u32 = TEST_VOCAB_SIZE + 1;

    assert(g->kvs[0].key != NULL);
    assert(g->kvs[1].key != NULL);
    assert(g->kvs[2].key != NULL);
    assert(g->kvs[3].key != NULL);
    return g;
}

static const char *const test_inputs[] = {
    "Hello",
    " thread-safe tokenizer",
    "ASCII only 12345 !?",
    "caf\xC3\xA9 au lait",
    "UTF-8 \xE2\x80\x94 \xF0\x9F\x98\x80",
};

typedef enum {
    TEST_WORK_TOKENIZE,
    TEST_WORK_DETOKENIZE,
} test_worker_mode_t;

typedef struct {
    pthread_barrier_t *start_barrier;
    bn_tokenizer_t *tokenizer;
    test_worker_mode_t mode;
    int thread_index;
} test_worker_arg_t;

static void test_long_single_span_round_trip(bn_tokenizer_t *tokenizer) {
    enum {
        TEST_LONG_SPAN_CODEPOINTS = 5000,
    };

    char *input = (char *)malloc((size_t)TEST_LONG_SPAN_CODEPOINTS + 1);
    int *tokens = (int *)malloc((size_t)(TEST_LONG_SPAN_CODEPOINTS + 1) * sizeof(tokens[0]));
    assert(input != NULL);
    assert(tokens != NULL);

    memset(input, 'a', (size_t)TEST_LONG_SPAN_CODEPOINTS);
    input[TEST_LONG_SPAN_CODEPOINTS] = '\0';

    int n = bn_tokenize(tokenizer, input, tokens, TEST_LONG_SPAN_CODEPOINTS + 1);
    assert(n > 1);

    char *decoded = bn_detokenize(tokenizer, tokens + 1, n - 1);
    assert(decoded != NULL);
    assert(strcmp(decoded, input) == 0);

    free(decoded);
    free(tokens);
    free(input);
}

static void *test_worker_main(void *arg_ptr) {
    test_worker_arg_t *arg = (test_worker_arg_t *)arg_ptr;
    (void)pthread_barrier_wait(arg->start_barrier);

    for (int iter = 0; iter < TEST_ITERATIONS; iter++) {
        const char *input =
            test_inputs[(iter + arg->thread_index) %
                        (int)(sizeof(test_inputs) / sizeof(test_inputs[0]))];

        if (arg->mode == TEST_WORK_TOKENIZE) {
            int tokens[256];
            int n = bn_tokenize(arg->tokenizer, input, tokens, 256);
            assert(n > 1);

            char *decoded = bn_detokenize(arg->tokenizer, tokens + 1, n - 1);
            assert(decoded != NULL);
            assert(strcmp(decoded, input) == 0);
            free(decoded);
        } else {
            size_t len = strlen(input);
            int tokens[256];
            assert(len < sizeof(tokens) / sizeof(tokens[0]));
            for (size_t i = 0; i < len; i++) {
                tokens[i] = (uint8_t)input[i];
            }

            char *decoded = bn_detokenize(arg->tokenizer, tokens, (int)len);
            assert(decoded != NULL);
            assert(strcmp(decoded, input) == 0);
            free(decoded);
        }
    }

    return NULL;
}

int main(void) {
    printf("=== Tokenizer Thread-Safety Test ===\n\n");

    bn_gguf_t *g = test_make_tokenizer_gguf();
    bn_tokenizer_t *tokenizer = bn_tokenizer_create(g);
    assert(tokenizer != NULL);

    test_long_single_span_round_trip(tokenizer);
    printf("Long single-span round-trip passed.\n");

    pthread_barrier_t start_barrier;
    assert(pthread_barrier_init(&start_barrier, NULL, TEST_THREAD_COUNT + 1) == 0);

    pthread_t threads[TEST_THREAD_COUNT];
    test_worker_arg_t args[TEST_THREAD_COUNT];

    for (int i = 0; i < TEST_THREAD_COUNT; i++) {
        args[i].start_barrier = &start_barrier;
        args[i].tokenizer = tokenizer;
        args[i].mode = (i % 2 == 0) ? TEST_WORK_TOKENIZE : TEST_WORK_DETOKENIZE;
        args[i].thread_index = i;
        assert(pthread_create(&threads[i], NULL, test_worker_main, &args[i]) == 0);
    }

    (void)pthread_barrier_wait(&start_barrier);

    for (int i = 0; i < TEST_THREAD_COUNT; i++) {
        assert(pthread_join(threads[i], NULL) == 0);
    }

    assert(pthread_barrier_destroy(&start_barrier) == 0);
    bn_tokenizer_free(tokenizer);
    bn_gguf_close(g);

    printf("Concurrent first-use encode/decode round-trips passed.\n");
    return 0;
}
