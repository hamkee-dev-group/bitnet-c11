#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TEST_UTF8_TOKEN_COUNT = 3,
};

static char *test_dup_token(const unsigned char *bytes, size_t len) {
    char *token = (char *)malloc(len + 1);
    assert(token != NULL);
    memcpy(token, bytes, len);
    token[len] = '\0';
    return token;
}

static bn_gguf_t *test_make_malformed_tokenizer_gguf(void) {
    static const unsigned char token0[] = { 0xC2 };
    static const unsigned char token1[] = { 0xE2, 0x82 };
    static const unsigned char token2[] = { 0xF0, 0x9F, 0x92 };

    bn_gguf_t *g = (bn_gguf_t *)calloc(1, sizeof(*g));
    assert(g != NULL);

    g->fd = -1;
    g->n_kv = 4;
    g->kvs = (bn_gguf_kv_t *)calloc((size_t)g->n_kv, sizeof(g->kvs[0]));
    assert(g->kvs != NULL);

    char **tokens = (char **)calloc(TEST_UTF8_TOKEN_COUNT, sizeof(tokens[0]));
    char **merges = (char **)calloc(1, sizeof(merges[0]));
    assert(tokens != NULL);
    assert(merges != NULL);

    tokens[0] = test_dup_token(token0, sizeof(token0));
    tokens[1] = test_dup_token(token1, sizeof(token1));
    tokens[2] = test_dup_token(token2, sizeof(token2));

    g->kvs[0].key = strdup("tokenizer.ggml.tokens");
    g->kvs[0].type = BN_GGUF_TYPE_ARRAY;
    g->kvs[0].val.arr.type = BN_GGUF_TYPE_STRING;
    g->kvs[0].val.arr.len = TEST_UTF8_TOKEN_COUNT;
    g->kvs[0].val.arr.data = tokens;

    g->kvs[1].key = strdup("tokenizer.ggml.merges");
    g->kvs[1].type = BN_GGUF_TYPE_ARRAY;
    g->kvs[1].val.arr.type = BN_GGUF_TYPE_STRING;
    g->kvs[1].val.arr.len = 0;
    g->kvs[1].val.arr.data = merges;

    g->kvs[2].key = strdup("tokenizer.ggml.bos_token_id");
    g->kvs[2].type = BN_GGUF_TYPE_UINT32;
    g->kvs[2].val.u32 = TEST_UTF8_TOKEN_COUNT;

    g->kvs[3].key = strdup("tokenizer.ggml.eos_token_id");
    g->kvs[3].type = BN_GGUF_TYPE_UINT32;
    g->kvs[3].val.u32 = TEST_UTF8_TOKEN_COUNT + 1;

    assert(g->kvs[0].key != NULL);
    assert(g->kvs[1].key != NULL);
    assert(g->kvs[2].key != NULL);
    assert(g->kvs[3].key != NULL);
    return g;
}

int main(void) {
    static const unsigned char expected0[] = { 0xC2 };
    static const unsigned char expected1[] = { 0xE2, 0x82 };
    static const unsigned char expected2[] = { 0xF0, 0x9F, 0x92 };
    static const unsigned char expected_all[] = {
        0xC2, 0xE2, 0x82, 0xF0, 0x9F, 0x92
    };
    const int tokens[] = { 0, 1, 2 };

    printf("=== Tokenizer UTF-8 Guard Tests ===\n\n");

    bn_gguf_t *g = test_make_malformed_tokenizer_gguf();
    bn_tokenizer_t *tokenizer = bn_tokenizer_create(g);
    assert(tokenizer != NULL);

    printf("Test 1: Token text for truncated 2-byte sequence...\n");
    const char *text0 = bn_token_text(tokenizer, 0);
    assert(strlen(text0) == sizeof(expected0));
    assert(memcmp(text0, expected0, sizeof(expected0)) == 0);
    printf("  OK\n");

    printf("Test 2: Token text for truncated 3-byte sequence...\n");
    const char *text1 = bn_token_text(tokenizer, 1);
    assert(strlen(text1) == sizeof(expected1));
    assert(memcmp(text1, expected1, sizeof(expected1)) == 0);
    printf("  OK\n");

    printf("Test 3: Token text for truncated 4-byte sequence...\n");
    const char *text2 = bn_token_text(tokenizer, 2);
    assert(strlen(text2) == sizeof(expected2));
    assert(memcmp(text2, expected2, sizeof(expected2)) == 0);
    printf("  OK\n");

    printf("Test 4: Detokenize malformed UTF-8 tokens without overread...\n");
    char *decoded = bn_detokenize(tokenizer, tokens, TEST_UTF8_TOKEN_COUNT);
    assert(decoded != NULL);
    assert(strlen(decoded) == sizeof(expected_all));
    assert(memcmp(decoded, expected_all, sizeof(expected_all)) == 0);
    free(decoded);
    printf("  OK\n");

    bn_tokenizer_free(tokenizer);
    bn_gguf_close(g);

    printf("\n=== UTF-8 guard tests passed ===\n");
    return 0;
}
