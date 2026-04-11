#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t tokens_elem_type;
    uint32_t merges_elem_type;
    uint32_t bos_type;
    uint32_t eos_type;
} test_metadata_config_t;

static bn_gguf_t *test_make_tokenizer_gguf(const test_metadata_config_t *config) {
    static const char *const valid_tokens[] = { "A", "B", "C", "D" };
    static const char *const valid_merge = "A B";
    static const uint32_t invalid_u32_values[] = { 0, 1, 2, 3 };
    static const uint32_t invalid_merge_value[] = { 0 };
    bn_gguf_t *g = (bn_gguf_t *)calloc(1, sizeof(*g));
    char **tokens;
    char **merges;
    uint32_t *u32_tokens;
    uint32_t *u32_merges;

    assert(g != NULL);

    g->fd = -1;
    g->n_kv = 4;
    g->kvs = (bn_gguf_kv_t *)calloc((size_t)g->n_kv, sizeof(g->kvs[0]));
    assert(g->kvs != NULL);

    g->kvs[0].key = strdup("tokenizer.ggml.tokens");
    g->kvs[0].type = BN_GGUF_TYPE_ARRAY;
    g->kvs[0].val.arr.type = config->tokens_elem_type;
    g->kvs[0].val.arr.len = 4;
    assert(g->kvs[0].key != NULL);
    if (config->tokens_elem_type == BN_GGUF_TYPE_STRING) {
        tokens = (char **)calloc(4, sizeof(tokens[0]));
        assert(tokens != NULL);
        for (int i = 0; i < 4; i++) {
            tokens[i] = strdup(valid_tokens[i]);
            assert(tokens[i] != NULL);
        }
        g->kvs[0].val.arr.data = tokens;
    } else {
        u32_tokens = (uint32_t *)malloc(sizeof(invalid_u32_values));
        assert(u32_tokens != NULL);
        memcpy(u32_tokens, invalid_u32_values, sizeof(invalid_u32_values));
        g->kvs[0].val.arr.data = u32_tokens;
    }

    g->kvs[1].key = strdup("tokenizer.ggml.merges");
    g->kvs[1].type = BN_GGUF_TYPE_ARRAY;
    g->kvs[1].val.arr.type = config->merges_elem_type;
    g->kvs[1].val.arr.len = 1;
    assert(g->kvs[1].key != NULL);
    if (config->merges_elem_type == BN_GGUF_TYPE_STRING) {
        merges = (char **)calloc(1, sizeof(merges[0]));
        assert(merges != NULL);
        merges[0] = strdup(valid_merge);
        assert(merges[0] != NULL);
        g->kvs[1].val.arr.data = merges;
    } else {
        u32_merges = (uint32_t *)malloc(sizeof(invalid_merge_value));
        assert(u32_merges != NULL);
        memcpy(u32_merges, invalid_merge_value, sizeof(invalid_merge_value));
        g->kvs[1].val.arr.data = u32_merges;
    }

    g->kvs[2].key = strdup("tokenizer.ggml.bos_token_id");
    g->kvs[2].type = config->bos_type;
    assert(g->kvs[2].key != NULL);
    if (config->bos_type == BN_GGUF_TYPE_UINT32) {
        g->kvs[2].val.u32 = 0;
    } else {
        g->kvs[2].val.str.len = 3;
        g->kvs[2].val.str.data = strdup("bos");
        assert(g->kvs[2].val.str.data != NULL);
    }

    g->kvs[3].key = strdup("tokenizer.ggml.eos_token_id");
    g->kvs[3].type = config->eos_type;
    assert(g->kvs[3].key != NULL);
    if (config->eos_type == BN_GGUF_TYPE_UINT32) {
        g->kvs[3].val.u32 = 1;
    } else {
        g->kvs[3].val.str.len = 3;
        g->kvs[3].val.str.data = strdup("eos");
        assert(g->kvs[3].val.str.data != NULL);
    }

    return g;
}

static void run_malformed_case(const char *label,
                               const test_metadata_config_t *config) {
    bn_gguf_t *g = test_make_tokenizer_gguf(config);
    bn_tokenizer_t *t;

    t = bn_tokenizer_create(g);
    assert(t == NULL);
    bn_gguf_close(g);

    printf("%s: OK\n", label);
}

int main(void) {
    static const test_metadata_config_t malformed_tokens = {
        BN_GGUF_TYPE_UINT32,
        BN_GGUF_TYPE_STRING,
        BN_GGUF_TYPE_UINT32,
        BN_GGUF_TYPE_UINT32,
    };
    static const test_metadata_config_t malformed_merges = {
        BN_GGUF_TYPE_STRING,
        BN_GGUF_TYPE_UINT32,
        BN_GGUF_TYPE_UINT32,
        BN_GGUF_TYPE_UINT32,
    };
    static const test_metadata_config_t malformed_bos = {
        BN_GGUF_TYPE_STRING,
        BN_GGUF_TYPE_STRING,
        BN_GGUF_TYPE_STRING,
        BN_GGUF_TYPE_UINT32,
    };
    static const test_metadata_config_t malformed_eos = {
        BN_GGUF_TYPE_STRING,
        BN_GGUF_TYPE_STRING,
        BN_GGUF_TYPE_UINT32,
        BN_GGUF_TYPE_STRING,
    };

    printf("=== Tokenizer Metadata Validation Tests ===\n\n");

    run_malformed_case("Reject non-string tokenizer token arrays",
                       &malformed_tokens);
    run_malformed_case("Reject non-string tokenizer merge arrays",
                       &malformed_merges);
    run_malformed_case("Reject non-uint32 BOS token ids",
                       &malformed_bos);
    run_malformed_case("Reject non-uint32 EOS token ids",
                       &malformed_eos);

    /* Zero-length token array: should reject empty vocabulary */
    {
        bn_gguf_t *g = (bn_gguf_t *)calloc(1, sizeof(*g));
        assert(g != NULL);
        g->fd = -1;
        g->n_kv = 4;
        g->kvs = (bn_gguf_kv_t *)calloc((size_t)g->n_kv, sizeof(g->kvs[0]));
        assert(g->kvs != NULL);

        g->kvs[0].key = strdup("tokenizer.ggml.tokens");
        g->kvs[0].type = BN_GGUF_TYPE_ARRAY;
        g->kvs[0].val.arr.type = BN_GGUF_TYPE_STRING;
        g->kvs[0].val.arr.len = 0;
        g->kvs[0].val.arr.data = NULL;

        g->kvs[1].key = strdup("tokenizer.ggml.merges");
        g->kvs[1].type = BN_GGUF_TYPE_ARRAY;
        g->kvs[1].val.arr.type = BN_GGUF_TYPE_STRING;
        g->kvs[1].val.arr.len = 0;
        g->kvs[1].val.arr.data = NULL;

        g->kvs[2].key = strdup("tokenizer.ggml.bos_token_id");
        g->kvs[2].type = BN_GGUF_TYPE_UINT32;
        g->kvs[2].val.u32 = 0;

        g->kvs[3].key = strdup("tokenizer.ggml.eos_token_id");
        g->kvs[3].type = BN_GGUF_TYPE_UINT32;
        g->kvs[3].val.u32 = 1;

        bn_tokenizer_t *t = bn_tokenizer_create(g);
        assert(t == NULL);
        bn_gguf_close(g);
        printf("Reject zero-length token vocabulary: OK\n");
    }

    /* Zero-length merge array with valid tokens: should succeed */
    {
        static const char *const toks[] = { "A", "B", "C", "D" };
        bn_gguf_t *g = (bn_gguf_t *)calloc(1, sizeof(*g));
        assert(g != NULL);
        g->fd = -1;
        g->n_kv = 4;
        g->kvs = (bn_gguf_kv_t *)calloc((size_t)g->n_kv, sizeof(g->kvs[0]));
        assert(g->kvs != NULL);

        char **tokens = (char **)calloc(4, sizeof(tokens[0]));
        assert(tokens != NULL);
        for (int i = 0; i < 4; i++) {
            tokens[i] = strdup(toks[i]);
            assert(tokens[i] != NULL);
        }

        g->kvs[0].key = strdup("tokenizer.ggml.tokens");
        g->kvs[0].type = BN_GGUF_TYPE_ARRAY;
        g->kvs[0].val.arr.type = BN_GGUF_TYPE_STRING;
        g->kvs[0].val.arr.len = 4;
        g->kvs[0].val.arr.data = tokens;

        g->kvs[1].key = strdup("tokenizer.ggml.merges");
        g->kvs[1].type = BN_GGUF_TYPE_ARRAY;
        g->kvs[1].val.arr.type = BN_GGUF_TYPE_STRING;
        g->kvs[1].val.arr.len = 0;
        g->kvs[1].val.arr.data = NULL;

        g->kvs[2].key = strdup("tokenizer.ggml.bos_token_id");
        g->kvs[2].type = BN_GGUF_TYPE_UINT32;
        g->kvs[2].val.u32 = 0;

        g->kvs[3].key = strdup("tokenizer.ggml.eos_token_id");
        g->kvs[3].type = BN_GGUF_TYPE_UINT32;
        g->kvs[3].val.u32 = 1;

        bn_tokenizer_t *t = bn_tokenizer_create(g);
        assert(t != NULL);
        bn_tokenizer_free(t);
        bn_gguf_close(g);
        printf("Accept zero-length merge array with valid tokens: OK\n");
    }

    printf("\n=== Tokenizer metadata validation tests passed ===\n");
    return 0;
}
