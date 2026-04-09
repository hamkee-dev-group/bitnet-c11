#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifndef MODEL_PATH
#define MODEL_PATH "models/ggml-model-i2_s.gguf"
#endif

int main(void) {
    const char *model_path = getenv("BITNET_MODEL");
    if (!model_path) model_path = MODEL_PATH;

    printf("=== GGUF Parser Tests ===\n\n");

    printf("Test 1: Load model... ");
    bn_gguf_t *g = bn_gguf_open(model_path);
    assert(g != NULL);
    printf("OK (v%u, %lu tensors, %lu kvs)\n",
           g->version, (unsigned long)g->n_tensors, (unsigned long)g->n_kv);

    printf("Test 2: Check metadata...\n");
    const char *arch = bn_gguf_get_str(g, "general.architecture");
    assert(arch != NULL);
    printf("  architecture: %s\n", arch);
    assert(strcmp(arch, "bitnet-b1.58") == 0);

    uint32_t n_layers = bn_gguf_get_u32(g, "bitnet-b1.58.block_count");
    printf("  block_count: %u\n", n_layers);
    assert(n_layers == 30);

    uint32_t n_embd = bn_gguf_get_u32(g, "bitnet-b1.58.embedding_length");
    printf("  embedding_length: %u\n", n_embd);
    assert(n_embd == 2560);

    uint32_t n_head = bn_gguf_get_u32(g, "bitnet-b1.58.attention.head_count");
    printf("  head_count: %u\n", n_head);
    assert(n_head == 20);

    printf("  OK\n");

    printf("Test 3: Check tensors...\n");
    bn_gguf_tensor_t *te = bn_gguf_find_tensor(g, "token_embd.weight");
    assert(te != NULL);
    printf("  token_embd: type=%u, shape=[%lu, %lu]\n",
           te->type, (unsigned long)te->ne[0], (unsigned long)te->ne[1]);
    assert(te->type == BN_GGML_TYPE_F16);
    assert(te->ne[0] == 2560);
    assert(te->ne[1] == 128256);

    bn_gguf_tensor_t *aq = bn_gguf_find_tensor(g, "blk.0.attn_q.weight");
    assert(aq != NULL);
    printf("  blk.0.attn_q: type=%u, shape=[%lu, %lu]\n",
           aq->type, (unsigned long)aq->ne[0], (unsigned long)aq->ne[1]);
    assert(aq->type == BN_GGML_TYPE_I2_S);
    assert(aq->ne[0] == 2560);
    assert(aq->ne[1] == 2560);

    int n_f32 = 0, n_f16 = 0, n_i2s = 0;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        switch (g->tensors[i].type) {
        case BN_GGML_TYPE_F32:  n_f32++; break;
        case BN_GGML_TYPE_F16:  n_f16++; break;
        case BN_GGML_TYPE_I2_S: n_i2s++; break;
        }
    }
    printf("  Type counts: F32=%d, F16=%d, I2_S=%d, total=%lu\n",
           n_f32, n_f16, n_i2s, (unsigned long)g->n_tensors);
    assert(n_i2s == 210);
    assert(n_f16 == 1);
    printf("  OK\n");

    printf("Test 4: All metadata keys:\n");
    for (uint64_t i = 0; i < g->n_kv; i++) {
        printf("  [%2lu] type=%2u key=%s\n",
               (unsigned long)i, g->kvs[i].type, g->kvs[i].key);
    }

    bn_gguf_close(g);

    printf("\n=== All GGUF tests passed ===\n");
    return 0;
}
