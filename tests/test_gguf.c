#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>

#ifndef MODEL_PATH
#define MODEL_PATH "models/ggml-model-i2_s.gguf"
#endif

static void write_u32(FILE *fp, uint32_t value) {
    assert(fwrite(&value, sizeof(value), 1, fp) == 1);
}

static void write_u64(FILE *fp, uint64_t value) {
    assert(fwrite(&value, sizeof(value), 1, fp) == 1);
}

static void write_str(FILE *fp, const char *value) {
    uint64_t len = strlen(value);
    write_u64(fp, len);
    assert(fwrite(value, 1, len, fp) == len);
}

static void write_padding(FILE *fp, size_t alignment) {
    long pos = ftell(fp);
    assert(pos >= 0);
    size_t pad = (alignment - ((size_t)pos % alignment)) % alignment;
    static const unsigned char zeros[32] = {0};
    assert(pad <= sizeof(zeros));
    if (pad > 0) assert(fwrite(zeros, 1, pad, fp) == pad);
}

typedef struct {
    const char *name;
    uint32_t type;
    uint32_t n_dims;
    uint64_t ne[2];
} fixture_tensor_t;

static size_t fixture_tensor_nbytes(const fixture_tensor_t *t) {
    size_t n_elem = 1;
    for (uint32_t i = 0; i < t->n_dims; i++) {
        n_elem *= (size_t)t->ne[i];
    }

    switch (t->type) {
    case BN_GGML_TYPE_F32:
        return n_elem * sizeof(float);
    case BN_GGML_TYPE_I2_S:
        return (n_elem / 4) + sizeof(float);
    default:
        assert(!"unsupported fixture tensor type");
        return 0;
    }
}

static void write_fixture_tensor_data(FILE *fp, const fixture_tensor_t *t) {
    size_t nbytes = fixture_tensor_nbytes(t);
    if (t->type == BN_GGML_TYPE_F32) {
        size_t n_elem = nbytes / sizeof(float);
        for (size_t i = 0; i < n_elem; i++) {
            float value = 1.0f + (float)i;
            assert(fwrite(&value, sizeof(value), 1, fp) == 1);
        }
        return;
    }

    size_t packed_bytes = nbytes - sizeof(float);
    for (size_t i = 0; i < packed_bytes; i++) {
        uint8_t value = (uint8_t)(i & 0xffu);
        assert(fwrite(&value, sizeof(value), 1, fp) == 1);
    }
    {
        float weight_scale = 1.0f;
        assert(fwrite(&weight_scale, sizeof(weight_scale), 1, fp) == 1);
    }
}

static uint64_t align_up(uint64_t val, uint64_t alignment) {
    return ((val + alignment - 1) / alignment) * alignment;
}

static void write_zero_padding_to(FILE *fp, uint64_t target_offset,
                                   uint64_t data_section_start) {
    long cur = ftell(fp);
    assert(cur >= 0);
    uint64_t cur_data_rel = (uint64_t)cur - data_section_start;
    if (cur_data_rel < target_offset) {
        uint64_t pad = target_offset - cur_data_rel;
        static const unsigned char zeros[64] = {0};
        while (pad > 0) {
            uint64_t chunk = pad < sizeof(zeros) ? pad : sizeof(zeros);
            assert(fwrite(zeros, 1, (size_t)chunk, fp) == (size_t)chunk);
            pad -= chunk;
        }
    }
}

static void create_minimal_model_fixture(char path_template[],
                                         const char *missing_tensor) {
    static const fixture_tensor_t tensors[] = {
        { "token_embd.weight",        BN_GGML_TYPE_F32, 2, { 4, 4 } },
        { "output_norm.weight",       BN_GGML_TYPE_F32, 1, { 4, 1 } },
        { "blk.0.attn_norm.weight",   BN_GGML_TYPE_F32, 1, { 4, 1 } },
        { "blk.0.attn_sub_norm.weight", BN_GGML_TYPE_F32, 1, { 4, 1 } },
        { "blk.0.ffn_norm.weight",    BN_GGML_TYPE_F32, 1, { 4, 1 } },
        { "blk.0.ffn_sub_norm.weight", BN_GGML_TYPE_F32, 1, { 4, 1 } },
        { "blk.0.attn_q.weight",      BN_GGML_TYPE_I2_S, 2, { 4, 4 } },
        { "blk.0.attn_q.weight.scale", BN_GGML_TYPE_F32, 1, { 1, 1 } },
        { "blk.0.attn_k.weight",      BN_GGML_TYPE_I2_S, 2, { 4, 4 } },
        { "blk.0.attn_k.weight.scale", BN_GGML_TYPE_F32, 1, { 1, 1 } },
        { "blk.0.attn_v.weight",      BN_GGML_TYPE_I2_S, 2, { 4, 4 } },
        { "blk.0.attn_v.weight.scale", BN_GGML_TYPE_F32, 1, { 1, 1 } },
        { "blk.0.attn_output.weight", BN_GGML_TYPE_I2_S, 2, { 4, 4 } },
        { "blk.0.attn_output.weight.scale", BN_GGML_TYPE_F32, 1, { 1, 1 } },
        { "blk.0.ffn_gate.weight",    BN_GGML_TYPE_I2_S, 2, { 4, 4 } },
        { "blk.0.ffn_gate.weight.scale", BN_GGML_TYPE_F32, 1, { 1, 1 } },
        { "blk.0.ffn_up.weight",      BN_GGML_TYPE_I2_S, 2, { 4, 4 } },
        { "blk.0.ffn_up.weight.scale", BN_GGML_TYPE_F32, 1, { 1, 1 } },
        { "blk.0.ffn_down.weight",    BN_GGML_TYPE_I2_S, 2, { 4, 4 } },
        { "blk.0.ffn_down.weight.scale", BN_GGML_TYPE_F32, 1, { 1, 1 } },
    };
    const size_t n_tensors_total = sizeof(tensors) / sizeof(tensors[0]);
    size_t n_tensors = 0;
    uint64_t offset = 0;

    int fd = mkstemp(path_template);
    assert(fd >= 0);

    FILE *fp = fdopen(fd, "wb");
    assert(fp != NULL);

    for (size_t i = 0; i < n_tensors_total; i++) {
        if (missing_tensor && strcmp(tensors[i].name, missing_tensor) == 0) continue;
        n_tensors++;
    }

    assert(fwrite("GGUF", 1, 4, fp) == 4);
    write_u32(fp, 3);
    write_u64(fp, n_tensors);
    write_u64(fp, 11);

    write_str(fp, "general.architecture");
    write_u32(fp, BN_GGUF_TYPE_STRING);
    write_str(fp, "bitnet-b1.58");

    write_str(fp, "general.alignment");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 32);

    write_str(fp, "bitnet-b1.58.vocab_size");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 4);

    write_str(fp, "bitnet-b1.58.embedding_length");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 4);

    write_str(fp, "bitnet-b1.58.block_count");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 1);

    write_str(fp, "bitnet-b1.58.attention.head_count");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 1);

    write_str(fp, "bitnet-b1.58.attention.head_count_kv");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 1);

    write_str(fp, "bitnet-b1.58.feed_forward_length");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 4);

    write_str(fp, "bitnet-b1.58.context_length");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 8);

    write_str(fp, "bitnet-b1.58.attention.layer_norm_rms_epsilon");
    write_u32(fp, BN_GGUF_TYPE_FLOAT32);
    {
        float eps = 1e-5f;
        assert(fwrite(&eps, sizeof(eps), 1, fp) == 1);
    }

    write_str(fp, "bitnet-b1.58.rope.freq_base");
    write_u32(fp, BN_GGUF_TYPE_FLOAT32);
    {
        float freq_base = 10000.0f;
        assert(fwrite(&freq_base, sizeof(freq_base), 1, fp) == 1);
    }

    uint64_t offsets[sizeof(tensors)/sizeof(tensors[0])];
    for (size_t i = 0; i < n_tensors_total; i++) {
        const fixture_tensor_t *t = &tensors[i];
        if (missing_tensor && strcmp(t->name, missing_tensor) == 0) continue;

        offsets[i] = offset;
        write_str(fp, t->name);
        write_u32(fp, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++) {
            write_u64(fp, t->ne[d]);
        }
        write_u32(fp, t->type);
        write_u64(fp, offset);
        offset += fixture_tensor_nbytes(t);
        offset = align_up(offset, 32);
    }

    write_padding(fp, 32);
    uint64_t data_start = (uint64_t)ftell(fp);

    for (size_t i = 0; i < n_tensors_total; i++) {
        if (missing_tensor && strcmp(tensors[i].name, missing_tensor) == 0) continue;
        write_zero_padding_to(fp, offsets[i], data_start);
        write_fixture_tensor_data(fp, &tensors[i]);
    }

    assert(fclose(fp) == 0);
}

static void create_minimal_context_model_fixture(char path_template[]) {
    static const fixture_tensor_t tensors[] = {
        { "token_embd.weight",        BN_GGML_TYPE_F32, 2, { 4, 4 } },
        { "output_norm.weight",       BN_GGML_TYPE_F32, 1, { 4, 1 } },
        { "blk.0.attn_norm.weight",   BN_GGML_TYPE_F32, 1, { 4, 1 } },
        { "blk.0.attn_sub_norm.weight", BN_GGML_TYPE_F32, 1, { 4, 1 } },
        { "blk.0.ffn_norm.weight",    BN_GGML_TYPE_F32, 1, { 4, 1 } },
        { "blk.0.ffn_sub_norm.weight", BN_GGML_TYPE_F32, 1, { 4, 1 } },
        { "blk.0.attn_q.weight",      BN_GGML_TYPE_I2_S, 2, { 4, 4 } },
        { "blk.0.attn_q.weight.scale", BN_GGML_TYPE_F32, 1, { 1, 1 } },
        { "blk.0.attn_k.weight",      BN_GGML_TYPE_I2_S, 2, { 4, 4 } },
        { "blk.0.attn_k.weight.scale", BN_GGML_TYPE_F32, 1, { 1, 1 } },
        { "blk.0.attn_v.weight",      BN_GGML_TYPE_I2_S, 2, { 4, 4 } },
        { "blk.0.attn_v.weight.scale", BN_GGML_TYPE_F32, 1, { 1, 1 } },
        { "blk.0.attn_output.weight", BN_GGML_TYPE_I2_S, 2, { 4, 4 } },
        { "blk.0.attn_output.weight.scale", BN_GGML_TYPE_F32, 1, { 1, 1 } },
        { "blk.0.ffn_gate.weight",    BN_GGML_TYPE_I2_S, 2, { 4, 4 } },
        { "blk.0.ffn_gate.weight.scale", BN_GGML_TYPE_F32, 1, { 1, 1 } },
        { "blk.0.ffn_up.weight",      BN_GGML_TYPE_I2_S, 2, { 4, 4 } },
        { "blk.0.ffn_up.weight.scale", BN_GGML_TYPE_F32, 1, { 1, 1 } },
        { "blk.0.ffn_down.weight",    BN_GGML_TYPE_I2_S, 2, { 4, 4 } },
        { "blk.0.ffn_down.weight.scale", BN_GGML_TYPE_F32, 1, { 1, 1 } },
    };
    static const char *tokens[] = { "a", "b", "c", "d" };
    size_t n_tensors = sizeof(tensors) / sizeof(tensors[0]);
    uint64_t offset = 0;

    int fd = mkstemp(path_template);
    assert(fd >= 0);

    FILE *fp = fdopen(fd, "wb");
    assert(fp != NULL);

    assert(fwrite("GGUF", 1, 4, fp) == 4);
    write_u32(fp, 3);
    write_u64(fp, n_tensors);
    write_u64(fp, 15);

    write_str(fp, "general.architecture");
    write_u32(fp, BN_GGUF_TYPE_STRING);
    write_str(fp, "bitnet-b1.58");

    write_str(fp, "general.alignment");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 32);

    write_str(fp, "bitnet-b1.58.vocab_size");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 4);

    write_str(fp, "bitnet-b1.58.embedding_length");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 4);

    write_str(fp, "bitnet-b1.58.block_count");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 1);

    write_str(fp, "bitnet-b1.58.attention.head_count");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 1);

    write_str(fp, "bitnet-b1.58.attention.head_count_kv");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 1);

    write_str(fp, "bitnet-b1.58.feed_forward_length");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 4);

    write_str(fp, "bitnet-b1.58.context_length");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 8);

    write_str(fp, "bitnet-b1.58.attention.layer_norm_rms_epsilon");
    write_u32(fp, BN_GGUF_TYPE_FLOAT32);
    {
        float eps = 1e-5f;
        assert(fwrite(&eps, sizeof(eps), 1, fp) == 1);
    }

    write_str(fp, "bitnet-b1.58.rope.freq_base");
    write_u32(fp, BN_GGUF_TYPE_FLOAT32);
    {
        float freq_base = 10000.0f;
        assert(fwrite(&freq_base, sizeof(freq_base), 1, fp) == 1);
    }

    write_str(fp, "tokenizer.ggml.tokens");
    write_u32(fp, BN_GGUF_TYPE_ARRAY);
    write_u32(fp, BN_GGUF_TYPE_STRING);
    write_u64(fp, 4);
    for (size_t i = 0; i < 4; i++) write_str(fp, tokens[i]);

    write_str(fp, "tokenizer.ggml.merges");
    write_u32(fp, BN_GGUF_TYPE_ARRAY);
    write_u32(fp, BN_GGUF_TYPE_STRING);
    write_u64(fp, 0);

    write_str(fp, "tokenizer.ggml.bos_token_id");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 0);

    write_str(fp, "tokenizer.ggml.eos_token_id");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 1);

    uint64_t offsets[sizeof(tensors)/sizeof(tensors[0])];
    for (size_t i = 0; i < n_tensors; i++) {
        const fixture_tensor_t *t = &tensors[i];
        offsets[i] = offset;
        write_str(fp, t->name);
        write_u32(fp, t->n_dims);
        for (uint32_t d = 0; d < t->n_dims; d++) {
            write_u64(fp, t->ne[d]);
        }
        write_u32(fp, t->type);
        write_u64(fp, offset);
        offset += fixture_tensor_nbytes(t);
        offset = align_up(offset, 32);
    }

    write_padding(fp, 32);
    uint64_t data_start = (uint64_t)ftell(fp);

    for (size_t i = 0; i < n_tensors; i++) {
        write_zero_padding_to(fp, offsets[i], data_start);
        write_fixture_tensor_data(fp, &tensors[i]);
    }

    assert(fclose(fp) == 0);
}

static void create_valid_gguf_fixture(char path_template[]) {
    int fd = mkstemp(path_template);
    assert(fd >= 0);

    FILE *fp = fdopen(fd, "wb");
    assert(fp != NULL);

    assert(fwrite("GGUF", 1, 4, fp) == 4);
    write_u32(fp, 3);
    write_u64(fp, 1);
    write_u64(fp, 2);

    write_str(fp, "general.architecture");
    write_u32(fp, BN_GGUF_TYPE_STRING);
    write_str(fp, "unit-test");

    write_str(fp, "general.alignment");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 32);

    write_str(fp, "tensor0");
    write_u32(fp, 2);
    write_u64(fp, 4);
    write_u64(fp, 2);
    write_u32(fp, BN_GGML_TYPE_F32);
    write_u64(fp, 0);

    write_padding(fp, 32);

    const float data[] = {1.0f, 2.0f, 3.0f, 4.0f,
                          5.0f, 6.0f, 7.0f, 8.0f};
    assert(fwrite(data, sizeof(float), 8, fp) == 8);
    assert(fclose(fp) == 0);
}

static void create_truncated_tensor_data_fixture(char path_template[]) {
    int fd = mkstemp(path_template);
    assert(fd >= 0);

    FILE *fp = fdopen(fd, "wb");
    assert(fp != NULL);

    assert(fwrite("GGUF", 1, 4, fp) == 4);
    write_u32(fp, 3);
    write_u64(fp, 1);
    write_u64(fp, 1);

    write_str(fp, "general.alignment");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 32);

    write_str(fp, "tensor0");
    write_u32(fp, 2);
    write_u64(fp, 4);
    write_u64(fp, 2);
    write_u32(fp, BN_GGML_TYPE_F32);
    write_u64(fp, 0);

    write_padding(fp, 32);

    {
        const float truncated_data[] = {1.0f, 2.0f};
        assert(fwrite(truncated_data, sizeof(float), 2, fp) == 2);
    }
    assert(fclose(fp) == 0);
}

static void create_zero_alignment_fixture(char path_template[]) {
    int fd = mkstemp(path_template);
    assert(fd >= 0);

    FILE *fp = fdopen(fd, "wb");
    assert(fp != NULL);

    assert(fwrite("GGUF", 1, 4, fp) == 4);
    write_u32(fp, 3);
    write_u64(fp, 1);
    write_u64(fp, 1);

    write_str(fp, "general.alignment");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 0);

    write_str(fp, "tensor0");
    write_u32(fp, 2);
    write_u64(fp, 4);
    write_u64(fp, 2);
    write_u32(fp, BN_GGML_TYPE_F32);
    write_u64(fp, 0);

    write_padding(fp, 32);

    {
        const float data[] = {1.0f, 2.0f, 3.0f, 4.0f,
                              5.0f, 6.0f, 7.0f, 8.0f};
        assert(fwrite(data, sizeof(float), 8, fp) == 8);
    }
    assert(fclose(fp) == 0);
}

static void create_non_pow2_alignment_fixture(char path_template[]) {
    int fd = mkstemp(path_template);
    assert(fd >= 0);

    FILE *fp = fdopen(fd, "wb");
    assert(fp != NULL);

    assert(fwrite("GGUF", 1, 4, fp) == 4);
    write_u32(fp, 3);
    write_u64(fp, 1);
    write_u64(fp, 1);

    write_str(fp, "general.alignment");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 3);  /* 3 is not a power of two */

    write_str(fp, "tensor0");
    write_u32(fp, 2);
    write_u64(fp, 4);
    write_u64(fp, 2);
    write_u32(fp, BN_GGML_TYPE_F32);
    write_u64(fp, 0);

    write_padding(fp, 32);

    {
        const float data[] = {1.0f, 2.0f, 3.0f, 4.0f,
                              5.0f, 6.0f, 7.0f, 8.0f};
        assert(fwrite(data, sizeof(float), 8, fp) == 8);
    }
    assert(fclose(fp) == 0);
}

static void create_wrong_type_alignment_fixture(char path_template[]) {
    int fd = mkstemp(path_template);
    assert(fd >= 0);

    FILE *fp = fdopen(fd, "wb");
    assert(fp != NULL);

    assert(fwrite("GGUF", 1, 4, fp) == 4);
    write_u32(fp, 3);
    write_u64(fp, 1);
    write_u64(fp, 1);

    /* Write general.alignment as STRING instead of UINT32 */
    write_str(fp, "general.alignment");
    write_u32(fp, BN_GGUF_TYPE_STRING);
    write_str(fp, "32");

    write_str(fp, "tensor0");
    write_u32(fp, 2);
    write_u64(fp, 4);
    write_u64(fp, 2);
    write_u32(fp, BN_GGML_TYPE_F32);
    write_u64(fp, 0);

    write_padding(fp, 32);

    {
        const float data[] = {1.0f, 2.0f, 3.0f, 4.0f,
                              5.0f, 6.0f, 7.0f, 8.0f};
        assert(fwrite(data, sizeof(float), 8, fp) == 8);
    }
    assert(fclose(fp) == 0);
}

static void create_misaligned_tensor_offset_fixture(char path_template[]) {
    int fd = mkstemp(path_template);
    assert(fd >= 0);

    FILE *fp = fdopen(fd, "wb");
    assert(fp != NULL);

    assert(fwrite("GGUF", 1, 4, fp) == 4);
    write_u32(fp, 3);
    write_u64(fp, 1);
    write_u64(fp, 1);

    write_str(fp, "general.alignment");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 32);

    write_str(fp, "tensor0");
    write_u32(fp, 2);
    write_u64(fp, 4);
    write_u64(fp, 2);
    write_u32(fp, BN_GGML_TYPE_F32);
    write_u64(fp, 7);  /* offset 7 is not a multiple of alignment 32 */

    write_padding(fp, 32);

    /* Write enough data so the tensor range doesn't exceed EOF */
    {
        uint8_t pad[64] = {0};
        assert(fwrite(pad, 1, sizeof(pad), fp) == sizeof(pad));
    }
    assert(fclose(fp) == 0);
}

static void create_too_many_dims_fixture(char path_template[]) {
    int fd = mkstemp(path_template);
    assert(fd >= 0);

    FILE *fp = fdopen(fd, "wb");
    assert(fp != NULL);

    assert(fwrite("GGUF", 1, 4, fp) == 4);
    write_u32(fp, 3);
    write_u64(fp, 1);
    write_u64(fp, 0);

    write_str(fp, "tensor0");
    write_u32(fp, BN_GGUF_MAX_DIMS + 1);
    assert(fclose(fp) == 0);
}

static void create_missing_token_embd_fixture(char path_template[]) {
    int fd = mkstemp(path_template);
    assert(fd >= 0);

    FILE *fp = fdopen(fd, "wb");
    assert(fp != NULL);

    assert(fwrite("GGUF", 1, 4, fp) == 4);
    write_u32(fp, 3);
    write_u64(fp, 0);
    write_u64(fp, 1);

    write_str(fp, "general.architecture");
    write_u32(fp, BN_GGUF_TYPE_STRING);
    write_str(fp, "bitnet-b1.58");

    assert(fclose(fp) == 0);
}

static void create_invalid_fixture(char path_template[],
                                   const void *data,
                                   size_t size) {
    int fd = mkstemp(path_template);
    assert(fd >= 0);
    assert(write(fd, data, size) == (ssize_t)size);
    assert(close(fd) == 0);
}

/* Build a GGUF with only metadata (0 tensors) for metadata-validation tests.
 * arch: architecture string, or NULL to omit the general.architecture KV.
 * skip_kv_suffix: if non-NULL, omit this arch-scoped key.
 * mistype_kv_suffix: if non-NULL, write this key with the wrong GGUF type. */
static void create_metadata_only_fixture(char path_template[],
                                         const char *arch,
                                         const char *skip_kv_suffix,
                                         const char *mistype_kv_suffix) {
    const char *prefix = arch ? arch : "bitnet-b1.58";

    struct { const char *suffix; uint32_t type; uint32_t u32_val; float f32_val; } keys[] = {
        { "vocab_size",                      BN_GGUF_TYPE_UINT32,  4,      0.0f    },
        { "embedding_length",                BN_GGUF_TYPE_UINT32,  4,      0.0f    },
        { "block_count",                     BN_GGUF_TYPE_UINT32,  1,      0.0f    },
        { "attention.head_count",            BN_GGUF_TYPE_UINT32,  1,      0.0f    },
        { "attention.head_count_kv",         BN_GGUF_TYPE_UINT32,  1,      0.0f    },
        { "feed_forward_length",             BN_GGUF_TYPE_UINT32,  4,      0.0f    },
        { "context_length",                  BN_GGUF_TYPE_UINT32,  8,      0.0f    },
        { "attention.layer_norm_rms_epsilon", BN_GGUF_TYPE_FLOAT32, 0,     1e-5f   },
        { "rope.freq_base",                  BN_GGUF_TYPE_FLOAT32, 0,     10000.0f },
    };
    const size_t n_keys = sizeof(keys) / sizeof(keys[0]);

    int n_kv = 1; /* general.alignment always present */
    if (arch) n_kv++;
    for (size_t i = 0; i < n_keys; i++) {
        if (skip_kv_suffix && strcmp(keys[i].suffix, skip_kv_suffix) == 0) continue;
        n_kv++;
    }

    int fd = mkstemp(path_template);
    assert(fd >= 0);
    FILE *fp = fdopen(fd, "wb");
    assert(fp != NULL);

    assert(fwrite("GGUF", 1, 4, fp) == 4);
    write_u32(fp, 3);
    write_u64(fp, 0); /* n_tensors */
    write_u64(fp, (uint64_t)n_kv);

    if (arch) {
        write_str(fp, "general.architecture");
        write_u32(fp, BN_GGUF_TYPE_STRING);
        write_str(fp, arch);
    }

    write_str(fp, "general.alignment");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 32);

    char key_buf[256];
    for (size_t i = 0; i < n_keys; i++) {
        if (skip_kv_suffix && strcmp(keys[i].suffix, skip_kv_suffix) == 0) continue;

        snprintf(key_buf, sizeof(key_buf), "%s.%s", prefix, keys[i].suffix);
        write_str(fp, key_buf);

        int mistype = mistype_kv_suffix &&
                      strcmp(keys[i].suffix, mistype_kv_suffix) == 0;
        if (mistype) {
            /* Swap u32 ↔ f32 to produce a type mismatch. */
            if (keys[i].type == BN_GGUF_TYPE_UINT32) {
                write_u32(fp, BN_GGUF_TYPE_FLOAT32);
                float v = 1.0f;
                assert(fwrite(&v, sizeof(v), 1, fp) == 1);
            } else {
                write_u32(fp, BN_GGUF_TYPE_UINT32);
                write_u32(fp, 42);
            }
        } else {
            write_u32(fp, keys[i].type);
            if (keys[i].type == BN_GGUF_TYPE_UINT32) {
                write_u32(fp, keys[i].u32_val);
            } else {
                assert(fwrite(&keys[i].f32_val, sizeof(float), 1, fp) == 1);
            }
        }
    }

    write_padding(fp, 32);
    assert(fclose(fp) == 0);
}

/* Build a metadata-only GGUF with custom head geometry values for validation
 * tests.  Any override value of 0 means "use the default". */
static void create_head_geometry_fixture(char path_template[],
                                         uint32_t n_embd,
                                         uint32_t n_head,
                                         uint32_t n_head_kv) {
    if (n_embd == 0) n_embd = 4;
    if (n_head == 0) n_head = 1;
    if (n_head_kv == 0) n_head_kv = 1;

    struct { const char *suffix; uint32_t type; uint32_t u32_val; float f32_val; } keys[] = {
        { "vocab_size",                      BN_GGUF_TYPE_UINT32,  4,        0.0f    },
        { "embedding_length",                BN_GGUF_TYPE_UINT32,  n_embd,   0.0f    },
        { "block_count",                     BN_GGUF_TYPE_UINT32,  1,        0.0f    },
        { "attention.head_count",            BN_GGUF_TYPE_UINT32,  n_head,   0.0f    },
        { "attention.head_count_kv",         BN_GGUF_TYPE_UINT32,  n_head_kv,0.0f    },
        { "feed_forward_length",             BN_GGUF_TYPE_UINT32,  4,        0.0f    },
        { "context_length",                  BN_GGUF_TYPE_UINT32,  8,        0.0f    },
        { "attention.layer_norm_rms_epsilon", BN_GGUF_TYPE_FLOAT32, 0,       1e-5f   },
        { "rope.freq_base",                  BN_GGUF_TYPE_FLOAT32, 0,       10000.0f },
    };
    const size_t n_keys = sizeof(keys) / sizeof(keys[0]);

    int n_kv = 2; /* general.architecture + general.alignment */
    n_kv += (int)n_keys;

    int fd = mkstemp(path_template);
    assert(fd >= 0);
    FILE *fp = fdopen(fd, "wb");
    assert(fp != NULL);

    assert(fwrite("GGUF", 1, 4, fp) == 4);
    write_u32(fp, 3);
    write_u64(fp, 0); /* n_tensors */
    write_u64(fp, (uint64_t)n_kv);

    write_str(fp, "general.architecture");
    write_u32(fp, BN_GGUF_TYPE_STRING);
    write_str(fp, "bitnet-b1.58");

    write_str(fp, "general.alignment");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 32);

    char key_buf[256];
    for (size_t i = 0; i < n_keys; i++) {
        snprintf(key_buf, sizeof(key_buf), "bitnet-b1.58.%s", keys[i].suffix);
        write_str(fp, key_buf);
        write_u32(fp, keys[i].type);
        if (keys[i].type == BN_GGUF_TYPE_UINT32) {
            write_u32(fp, keys[i].u32_val);
        } else {
            assert(fwrite(&keys[i].f32_val, sizeof(float), 1, fp) == 1);
        }
    }

    write_padding(fp, 32);
    assert(fclose(fp) == 0);
}

/* Build a metadata-only GGUF where embedding_length is written as
 * BN_GGUF_TYPE_STRING instead of BN_GGUF_TYPE_UINT32. */
static void create_string_typed_u32_fixture(char path_template[]) {
    int fd = mkstemp(path_template);
    assert(fd >= 0);
    FILE *fp = fdopen(fd, "wb");
    assert(fp != NULL);

    assert(fwrite("GGUF", 1, 4, fp) == 4);
    write_u32(fp, 3);
    write_u64(fp, 0); /* n_tensors */
    write_u64(fp, 11); /* n_kv */

    write_str(fp, "general.architecture");
    write_u32(fp, BN_GGUF_TYPE_STRING);
    write_str(fp, "bitnet-b1.58");

    write_str(fp, "general.alignment");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 32);

    write_str(fp, "bitnet-b1.58.vocab_size");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 4);

    /* This is the mismatch: embedding_length as STRING instead of UINT32. */
    write_str(fp, "bitnet-b1.58.embedding_length");
    write_u32(fp, BN_GGUF_TYPE_STRING);
    write_str(fp, "4");

    write_str(fp, "bitnet-b1.58.block_count");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 1);

    write_str(fp, "bitnet-b1.58.attention.head_count");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 1);

    write_str(fp, "bitnet-b1.58.attention.head_count_kv");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 1);

    write_str(fp, "bitnet-b1.58.feed_forward_length");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 4);

    write_str(fp, "bitnet-b1.58.context_length");
    write_u32(fp, BN_GGUF_TYPE_UINT32);
    write_u32(fp, 8);

    write_str(fp, "bitnet-b1.58.attention.layer_norm_rms_epsilon");
    write_u32(fp, BN_GGUF_TYPE_FLOAT32);
    {
        float eps = 1e-5f;
        assert(fwrite(&eps, sizeof(eps), 1, fp) == 1);
    }

    write_str(fp, "bitnet-b1.58.rope.freq_base");
    write_u32(fp, BN_GGUF_TYPE_FLOAT32);
    {
        float freq_base = 10000.0f;
        assert(fwrite(&freq_base, sizeof(freq_base), 1, fp) == 1);
    }

    write_padding(fp, 32);
    assert(fclose(fp) == 0);
}

static void create_missing_path(char path_template[]) {
    int fd = mkstemp(path_template);
    assert(fd >= 0);
    assert(close(fd) == 0);
    assert(unlink(path_template) == 0);
}

int main(void) {
    const char *model_path = getenv("BITNET_MODEL");
    if (!model_path) model_path = MODEL_PATH;
    bitnet_params_t params = bitnet_params_default();

    printf("=== GGUF Parser Tests ===\n\n");

    printf("Test 1: Reject NULL model when creating context... ");
    assert(bitnet_ctx_new(NULL, params) == NULL);
    printf("OK\n");

    char valid_ctx_model_path[] = "/tmp/bn_gguf_ctx_valid_XXXXXX";
    create_minimal_context_model_fixture(valid_ctx_model_path);
    bitnet_model_t *valid_ctx_model = bitnet_model_load(valid_ctx_model_path);
    assert(valid_ctx_model != NULL);

    printf("Test 2: Reject non-positive n_ctx when creating context... ");
    params.n_ctx = 0;
    assert(bitnet_ctx_new(valid_ctx_model, params) == NULL);
    params.n_ctx = -1;
    assert(bitnet_ctx_new(valid_ctx_model, params) == NULL);
    params = bitnet_params_default();
    printf("OK\n");

    printf("Test 3: Reject n_ctx larger than the model context window... ");
    params.n_ctx = 9;
    assert(bitnet_ctx_new(valid_ctx_model, params) == NULL);
    params = bitnet_params_default();
    printf("OK\n");

    printf("Test 4: Reject non-positive n_threads when creating context... ");
    params.n_ctx = 8;
    params.n_threads = 0;
    assert(bitnet_ctx_new(valid_ctx_model, params) == NULL);
    params.n_threads = -1;
    assert(bitnet_ctx_new(valid_ctx_model, params) == NULL);
    params = bitnet_params_default();
    printf("OK\n");

    bitnet_model_free(valid_ctx_model);
    assert(unlink(valid_ctx_model_path) == 0);

    char valid_path[] = "/tmp/bn_gguf_open_valid_XXXXXX";
    create_valid_gguf_fixture(valid_path);

    printf("Test 5: Open valid GGUF fixture... ");
    bn_gguf_t *g = bn_gguf_open(valid_path);
    assert(g != NULL);
    assert(g->version == 3);
    assert(g->n_tensors == 1);
    assert(g->n_kv == 2);
    assert(g->alignment == 32);
    assert(strcmp(bn_gguf_get_str(g, "general.architecture"), "unit-test") == 0);

    bn_gguf_tensor_t *fixture_tensor = bn_gguf_find_tensor(g, "tensor0");
    assert(fixture_tensor != NULL);
    assert(fixture_tensor->type == BN_GGML_TYPE_F32);
    assert(fixture_tensor->n_dims == 2);
    assert(fixture_tensor->ne[0] == 4);
    assert(fixture_tensor->ne[1] == 2);
    assert(((float *)fixture_tensor->data)[0] == 1.0f);
    assert(((float *)fixture_tensor->data)[7] == 8.0f);
    bn_gguf_close(g);
    assert(unlink(valid_path) == 0);
    printf("OK\n");

    char missing_path[] = "/tmp/bn_gguf_open_missing_XXXXXX";
    create_missing_path(missing_path);
    printf("Test 5: Reject missing path... ");
    assert(bn_gguf_open(missing_path) == NULL);
    printf("OK\n");

    char bad_magic_path[] = "/tmp/bn_gguf_open_bad_magic_XXXXXX";
    create_invalid_fixture(bad_magic_path, "NOTG", 4);
    printf("Test 6: Reject invalid GGUF magic... ");
    assert(bn_gguf_open(bad_magic_path) == NULL);
    assert(unlink(bad_magic_path) == 0);
    printf("OK\n");

    char too_many_dims_path[] = "/tmp/bn_gguf_open_too_many_dims_XXXXXX";
    create_too_many_dims_fixture(too_many_dims_path);
    printf("Test 7: Reject tensor with too many dimensions... ");
    assert(bn_gguf_open(too_many_dims_path) == NULL);
    assert(unlink(too_many_dims_path) == 0);
    printf("OK\n");

    char missing_token_embd_path[] = "/tmp/bn_gguf_open_missing_token_embd_XXXXXX";
    create_missing_token_embd_fixture(missing_token_embd_path);
    printf("Test 8: Reject model missing token_embd.weight... ");
    assert(bitnet_model_load(missing_token_embd_path) == NULL);
    assert(unlink(missing_token_embd_path) == 0);
    printf("OK\n");

    char zero_alignment_path[] = "/tmp/bn_gguf_open_zero_alignment_XXXXXX";
    create_zero_alignment_fixture(zero_alignment_path);
    printf("Test 9: Reject zero alignment... ");
    assert(bn_gguf_open(zero_alignment_path) == NULL);
    assert(unlink(zero_alignment_path) == 0);
    printf("OK\n");

    char non_pow2_alignment_path[] = "/tmp/bn_gguf_open_non_pow2_align_XXXXXX";
    create_non_pow2_alignment_fixture(non_pow2_alignment_path);
    printf("Test 9a: Reject non-power-of-two alignment... ");
    assert(bn_gguf_open(non_pow2_alignment_path) == NULL);
    assert(unlink(non_pow2_alignment_path) == 0);
    printf("OK\n");

    char wrong_type_alignment_path[] = "/tmp/bn_gguf_open_wrong_type_align_XXXXXX";
    create_wrong_type_alignment_fixture(wrong_type_alignment_path);
    printf("Test 9b: Reject wrong-typed general.alignment... ");
    assert(bn_gguf_open(wrong_type_alignment_path) == NULL);
    assert(unlink(wrong_type_alignment_path) == 0);
    printf("OK\n");

    char misaligned_offset_path[] = "/tmp/bn_gguf_open_misaligned_offset_XXXXXX";
    create_misaligned_tensor_offset_fixture(misaligned_offset_path);
    printf("Test 9c: Reject misaligned tensor offset... ");
    assert(bn_gguf_open(misaligned_offset_path) == NULL);
    assert(unlink(misaligned_offset_path) == 0);
    printf("OK\n");

    char truncated_tensor_path[] = "/tmp/bn_gguf_open_truncated_tensor_XXXXXX";
    create_truncated_tensor_data_fixture(truncated_tensor_path);
    printf("Test 10: Reject tensor data range past EOF... ");
    assert(bn_gguf_open(truncated_tensor_path) == NULL);
    assert(unlink(truncated_tensor_path) == 0);
    printf("OK\n");

    char missing_output_norm_path[] = "/tmp/bn_gguf_open_missing_output_norm_XXXXXX";
    create_minimal_model_fixture(missing_output_norm_path, "output_norm.weight");
    printf("Test 11: Reject model missing output_norm.weight... ");
    assert(bitnet_model_load(missing_output_norm_path) == NULL);
    assert(unlink(missing_output_norm_path) == 0);
    printf("OK\n");

    char missing_attn_q_path[] = "/tmp/bn_gguf_open_missing_attn_q_XXXXXX";
    create_minimal_model_fixture(missing_attn_q_path, "blk.0.attn_q.weight");
    printf("Test 12: Reject model missing blk.0.attn_q.weight... ");
    assert(bitnet_model_load(missing_attn_q_path) == NULL);
    assert(unlink(missing_attn_q_path) == 0);
    printf("OK\n");

    /* --- Malformed-metadata tests using metadata-only fixtures --- */
    {
        char p[] = "/tmp/bn_meta_no_arch_XXXXXX";
        create_metadata_only_fixture(p, NULL, NULL, NULL);
        printf("Test 13: Reject model with missing general.architecture... ");
        assert(bitnet_model_load(p) == NULL);
        assert(unlink(p) == 0);
        printf("OK\n");
    }
    {
        char p[] = "/tmp/bn_meta_wrong_arch_XXXXXX";
        create_metadata_only_fixture(p, "llama", NULL, NULL);
        printf("Test 14: Reject model with wrong architecture... ");
        assert(bitnet_model_load(p) == NULL);
        assert(unlink(p) == 0);
        printf("OK\n");
    }

    /* Test each required arch-scoped key missing. */
    {
        static const char *required_keys[] = {
            "vocab_size", "embedding_length", "block_count",
            "attention.head_count", "attention.head_count_kv",
            "feed_forward_length", "context_length",
            "attention.layer_norm_rms_epsilon", "rope.freq_base",
        };
        for (size_t i = 0; i < sizeof(required_keys)/sizeof(required_keys[0]); i++) {
            char p[] = "/tmp/bn_meta_miss_kv_XXXXXX";
            create_metadata_only_fixture(p, "bitnet-b1.58", required_keys[i], NULL);
            printf("Test 15.%zu: Reject model missing %s... ", i, required_keys[i]);
            assert(bitnet_model_load(p) == NULL);
            assert(unlink(p) == 0);
            printf("OK\n");
        }
    }

    /* Test wrong-typed keys (u32 written as f32, f32 written as u32). */
    {
        char p1[] = "/tmp/bn_meta_mistype_u32_XXXXXX";
        create_metadata_only_fixture(p1, "bitnet-b1.58", NULL, "vocab_size");
        printf("Test 16: Reject model with wrong-typed vocab_size... ");
        assert(bitnet_model_load(p1) == NULL);
        assert(unlink(p1) == 0);
        printf("OK\n");

        char p2[] = "/tmp/bn_meta_mistype_f32_XXXXXX";
        create_metadata_only_fixture(p2, "bitnet-b1.58", NULL, "rope.freq_base");
        printf("Test 17: Reject model with wrong-typed rope.freq_base... ");
        assert(bitnet_model_load(p2) == NULL);
        assert(unlink(p2) == 0);
        printf("OK\n");
    }

    /* --- Head geometry validation tests --- */
    {
        char p[] = "/tmp/bn_geom_embd_mod_head_XXXXXX";
        create_head_geometry_fixture(p, 5, 3, 1); /* 5 % 3 != 0 */
        printf("Test 19: Reject n_embd not divisible by n_head... ");
        assert(bitnet_model_load(p) == NULL);
        assert(unlink(p) == 0);
        printf("OK\n");
    }
    {
        char p[] = "/tmp/bn_geom_head_mod_kv_XXXXXX";
        create_head_geometry_fixture(p, 6, 3, 2); /* 3 % 2 != 0 */
        printf("Test 20: Reject n_head not divisible by n_head_kv... ");
        assert(bitnet_model_load(p) == NULL);
        assert(unlink(p) == 0);
        printf("OK\n");
    }
    {
        char p[] = "/tmp/bn_geom_odd_embd_head_XXXXXX";
        create_head_geometry_fixture(p, 3, 1, 1); /* n_embd_head = 3 (odd) */
        printf("Test 21: Reject odd n_embd_head... ");
        assert(bitnet_model_load(p) == NULL);
        assert(unlink(p) == 0);
        printf("OK\n");
    }
    {
        char p[] = "/tmp/bn_geom_kv_gt_head_XXXXXX";
        create_head_geometry_fixture(p, 4, 1, 2); /* n_head < n_head_kv */
        printf("Test 22: Reject n_head_kv > n_head... ");
        assert(bitnet_model_load(p) == NULL);
        assert(unlink(p) == 0);
        printf("OK\n");
    }

    /* Test embedding_length stored as STRING instead of UINT32. */
    {
        char p[] = "/tmp/bn_meta_str_embd_XXXXXX";
        create_string_typed_u32_fixture(p);
        printf("Test 23: Reject model with embedding_length as STRING... ");
        assert(bitnet_model_load(p) == NULL);
        assert(unlink(p) == 0);
        printf("OK\n");
    }

    char missing_tokenizer_kv_path[] = "/tmp/bn_gguf_open_missing_tokenizer_kv_XXXXXX";
    create_minimal_model_fixture(missing_tokenizer_kv_path, NULL);
    printf("Test 18: Reject context creation when tokenizer metadata is missing... ");
    bitnet_model_t *model = bitnet_model_load(missing_tokenizer_kv_path);
    assert(model != NULL);
    params.n_ctx = 8;
    bitnet_ctx_t *ctx = bitnet_ctx_new(model, params);
    assert(ctx == NULL);
    params = bitnet_params_default();
    bitnet_model_free(model);
    assert(unlink(missing_tokenizer_kv_path) == 0);
    printf("OK\n");

    if (access(model_path, R_OK) != 0) {
        printf("Test 14: Model-backed integration checks skipped (%s)\n",
               strerror(errno));
        printf("\n=== GGUF open tests passed ===\n");
        return 0;
    }

    printf("Test 14: Load model... ");
    g = bn_gguf_open(model_path);
    assert(g != NULL);
    printf("OK (v%u, %lu tensors, %lu kvs)\n",
           g->version, (unsigned long)g->n_tensors, (unsigned long)g->n_kv);

    printf("Test 15: Check metadata...\n");
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

    printf("Test 13: Check tensors...\n");
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

    printf("Test 14: All metadata keys:\n");
    for (uint64_t i = 0; i < g->n_kv; i++) {
        printf("  [%2lu] type=%2u key=%s\n",
               (unsigned long)i, g->kvs[i].type, g->kvs[i].key);
    }

    bn_gguf_close(g);

    printf("\n=== All GGUF tests passed ===\n");
    return 0;
}
