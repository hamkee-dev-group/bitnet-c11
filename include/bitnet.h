#ifndef BITNET_H
#define BITNET_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BITNET_VERSION_MAJOR 0
#define BITNET_VERSION_MINOR 1
#define BITNET_VERSION_PATCH 0

enum bn_gguf_type {
    BN_GGUF_TYPE_UINT8   = 0,
    BN_GGUF_TYPE_INT8    = 1,
    BN_GGUF_TYPE_UINT16  = 2,
    BN_GGUF_TYPE_INT16   = 3,
    BN_GGUF_TYPE_UINT32  = 4,
    BN_GGUF_TYPE_INT32   = 5,
    BN_GGUF_TYPE_FLOAT32 = 6,
    BN_GGUF_TYPE_BOOL    = 7,
    BN_GGUF_TYPE_STRING  = 8,
    BN_GGUF_TYPE_ARRAY   = 9,
    BN_GGUF_TYPE_UINT64  = 10,
    BN_GGUF_TYPE_INT64   = 11,
    BN_GGUF_TYPE_FLOAT64 = 12,
};

enum bn_ggml_type {
    BN_GGML_TYPE_F32  = 0,
    BN_GGML_TYPE_F16  = 1,
    BN_GGML_TYPE_I2_S = 36,
};

#define BN_GGUF_MAX_DIMS 4

typedef struct {
    char    *key;
    uint32_t type;
    union {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        float    f32;
        bool     b;
        uint64_t u64;
        int64_t  i64;
        double   f64;
        struct { uint64_t len; char *data; } str;
        struct {
            uint32_t type;
            uint64_t len;
            void    *data;
        } arr;
    } val;
} bn_gguf_kv_t;

typedef struct {
    char    *name;
    uint32_t n_dims;
    uint64_t ne[BN_GGUF_MAX_DIMS];
    uint32_t type;
    uint64_t offset;
    void    *data;
} bn_gguf_tensor_t;

typedef struct {
    uint32_t         version;
    uint64_t         n_tensors;
    uint64_t         n_kv;
    bn_gguf_kv_t    *kvs;
    bn_gguf_tensor_t*tensors;
    uint32_t         alignment;
    uint64_t         data_offset;
    void            *mmap_addr;
    size_t           mmap_len;
    int              fd;
} bn_gguf_t;

bn_gguf_t       *bn_gguf_open(const char *path);
void             bn_gguf_close(bn_gguf_t *g);
bn_gguf_kv_t    *bn_gguf_find_kv(bn_gguf_t *g, const char *key);
const char      *bn_gguf_get_str(bn_gguf_t *g, const char *key);
uint32_t         bn_gguf_get_u32(bn_gguf_t *g, const char *key);
float            bn_gguf_get_f32(bn_gguf_t *g, const char *key);
bn_gguf_tensor_t*bn_gguf_find_tensor(bn_gguf_t *g, const char *name);

typedef struct {
    uint8_t *base;
    size_t   size;
    size_t   used;
} bn_arena_t;

bn_arena_t bn_arena_create(size_t size);
void       bn_arena_free(bn_arena_t *a);
void      *bn_arena_alloc(bn_arena_t *a, size_t size);
void       bn_arena_reset(bn_arena_t *a);

typedef struct bn_tokenizer bn_tokenizer_t;

bn_tokenizer_t *bn_tokenizer_create(bn_gguf_t *g);
void            bn_tokenizer_free(bn_tokenizer_t *t);
/* Tokenizes `text` into the caller-owned `tokens` buffer and returns the
 * number of token IDs written, up to `max_tokens`. Returns -1 on invalid
 * arguments, including NULL `text`, negative `max_tokens`, or NULL `tokens`
 * with positive capacity. The tokenizer does not retain ownership of either
 * input pointer after the call returns. */
int             bn_tokenize(bn_tokenizer_t *t, const char *text,
                            int *tokens, int max_tokens);
/* Returns a newly allocated UTF-8 string for the provided token IDs. The
 * caller owns the returned buffer and must free it. Token IDs outside the
 * tokenizer vocabulary are ignored. */
char           *bn_detokenize(bn_tokenizer_t *t, const int *tokens, int n);
int             bn_token_bos(bn_tokenizer_t *t);
int             bn_token_eos(bn_tokenizer_t *t);
/* Returns a pointer to a thread-local decoded token string. The pointer
 * remains valid only until the next `bn_token_text()` call on the same
 * thread. Token IDs outside the tokenizer vocabulary return `""`. */
const char     *bn_token_text(bn_tokenizer_t *t, int id);

void bn_quantize_acts(const float *src, int8_t *dst, int n,
                      float *scale_out, int32_t *sum_out);
void bn_dequant_i2s(const uint8_t *packed, float *dst, int n,
                    float scale);

typedef void (*bn_i2s_gemv_fn)(const uint8_t *weights, const int8_t *acts,
                                float *out, int n_rows, int n_cols);

void bn_i2s_gemv_scalar(const uint8_t *weights, const int8_t *acts,
                        float *out, int n_rows, int n_cols);

#if defined(__AVX2__)
void bn_i2s_gemv_avx2(const uint8_t *weights, const int8_t *acts,
                      float *out, int n_rows, int n_cols);
#endif

typedef struct {
    float    temperature;
    int      top_k;
    float    top_p;
    uint64_t rng_state[4];
    void    *pairs_buf;
    int      pairs_cap;
} bn_sampler_t;

void bn_sampler_init(bn_sampler_t *s, float temp, int top_k, float top_p);
void bn_sampler_seed(bn_sampler_t *s, uint64_t seed);
int  bn_sample(bn_sampler_t *s, float *logits, int n_vocab);

typedef struct {
    int n_vocab;
    int n_embd;
    int n_layer;
    int n_head;
    int n_head_kv;
    int n_ff;
    int n_ctx;
    int n_embd_head;
    float rope_freq_base;
    float rms_norm_eps;

    float *token_embd;

    struct {
        float    *attn_norm;
        uint8_t  *attn_q;
        uint8_t  *attn_k;
        uint8_t  *attn_v;
        float     attn_q_scale;
        float     attn_k_scale;
        float     attn_v_scale;
        float    *attn_sub_norm;
        uint8_t  *attn_output;
        float     attn_output_scale;

        float    *ffn_norm;
        uint8_t  *ffn_gate;
        uint8_t  *ffn_up;
        float     ffn_gate_scale;
        float     ffn_up_scale;
        float    *ffn_sub_norm;
        uint8_t  *ffn_down;
        float     ffn_down_scale;

        float     attn_q_wscale;
        float     attn_k_wscale;
        float     attn_v_wscale;
        float     attn_output_wscale;
        float     ffn_gate_wscale;
        float     ffn_up_wscale;
        float     ffn_down_wscale;
    } *layers;

    float *output_norm;
    float *output;
    uint8_t *output_i2s;
    float    output_scale;
    float    output_wscale;
    bool     output_is_i2s;

    bn_gguf_t *gguf;
} bitnet_model_t;

typedef struct {
    bitnet_model_t *model;
    bn_tokenizer_t *tokenizer;
    bn_sampler_t    sampler;
    bn_i2s_gemv_fn  gemv;

    float *k_cache;
    float *v_cache;
    int    kv_len;

    bn_arena_t scratch;
    int n_ctx;
    int n_threads;

    float *logits_buf;
    int    logits_cap;
} bitnet_ctx_t;

typedef struct {
    int   n_threads;
    int   n_ctx;
    float temperature;
    int   top_k;
    float top_p;
    uint64_t seed;
} bitnet_params_t;

bitnet_model_t  *bitnet_model_load(const char *path);
void             bitnet_model_free(bitnet_model_t *model);

bitnet_params_t  bitnet_params_default(void);
bitnet_ctx_t    *bitnet_ctx_new(bitnet_model_t *model, bitnet_params_t params);
void             bitnet_ctx_free(bitnet_ctx_t *ctx);

/* Returns -1 on invalid arguments. This wrapper rejects NULL `ctx` or
 * tokenizer state, and otherwise follows `bn_tokenize()` argument rules. */
int  bitnet_tokenize(bitnet_ctx_t *ctx, const char *text,
                     int *tokens, int max_tokens);
char *bitnet_detokenize(bitnet_ctx_t *ctx, const int *tokens, int n);

float *bitnet_forward(bitnet_ctx_t *ctx, const int *tokens, int n_tokens,
                      bool compute_logits);

int bitnet_sample_token(bitnet_ctx_t *ctx, float *logits);

/* Returns 0 without sampling when n_prompt <= 0. */
int bitnet_generate(bitnet_ctx_t *ctx, const int *prompt, int n_prompt,
                    int n_predict,
                    void (*callback)(int token, const char *text, void *ud),
                    void *userdata);

#ifdef __cplusplus
}
#endif
#endif
