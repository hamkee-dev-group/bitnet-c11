#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifndef BN_PTHREAD_CREATE
#define BN_PTHREAD_CREATE pthread_create
#endif

static inline float bn_f16_scalar(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign;
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3ff;
            f = sign | (uint32_t)((127 - 15 + exp) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000u | (mant << 13);
    } else {
        f = sign | (uint32_t)((exp + 112) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, 4);
    return result;
}

static void bn_rmsnorm(float *out, const float *x, const float *w,
                        int n, float eps)
{
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

static void bn_rmsnorm_inplace(float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) x[i] *= scale * w[i];
}

static void bn_rope(float *q, int dim, int head_dim, int pos, float freq_base) {
    int n_heads = dim / head_dim;
    for (int h = 0; h < n_heads; h++) {
        float *v = q + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(freq_base, (float)i / (float)head_dim);
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

typedef struct {
    const uint8_t *weights;
    const int8_t  *acts;
    float         *out;
    int            n_cols;
    int            row_start;
    int            row_end;
    bn_i2s_gemv_fn gemv;
} bn_gemv_task_t;

static void *bn_gemv_thread(void *arg) {
    bn_gemv_task_t *task = (bn_gemv_task_t *)arg;
    int row_bytes = task->n_cols / 4;
    task->gemv(task->weights + (size_t)task->row_start * row_bytes,
               task->acts,
               task->out + task->row_start,
               task->row_end - task->row_start,
               task->n_cols);
    return NULL;
}

static void bn_gemv_mt(const uint8_t *weights, const int8_t *acts,
                        float *out, int n_rows, int n_cols,
                        bn_i2s_gemv_fn gemv, int n_threads)
{
    int nt = n_threads;
    if (nt <= 1 || n_rows < nt * 4) {
        gemv(weights, acts, out, n_rows, n_cols);
        return;
    }

    pthread_t threads[16];
    int started[16] = { 0 };
    bn_gemv_task_t tasks[16];
    if (nt > 16) nt = 16;

    int rows_per = n_rows / nt;
    for (int i = 0; i < nt; i++) {
        tasks[i].weights   = weights;
        tasks[i].acts      = acts;
        tasks[i].out       = out;
        tasks[i].n_cols    = n_cols;
        tasks[i].row_start = i * rows_per;
        tasks[i].row_end   = (i == nt - 1) ? n_rows : (i + 1) * rows_per;
        tasks[i].gemv      = gemv;
        int err = BN_PTHREAD_CREATE(&threads[i], NULL, bn_gemv_thread, &tasks[i]);
        if (err == 0) {
            started[i] = 1;
            continue;
        }
        fprintf(stderr, "bn_gemv_mt: pthread_create failed for worker %d: %s\n",
                i, strerror(err));
        bn_gemv_thread(&tasks[i]);
    }
    for (int i = 0; i < nt; i++) {
        if (started[i]) pthread_join(threads[i], NULL);
    }
}

static void bn_bitlinear(float *out, const uint8_t *weight,
                          const float *input, int n_out, int n_in,
                          float weight_scale, float layer_scale,
                          int8_t *act_buf, bn_i2s_gemv_fn gemv,
                          int n_threads)
{
    float act_scale;
    int32_t act_sum;
    bn_quantize_acts(input, act_buf, n_in, &act_scale, &act_sum);

    bn_gemv_mt(weight, act_buf, out, n_out, n_in, gemv, n_threads);

    float dscale = weight_scale / act_scale;
    if (layer_scale != 0.0f) dscale *= layer_scale;
    for (int i = 0; i < n_out; i++) {
        out[i] = (out[i] - (float)act_sum) * dscale;
    }
}

static bool bn_check_kv_capacity(const bitnet_ctx_t *ctx, int n_tokens,
                                 const char *caller)
{
    if (ctx->kv_len < 0 || ctx->kv_len > ctx->n_ctx ||
        n_tokens < 0 || n_tokens > ctx->n_ctx - ctx->kv_len) {
        fprintf(stderr, "%s: KV cache capacity exceeded (%d + %d > %d)\n",
                caller, ctx->kv_len, n_tokens, ctx->n_ctx);
        return false;
    }
    return true;
}

typedef struct {
    const float *w;
    const float *x;
    float       *out;
    int          n_in;
    int          row_start;
    int          row_end;
} bn_f32mv_task_t;

static void *bn_f32mv_thread(void *arg) {
    bn_f32mv_task_t *t = (bn_f32mv_task_t *)arg;
    const float *x = t->x;
    int n_in = t->n_in;
#if defined(__AVX2__)
    for (int i = t->row_start; i < t->row_end; i++) {
        const float *row = t->w + (size_t)i * n_in;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        int j = 0;
        for (; j + 15 < n_in; j += 16) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + j),
                                    _mm256_loadu_ps(x + j), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + j + 8),
                                    _mm256_loadu_ps(x + j + 8), acc1);
        }
        acc0 = _mm256_add_ps(acc0, acc1);
        __m128 hi = _mm256_extractf128_ps(acc0, 1);
        __m128 lo = _mm256_castps256_ps128(acc0);
        __m128 s4 = _mm_add_ps(lo, hi);
        __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
        float sum = _mm_cvtss_f32(s1);
        for (; j < n_in; j++) sum += row[j] * x[j];
        t->out[i] = sum;
    }
#else
    for (int i = t->row_start; i < t->row_end; i++) {
        const float *row = t->w + (size_t)i * n_in;
        float sum = 0.0f;
        for (int j = 0; j < n_in; j++) sum += row[j] * x[j];
        t->out[i] = sum;
    }
#endif
    return NULL;
}

static void bn_matmul_f32(float *out, const float *w,
                           const float *x, int n_out, int n_in,
                           int n_threads)
{
    int nt = n_threads;
    if (nt <= 1 || n_out < nt * 4) {
        bn_f32mv_task_t task = { w, x, out, n_in, 0, n_out };
        bn_f32mv_thread(&task);
        return;
    }
    if (nt > 16) nt = 16;
    pthread_t threads[16];
    int started[16] = { 0 };
    bn_f32mv_task_t tasks[16];
    int rows_per = n_out / nt;
    for (int i = 0; i < nt; i++) {
        tasks[i] = (bn_f32mv_task_t){
            w, x, out, n_in,
            i * rows_per,
            (i == nt - 1) ? n_out : (i + 1) * rows_per
        };
        int err = BN_PTHREAD_CREATE(&threads[i], NULL, bn_f32mv_thread, &tasks[i]);
        if (err == 0) {
            started[i] = 1;
            continue;
        }
        fprintf(stderr, "bn_matmul_f32: pthread_create failed for worker %d: %s\n",
                i, strerror(err));
        bn_f32mv_thread(&tasks[i]);
    }
    for (int i = 0; i < nt; i++) {
        if (started[i]) pthread_join(threads[i], NULL);
    }
}

static float bn_get_i2s_weight_scale(const void *data, int n_rows, int n_cols) {
    const float *sp = (const float *)((const uint8_t *)data +
                                       (size_t)n_rows * (size_t)n_cols / 4);
    return sp[0];
}

static bool bn_tensor_has_shape(const bn_gguf_tensor_t *t,
                                uint32_t n_dims,
                                const uint64_t *shape) {
    if (t->n_dims != n_dims) return false;
    for (uint32_t i = 0; i < n_dims; i++) {
        if (t->ne[i] != shape[i]) return false;
    }
    return true;
}

static bool bn_tensor_is_scalar(const bn_gguf_tensor_t *t) {
    if (t->n_dims == 0) return true;
    if (t->n_dims == 1 && t->ne[0] == 1) return true;
    return false;
}

static bool bn_require_f32_vector(bn_gguf_t *g, const char *name,
                                  uint64_t len, float **out) {
    uint64_t shape[1] = { len };
    bn_gguf_tensor_t *t = bn_gguf_find_tensor(g, name);
    if (!t) {
        fprintf(stderr, "model: missing %s\n", name);
        return false;
    }
    if (t->type != BN_GGML_TYPE_F32) {
        fprintf(stderr, "model: %s has type %u, expected F32\n",
                name, t->type);
        return false;
    }
    if (!bn_tensor_has_shape(t, 1, shape)) {
        fprintf(stderr, "model: %s has shape [%lu], expected [%lu]\n",
                name,
                (unsigned long)t->ne[0],
                (unsigned long)len);
        return false;
    }
    *out = (float *)t->data;
    return true;
}

static bool bn_require_f32_scalar(bn_gguf_t *g, const char *name, float *out) {
    bn_gguf_tensor_t *t = bn_gguf_find_tensor(g, name);
    if (!t) {
        fprintf(stderr, "model: missing %s\n", name);
        return false;
    }
    if (t->type != BN_GGML_TYPE_F32) {
        fprintf(stderr, "model: %s has type %u, expected F32\n",
                name, t->type);
        return false;
    }
    if (!bn_tensor_is_scalar(t)) {
        fprintf(stderr, "model: %s must be a scalar F32 tensor\n", name);
        return false;
    }
    *out = *(float *)t->data;
    return true;
}

static bool bn_require_i2s_matrix(bn_gguf_t *g, const char *name,
                                  uint64_t n_rows, uint64_t n_cols,
                                  uint8_t **out, float *wscale_out,
                                  float *scale_out) {
    char scale_name[256];
    uint64_t shape[2] = { n_cols, n_rows };
    bn_gguf_tensor_t *t = bn_gguf_find_tensor(g, name);
    if (!t) {
        fprintf(stderr, "model: missing %s\n", name);
        return false;
    }
    if (t->type != BN_GGML_TYPE_I2_S) {
        fprintf(stderr, "model: %s has type %u, expected I2_S\n",
                name, t->type);
        return false;
    }
    if (!bn_tensor_has_shape(t, 2, shape)) {
        fprintf(stderr, "model: %s has shape [%lu, %lu], expected [%lu, %lu]\n",
                name,
                (unsigned long)t->ne[0], (unsigned long)t->ne[1],
                (unsigned long)n_cols, (unsigned long)n_rows);
        return false;
    }

    snprintf(scale_name, sizeof(scale_name), "%s.scale", name);
    if (!bn_require_f32_scalar(g, scale_name, scale_out)) return false;

    *out = (uint8_t *)t->data;
    *wscale_out = bn_get_i2s_weight_scale(t->data, (int)n_rows, (int)n_cols);
    return true;
}

bitnet_model_t *bitnet_model_load(const char *path) {
    bn_gguf_t *g = bn_gguf_open(path);
    if (!g) return NULL;

    bitnet_model_t *m = (bitnet_model_t *)calloc(1, sizeof(bitnet_model_t));
    if (!m) { bn_gguf_close(g); return NULL; }
    m->gguf = g;

    const char *arch = bn_gguf_get_str(g, "general.architecture");
    if (!arch) arch = "bitnet-b1.58";

    char key[256];

    #define GET_U32(field, name) do { \
        snprintf(key, sizeof(key), "%s." name, arch); \
        m->field = (int)bn_gguf_get_u32(g, key); \
    } while(0)

    #define GET_F32(field, name) do { \
        snprintf(key, sizeof(key), "%s." name, arch); \
        m->field = bn_gguf_get_f32(g, key); \
    } while(0)

    GET_U32(n_vocab,     "vocab_size");
    GET_U32(n_embd,      "embedding_length");
    GET_U32(n_layer,     "block_count");
    GET_U32(n_head,      "attention.head_count");
    GET_U32(n_head_kv,   "attention.head_count_kv");
    GET_U32(n_ff,        "feed_forward_length");
    GET_U32(n_ctx,       "context_length");
    GET_F32(rms_norm_eps,"attention.layer_norm_rms_epsilon");
    GET_F32(rope_freq_base, "rope.freq_base");

    #undef GET_U32
    #undef GET_F32

    if (m->n_head == 0) m->n_head = 20;
    if (m->n_head_kv == 0) m->n_head_kv = 5;
    if (m->n_embd == 0) m->n_embd = 2560;
    if (m->n_layer == 0) m->n_layer = 30;
    if (m->n_ff == 0) m->n_ff = 6912;
    if (m->n_ctx == 0) m->n_ctx = 4096;
    if (m->n_vocab == 0) m->n_vocab = 128256;
    if (m->rms_norm_eps == 0.0f) m->rms_norm_eps = 1e-5f;
    if (m->rope_freq_base == 0.0f) m->rope_freq_base = 500000.0f;
    m->n_embd_head = m->n_embd / m->n_head;

    fprintf(stderr, "model: %s, embd=%d, layers=%d, heads=%d/%d, ff=%d, ctx=%d\n",
            arch, m->n_embd, m->n_layer, m->n_head, m->n_head_kv,
            m->n_ff, m->n_ctx);

    {
        uint64_t shape[2] = { (uint64_t)m->n_embd, (uint64_t)m->n_vocab };
        bn_gguf_tensor_t *te = bn_gguf_find_tensor(g, "token_embd.weight");
        if (!te) {
            fprintf(stderr, "model: missing token_embd.weight\n");
            bitnet_model_free(m);
            return NULL;
        }
        if (te->type != BN_GGML_TYPE_F32 && te->type != BN_GGML_TYPE_F16) {
            fprintf(stderr,
                    "model: token_embd.weight has type %u, expected F32 or F16\n",
                    te->type);
            bitnet_model_free(m);
            return NULL;
        }
        if (!bn_tensor_has_shape(te, 2, shape)) {
            fprintf(stderr,
                    "model: token_embd.weight has shape [%lu, %lu], expected [%lu, %lu]\n",
                    (unsigned long)te->ne[0], (unsigned long)te->ne[1],
                    (unsigned long)shape[0], (unsigned long)shape[1]);
            bitnet_model_free(m);
            return NULL;
        }
        if (te->type == BN_GGML_TYPE_F32) {
            m->token_embd = (float *)te->data;
        } else {
            int n = m->n_vocab * m->n_embd;
            m->token_embd = (float *)malloc((size_t)n * sizeof(float));
            if (!m->token_embd) {
                bitnet_model_free(m);
                return NULL;
            }
            const uint16_t *src = (const uint16_t *)te->data;
            for (int i = 0; i < n; i++) {
                m->token_embd[i] = bn_f16_scalar(src[i]);
            }
        }
    }

    if (!bn_require_f32_vector(g, "output_norm.weight",
                               (uint64_t)m->n_embd, &m->output_norm)) {
        bitnet_model_free(m);
        return NULL;
    }

    bn_gguf_tensor_t *oh = bn_gguf_find_tensor(g, "output.weight");
    if (oh) {
        uint64_t shape[2] = { (uint64_t)m->n_embd, (uint64_t)m->n_vocab };
        if (!bn_tensor_has_shape(oh, 2, shape)) {
            fprintf(stderr,
                    "model: output.weight has shape [%lu, %lu], expected [%lu, %lu]\n",
                    (unsigned long)oh->ne[0], (unsigned long)oh->ne[1],
                    (unsigned long)shape[0], (unsigned long)shape[1]);
            bitnet_model_free(m);
            return NULL;
        }
        if (oh->type == BN_GGML_TYPE_F32) {
            m->output = (float *)oh->data;
            m->output_is_i2s = false;
        } else if (oh->type == BN_GGML_TYPE_I2_S) {
            m->output_i2s = (uint8_t *)oh->data;
            m->output_is_i2s = true;
            m->output_wscale = bn_get_i2s_weight_scale(oh->data,
                                (int)oh->ne[1], (int)oh->ne[0]);
            if (!bn_require_f32_scalar(g, "output.scale", &m->output_scale)) {
                bitnet_model_free(m);
                return NULL;
            }
        } else {
            fprintf(stderr,
                    "model: output.weight has type %u, expected F32 or I2_S\n",
                    oh->type);
            bitnet_model_free(m);
            return NULL;
        }
    } else {
        m->output = m->token_embd;
        m->output_is_i2s = false;
    }

    m->layers = calloc((size_t)m->n_layer, sizeof(m->layers[0]));
    if (!m->layers) { bitnet_model_free(m); return NULL; }

    int kv_dim = m->n_head_kv * m->n_embd_head;

    for (int l = 0; l < m->n_layer; l++) {
        char name[256];

        #define LOAD_F32(dst, fmt, len, ...) do { \
            snprintf(name, sizeof(name), fmt, __VA_ARGS__); \
            if (!bn_require_f32_vector(g, name, (uint64_t)(len), &(dst))) { \
                bitnet_model_free(m); \
                return NULL; \
            } \
        } while(0)

        #define LOAD_I2S(dst, wscale_dst, scale_dst, fmt, nr, nc, ...) do { \
            snprintf(name, sizeof(name), fmt, __VA_ARGS__); \
            if (!bn_require_i2s_matrix(g, name, (uint64_t)(nr), (uint64_t)(nc), \
                                       &(dst), &(wscale_dst), &(scale_dst))) { \
                bitnet_model_free(m); \
                return NULL; \
            } \
        } while(0)

        LOAD_F32(m->layers[l].attn_norm, "blk.%d.attn_norm.weight", m->n_embd, l);
        LOAD_F32(m->layers[l].attn_sub_norm, "blk.%d.attn_sub_norm.weight", m->n_embd, l);
        LOAD_F32(m->layers[l].ffn_norm, "blk.%d.ffn_norm.weight", m->n_embd, l);
        LOAD_F32(m->layers[l].ffn_sub_norm, "blk.%d.ffn_sub_norm.weight", m->n_embd, l);

        LOAD_I2S(m->layers[l].attn_q, m->layers[l].attn_q_wscale,
                 m->layers[l].attn_q_scale,
                 "blk.%d.attn_q.weight", m->n_embd, m->n_embd, l);
        LOAD_I2S(m->layers[l].attn_k, m->layers[l].attn_k_wscale,
                 m->layers[l].attn_k_scale,
                 "blk.%d.attn_k.weight", kv_dim, m->n_embd, l);
        LOAD_I2S(m->layers[l].attn_v, m->layers[l].attn_v_wscale,
                 m->layers[l].attn_v_scale,
                 "blk.%d.attn_v.weight", kv_dim, m->n_embd, l);
        LOAD_I2S(m->layers[l].attn_output, m->layers[l].attn_output_wscale,
                 m->layers[l].attn_output_scale,
                 "blk.%d.attn_output.weight", m->n_embd, m->n_embd, l);

        LOAD_I2S(m->layers[l].ffn_gate, m->layers[l].ffn_gate_wscale,
                 m->layers[l].ffn_gate_scale,
                 "blk.%d.ffn_gate.weight", m->n_ff, m->n_embd, l);
        LOAD_I2S(m->layers[l].ffn_up, m->layers[l].ffn_up_wscale,
                 m->layers[l].ffn_up_scale,
                 "blk.%d.ffn_up.weight", m->n_ff, m->n_embd, l);
        LOAD_I2S(m->layers[l].ffn_down, m->layers[l].ffn_down_wscale,
                 m->layers[l].ffn_down_scale,
                 "blk.%d.ffn_down.weight", m->n_embd, m->n_ff, l);

        #undef LOAD_F32
        #undef LOAD_I2S
    }

    fprintf(stderr, "model: loaded %d layers\n", m->n_layer);
    return m;
}

void bitnet_model_free(bitnet_model_t *m) {
    if (!m) return;
    bn_gguf_tensor_t *te = bn_gguf_find_tensor(m->gguf, "token_embd.weight");
    if (te && te->type == BN_GGML_TYPE_F16) {
        free(m->token_embd);
    }
    free(m->layers);
    bn_gguf_close(m->gguf);
    free(m);
}

bitnet_params_t bitnet_params_default(void) {
    bitnet_params_t p = {0};
    p.n_threads = 4;
    p.n_ctx = 2048;
    p.temperature = 0.6f;
    p.top_k = 40;
    p.top_p = 0.9f;
    p.seed = 0;
    return p;
}

bitnet_ctx_t *bitnet_ctx_new(bitnet_model_t *model, bitnet_params_t params) {
    if (!model) {
        fprintf(stderr, "ctx: model must not be NULL\n");
        return NULL;
    }
    if (params.n_ctx <= 0) {
        fprintf(stderr, "ctx: n_ctx must be > 0 (got %d)\n", params.n_ctx);
        return NULL;
    }
    if (params.n_ctx > model->n_ctx) {
        fprintf(stderr, "ctx: n_ctx must be <= model context window (%d, got %d)\n",
                model->n_ctx, params.n_ctx);
        return NULL;
    }
    if (params.n_threads <= 0) {
        fprintf(stderr, "ctx: n_threads must be > 0 (got %d)\n", params.n_threads);
        return NULL;
    }

    bitnet_ctx_t *ctx = (bitnet_ctx_t *)calloc(1, sizeof(bitnet_ctx_t));
    if (!ctx) return NULL;
    ctx->model = model;
    ctx->n_ctx = params.n_ctx;
    ctx->n_threads = params.n_threads;

    ctx->logits_cap = model->n_vocab;
    ctx->logits_buf = (float *)malloc((size_t)ctx->logits_cap * sizeof(float));
    if (!ctx->logits_buf) goto fail;

    ctx->tokenizer = bn_tokenizer_create(model->gguf);
    if (!ctx->tokenizer) goto fail;

    bn_sampler_init(&ctx->sampler, params.temperature, params.top_k, params.top_p);
    if (params.seed != 0)
        bn_sampler_seed(&ctx->sampler, params.seed);

#if defined(__AVX2__)
    ctx->gemv = bn_i2s_gemv_avx2;
    fprintf(stderr, "ctx: using AVX2 kernel\n");
#else
    ctx->gemv = bn_i2s_gemv_scalar;
    fprintf(stderr, "ctx: using scalar kernel\n");
#endif

    int kv_dim = model->n_head_kv * model->n_embd_head;
    size_t kv_size = (size_t)model->n_layer * (size_t)params.n_ctx *
                     (size_t)kv_dim * sizeof(float);
    ctx->k_cache = (float *)calloc(1, kv_size);
    ctx->v_cache = (float *)calloc(1, kv_size);
    if (!ctx->k_cache || !ctx->v_cache) goto fail;
    ctx->kv_len = 0;

    size_t scratch_size = 64 * 1024 * 1024;
    ctx->scratch = bn_arena_create(scratch_size);
    if (!ctx->scratch.base) goto fail;

    return ctx;

fail:
    bitnet_ctx_free(ctx);
    return NULL;
}

void bitnet_ctx_free(bitnet_ctx_t *ctx) {
    if (!ctx) return;
    bn_tokenizer_free(ctx->tokenizer);
    free(ctx->k_cache);
    free(ctx->v_cache);
    free(ctx->logits_buf);
    free(ctx->sampler.pairs_buf);
    bn_arena_free(&ctx->scratch);
    free(ctx);
}

int bitnet_tokenize(bitnet_ctx_t *ctx, const char *text,
                    int *tokens, int max_tokens)
{
    if (!ctx) {
        fprintf(stderr, "bitnet_tokenize: ctx is NULL\n");
        return -1;
    }
    if (!ctx->tokenizer) {
        fprintf(stderr, "bitnet_tokenize: tokenizer is NULL\n");
        return -1;
    }
    return bn_tokenize(ctx->tokenizer, text, tokens, max_tokens);
}

char *bitnet_detokenize(bitnet_ctx_t *ctx, const int *tokens, int n) {
    if (!ctx || !ctx->tokenizer) return NULL;
    return bn_detokenize(ctx->tokenizer, tokens, n);
}

int bitnet_sample_token(bitnet_ctx_t *ctx, float *logits) {
    if (!ctx || !ctx->model || !logits) return -1;
    return bn_sample(&ctx->sampler, logits, ctx->model->n_vocab);
}

float *bitnet_forward(bitnet_ctx_t *ctx, const int *tokens, int n_tokens,
                      bool compute_logits) {
    bitnet_model_t *m = ctx->model;
    int embd   = m->n_embd;
    int n_head = m->n_head;
    int n_kv   = m->n_head_kv;
    int head_dim = m->n_embd_head;
    int kv_dim = n_kv * head_dim;
    int ff     = m->n_ff;
    float eps  = m->rms_norm_eps;
    int pos_start = ctx->kv_len;

    bn_arena_t *scratch = &ctx->scratch;

    if (n_tokens < 0 || (n_tokens > 0 && !tokens)) {
        fprintf(stderr, "forward: invalid token input\n");
        return NULL;
    }
    if (!bn_check_kv_capacity(ctx, n_tokens, "forward")) {
        return NULL;
    }
    for (int t = 0; t < n_tokens; t++) {
        int token = tokens[t];
        if (token < 0 || token >= m->n_vocab) {
            fprintf(stderr,
                    "forward: invalid token tokens[%d]=%d (valid range [0, %d))\n",
                    t, token, m->n_vocab);
            return NULL;
        }
    }

    if (ctx->logits_cap < m->n_vocab) {
        free(ctx->logits_buf);
        ctx->logits_buf = (float *)malloc((size_t)m->n_vocab * sizeof(float));
        ctx->logits_cap = m->n_vocab;
    }

    for (int t = 0; t < n_tokens; t++) {
        int pos = pos_start + t;
        int token = tokens[t];

        bn_arena_reset(scratch);

        float *x = (float *)bn_arena_alloc(scratch, (size_t)embd * sizeof(float));
        memcpy(x, m->token_embd + (size_t)token * embd,
               (size_t)embd * sizeof(float));

        float *xn  = (float *)bn_arena_alloc(scratch, (size_t)embd * sizeof(float));
        float *q   = (float *)bn_arena_alloc(scratch, (size_t)embd * sizeof(float));
        float *k   = (float *)bn_arena_alloc(scratch, (size_t)kv_dim * sizeof(float));
        float *v   = (float *)bn_arena_alloc(scratch, (size_t)kv_dim * sizeof(float));
        float *att = (float *)bn_arena_alloc(scratch, (size_t)n_head * (size_t)(pos + 1) * sizeof(float));
        float *attn_out = (float *)bn_arena_alloc(scratch, (size_t)embd * sizeof(float));
        float *gate = (float *)bn_arena_alloc(scratch, (size_t)ff * sizeof(float));
        float *up   = (float *)bn_arena_alloc(scratch, (size_t)ff * sizeof(float));
        float *ffn_out = (float *)bn_arena_alloc(scratch, (size_t)embd * sizeof(float));

        int max_act = embd > ff ? ff : embd;
        int8_t *act_buf = (int8_t *)bn_arena_alloc(scratch, (size_t)max_act);

        if (!x || !xn || !q || !k || !v || !att || !attn_out ||
            !gate || !up || !ffn_out || !act_buf) {
            return NULL;
        }

        for (int l = 0; l < m->n_layer; l++) {
            bn_rmsnorm(xn, x, m->layers[l].attn_norm, embd, eps);

            bn_bitlinear(q, m->layers[l].attn_q, xn, embd, embd,
                         m->layers[l].attn_q_wscale,
                         m->layers[l].attn_q_scale,
                         act_buf, ctx->gemv, ctx->n_threads);

            bn_bitlinear(k, m->layers[l].attn_k, xn, kv_dim, embd,
                         m->layers[l].attn_k_wscale,
                         m->layers[l].attn_k_scale,
                         act_buf, ctx->gemv, ctx->n_threads);

            bn_bitlinear(v, m->layers[l].attn_v, xn, kv_dim, embd,
                         m->layers[l].attn_v_wscale,
                         m->layers[l].attn_v_scale,
                         act_buf, ctx->gemv, ctx->n_threads);

            bn_rope(q, embd, head_dim, pos, m->rope_freq_base);
            bn_rope(k, kv_dim, head_dim, pos, m->rope_freq_base);

            size_t cache_layer_offset = (size_t)l * (size_t)ctx->n_ctx * (size_t)kv_dim;
            float *k_cache_pos = ctx->k_cache + cache_layer_offset +
                                 (size_t)pos * kv_dim;
            float *v_cache_pos = ctx->v_cache + cache_layer_offset +
                                 (size_t)pos * kv_dim;
            memcpy(k_cache_pos, k, (size_t)kv_dim * sizeof(float));
            memcpy(v_cache_pos, v, (size_t)kv_dim * sizeof(float));

            int gqa_ratio = n_head / n_kv;
            float scale = 1.0f / sqrtf((float)head_dim);

            memset(attn_out, 0, (size_t)embd * sizeof(float));

            for (int h = 0; h < n_head; h++) {
                int kv_h = h / gqa_ratio;
                float *q_h = q + h * head_dim;

                float *att_h = att + (size_t)h * (size_t)(pos + 1);
                for (int p = 0; p <= pos; p++) {
                    float *k_p = ctx->k_cache + cache_layer_offset +
                                 (size_t)p * kv_dim + kv_h * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++)
                        dot += q_h[d] * k_p[d];
                    att_h[p] = dot * scale;
                }

                float mx = att_h[0];
                for (int p = 1; p <= pos; p++)
                    if (att_h[p] > mx) mx = att_h[p];
                float sum = 0.0f;
                for (int p = 0; p <= pos; p++) {
                    att_h[p] = expf(att_h[p] - mx);
                    sum += att_h[p];
                }
                for (int p = 0; p <= pos; p++)
                    att_h[p] /= sum;

                float *out_h = attn_out + h * head_dim;
                for (int p = 0; p <= pos; p++) {
                    float w = att_h[p];
                    float *v_p = ctx->v_cache + cache_layer_offset +
                                 (size_t)p * kv_dim + kv_h * head_dim;
                    for (int d = 0; d < head_dim; d++)
                        out_h[d] += w * v_p[d];
                }
            }

            bn_rmsnorm_inplace(attn_out, m->layers[l].attn_sub_norm, embd, eps);

            float *proj_out = (float *)bn_arena_alloc(scratch, (size_t)embd * sizeof(float));
            if (!proj_out) return NULL;
            bn_bitlinear(proj_out, m->layers[l].attn_output, attn_out,
                         embd, embd,
                         m->layers[l].attn_output_wscale,
                         m->layers[l].attn_output_scale,
                         act_buf, ctx->gemv, ctx->n_threads);

            for (int i = 0; i < embd; i++) x[i] += proj_out[i];

            bn_rmsnorm(xn, x, m->layers[l].ffn_norm, embd, eps);

            bn_bitlinear(gate, m->layers[l].ffn_gate, xn, ff, embd,
                         m->layers[l].ffn_gate_wscale,
                         m->layers[l].ffn_gate_scale,
                         act_buf, ctx->gemv, ctx->n_threads);

            bn_bitlinear(up, m->layers[l].ffn_up, xn, ff, embd,
                         m->layers[l].ffn_up_wscale,
                         m->layers[l].ffn_up_scale,
                         act_buf, ctx->gemv, ctx->n_threads);

            for (int i = 0; i < ff; i++) {
                float g = gate[i] > 0.0f ? gate[i] : 0.0f;
                gate[i] = g * g * up[i];
            }

            bn_rmsnorm_inplace(gate, m->layers[l].ffn_sub_norm, ff, eps);

            int8_t *act_buf_ff = (int8_t *)bn_arena_alloc(scratch, (size_t)ff);
            if (!act_buf_ff) return NULL;
            bn_bitlinear(ffn_out, m->layers[l].ffn_down, gate, embd, ff,
                         m->layers[l].ffn_down_wscale,
                         m->layers[l].ffn_down_scale,
                         act_buf_ff, ctx->gemv, ctx->n_threads);

            for (int i = 0; i < embd; i++) x[i] += ffn_out[i];
        }

        bn_rmsnorm_inplace(x, m->output_norm, embd, eps);

        if (compute_logits && t == n_tokens - 1) {
            if (m->output_is_i2s) {
                int8_t *act_buf_out = (int8_t *)bn_arena_alloc(scratch, (size_t)embd);
                if (!act_buf_out) return NULL;
                bn_bitlinear(ctx->logits_buf, m->output_i2s, x,
                             m->n_vocab, embd,
                             m->output_wscale,
                             m->output_scale,
                             act_buf_out, ctx->gemv, ctx->n_threads);
            } else {
                bn_matmul_f32(ctx->logits_buf, m->output, x, m->n_vocab, embd,
                              ctx->n_threads);
            }
        }

        ctx->kv_len = pos + 1;
    }

    return ctx->logits_buf;
}

int bitnet_generate(bitnet_ctx_t *ctx, const int *prompt, int n_prompt,
                    int n_predict,
                    void (*callback)(int token, const char *text, void *ud),
                    void *userdata)
{
    if (n_prompt <= 0) return 0;

    int eos = bn_token_eos(ctx->tokenizer);
    int generated = 0;

    float *logits = NULL;
    for (int i = 0; i < n_prompt; i++) {
        bool need_logits = (i == n_prompt - 1);
        logits = bitnet_forward(ctx, &prompt[i], 1, need_logits);
        if (!logits) return -1;
    }

    for (int i = 0; i < n_predict; i++) {
        if (!bn_check_kv_capacity(ctx, 1, "generate")) return -1;

        int token = bn_sample(&ctx->sampler, logits, ctx->model->n_vocab);
        if (token == eos) break;

        const char *text = bn_token_text(ctx->tokenizer, token);
        if (callback) callback(token, text, userdata);

        generated++;
        logits = bitnet_forward(ctx, &token, 1, true);
        if (!logits) return -1;
    }

    return generated;
}
