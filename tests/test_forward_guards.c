#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>

typedef struct {
    char *left;
    char *right;
    int   rank;
} bn_merge_t;

struct bn_tokenizer {
    int    n_vocab;
    char **vocab;
    int   *vocab_len;

    int    n_merges;
    bn_merge_t *merges;

    int        ht_size;
    struct bn_ht_entry {
        char *key;
        int   val;
        int   next;
    } *ht;
    int *ht_buckets;
    int  ht_count;

    int        mt_size;
    struct bn_mt_entry {
        unsigned long long hash;
        int                rank;
        int                next;
    } *mt;
    int *mt_buckets;
    int  mt_count;

    int bos_id;
    int eos_id;
};

typedef struct {
    int count;
    int last_token;
    char text[32];
} gen_capture_t;

typedef struct {
    bitnet_model_t model;
    bitnet_ctx_t ctx;
    bn_tokenizer_t tokenizer;
    float token_embd[3];
    float output_norm[1];
    float output[3];
    float logits_buf[3];
    char *vocab[3];
    int vocab_len[3];
} tiny_fixture_t;

typedef struct {
    bitnet_model_t model;
    bitnet_ctx_t ctx;
    float *token_embd;
    float *output_norm;
    float *logits_buf;
    uint8_t *zero_weights;
    float *k_cache_buf;
    float *v_cache_buf;
} cache_guard_fixture_t;

typedef struct {
    bitnet_model_t model;
    bitnet_ctx_t ctx;
    float *token_embd;
    float *output_norm;
    float *ffn_sub_norm;
    float *logits_buf;
    uint8_t *zero_embd_embd;
    uint8_t *zero_ff_embd;
    uint8_t *zero_embd_ff;
    float *k_cache_buf;
    float *v_cache_buf;
} ffn_norm_fixture_t;

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
                out[group_pos] |= (uint8_t)(q << (6 - 2 * group_idx));
            }
            out += 32;
            col += blk;
        }
    }
    return packed;
}

static void init_tiny_fixture(tiny_fixture_t *fx, int n_ctx) {
    memset(fx, 0, sizeof(*fx));

    fx->token_embd[0] = 1.0f;
    fx->token_embd[1] = 1.0f;
    fx->token_embd[2] = 1.0f;
    fx->output_norm[0] = 1.0f;
    fx->output[0] = 0.0f;
    fx->output[1] = 2.0f;
    fx->output[2] = 1.0f;

    fx->model.n_vocab = 3;
    fx->model.n_embd = 1;
    fx->model.n_layer = 0;
    fx->model.n_head = 1;
    fx->model.n_head_kv = 1;
    fx->model.n_ff = 1;
    fx->model.n_ctx = n_ctx;
    fx->model.n_embd_head = 1;
    fx->model.rope_freq_base = 10000.0f;
    fx->model.rms_norm_eps = 1e-5f;
    fx->model.token_embd = fx->token_embd;
    fx->model.output_norm = fx->output_norm;
    fx->model.output = fx->output;
    fx->model.output_is_i2s = false;

    fx->vocab[0] = "";
    fx->vocab[1] = "B";
    fx->vocab[2] = "C";
    fx->vocab_len[0] = 0;
    fx->vocab_len[1] = 1;
    fx->vocab_len[2] = 1;
    fx->tokenizer.n_vocab = 3;
    fx->tokenizer.vocab = fx->vocab;
    fx->tokenizer.vocab_len = fx->vocab_len;
    fx->tokenizer.bos_id = 0;
    fx->tokenizer.eos_id = 2;

    fx->ctx.model = &fx->model;
    fx->ctx.tokenizer = &fx->tokenizer;
    fx->ctx.n_ctx = n_ctx;
    fx->ctx.n_threads = 1;
    fx->ctx.logits_buf = fx->logits_buf;
    fx->ctx.logits_cap = 3;
    fx->ctx.scratch = bn_arena_create(4096);
    bn_sampler_init(&fx->ctx.sampler, 0.0f, 1, 1.0f);
    bn_sampler_seed(&fx->ctx.sampler, 1);
}

static void free_tiny_fixture(tiny_fixture_t *fx) {
    free(fx->ctx.sampler.pairs_buf);
    bn_arena_free(&fx->ctx.scratch);
}

static void init_cache_guard_fixture(cache_guard_fixture_t *fx, int n_ctx) {
    const int n_vocab = 3;
    const int n_embd = 128;
    const int n_ff = 128;
    const int kv_dim = 128;
    const int cache_floats = n_ctx * kv_dim;
    const int guard_floats = 16;

    memset(fx, 0, sizeof(*fx));

    fx->token_embd = (float *)calloc((size_t)n_vocab * n_embd, sizeof(float));
    fx->output_norm = (float *)calloc((size_t)n_embd, sizeof(float));
    fx->logits_buf = (float *)calloc((size_t)n_vocab, sizeof(float));
    fx->model.layers = calloc(1, sizeof(fx->model.layers[0]));
    fx->k_cache_buf = (float *)calloc((size_t)cache_floats + guard_floats, sizeof(float));
    fx->v_cache_buf = (float *)calloc((size_t)cache_floats + guard_floats, sizeof(float));
    assert(fx->token_embd);
    assert(fx->output_norm);
    assert(fx->logits_buf);
    assert(fx->model.layers);
    assert(fx->k_cache_buf);
    assert(fx->v_cache_buf);

    for (int i = 0; i < n_vocab * n_embd; i++) {
        fx->token_embd[i] = 1.0f;
    }
    for (int i = 0; i < n_embd; i++) {
        fx->output_norm[i] = 1.0f;
    }

    int8_t *zeros = (int8_t *)calloc((size_t)n_embd * n_embd, sizeof(int8_t));
    assert(zeros);
    fx->zero_weights = pack_test_weights(zeros, n_embd, n_embd);
    free(zeros);
    assert(fx->zero_weights);

    for (int i = 0; i < cache_floats + guard_floats; i++) {
        fx->k_cache_buf[i] = 1234.5f;
        fx->v_cache_buf[i] = 6789.0f;
    }

    fx->model.n_vocab = n_vocab;
    fx->model.n_embd = n_embd;
    fx->model.n_layer = 1;
    fx->model.n_head = 1;
    fx->model.n_head_kv = 1;
    fx->model.n_ff = n_ff;
    fx->model.n_ctx = n_ctx;
    fx->model.n_embd_head = n_embd;
    fx->model.rope_freq_base = 10000.0f;
    fx->model.rms_norm_eps = 1e-5f;
    fx->model.token_embd = fx->token_embd;
    fx->model.output_norm = fx->output_norm;
    fx->model.output = NULL;
    fx->model.output_is_i2s = false;

    fx->model.layers[0].attn_norm = fx->output_norm;
    fx->model.layers[0].attn_q = fx->zero_weights;
    fx->model.layers[0].attn_k = fx->zero_weights;
    fx->model.layers[0].attn_v = fx->zero_weights;
    fx->model.layers[0].attn_q_scale = 1.0f;
    fx->model.layers[0].attn_k_scale = 1.0f;
    fx->model.layers[0].attn_v_scale = 1.0f;
    fx->model.layers[0].attn_sub_norm = fx->output_norm;
    fx->model.layers[0].attn_output = fx->zero_weights;
    fx->model.layers[0].attn_output_scale = 1.0f;
    fx->model.layers[0].ffn_norm = fx->output_norm;
    fx->model.layers[0].ffn_gate = fx->zero_weights;
    fx->model.layers[0].ffn_up = fx->zero_weights;
    fx->model.layers[0].ffn_gate_scale = 1.0f;
    fx->model.layers[0].ffn_up_scale = 1.0f;
    fx->model.layers[0].ffn_sub_norm = fx->output_norm;
    fx->model.layers[0].ffn_down = fx->zero_weights;
    fx->model.layers[0].ffn_down_scale = 1.0f;
    fx->model.layers[0].attn_q_wscale = 1.0f;
    fx->model.layers[0].attn_k_wscale = 1.0f;
    fx->model.layers[0].attn_v_wscale = 1.0f;
    fx->model.layers[0].attn_output_wscale = 1.0f;
    fx->model.layers[0].ffn_gate_wscale = 1.0f;
    fx->model.layers[0].ffn_up_wscale = 1.0f;
    fx->model.layers[0].ffn_down_wscale = 1.0f;

    fx->ctx.model = &fx->model;
    fx->ctx.n_ctx = n_ctx;
    fx->ctx.n_threads = 1;
    fx->ctx.gemv = bn_i2s_gemv_scalar;
    fx->ctx.logits_buf = fx->logits_buf;
    fx->ctx.logits_cap = n_vocab;
    fx->ctx.k_cache = fx->k_cache_buf;
    fx->ctx.v_cache = fx->v_cache_buf;
    fx->ctx.scratch = bn_arena_create(1u << 20);
    assert(fx->ctx.scratch.base);
}

static void free_cache_guard_fixture(cache_guard_fixture_t *fx) {
    bn_arena_free(&fx->ctx.scratch);
    free(fx->model.layers);
    free(fx->k_cache_buf);
    free(fx->v_cache_buf);
    free(fx->zero_weights);
    free(fx->logits_buf);
    free(fx->output_norm);
    free(fx->token_embd);
}

static void collect_cb(int token, const char *text, void *ud) {
    gen_capture_t *capture = (gen_capture_t *)ud;

    capture->count++;
    capture->last_token = token;
    if (text) {
        strncpy(capture->text, text, sizeof(capture->text) - 1);
        capture->text[sizeof(capture->text) - 1] = '\0';
    }
}

static void assert_bn_tokenize_invalid_logged(bn_tokenizer_t *tokenizer,
                                              const char *text,
                                              int *tokens,
                                              int max_tokens,
                                              const char *expected_fragment) {
    FILE *stderr_capture = tmpfile();
    assert(stderr_capture);

    fflush(stderr);
    int saved_stderr = dup(STDERR_FILENO);
    assert(saved_stderr >= 0);
    assert(dup2(fileno(stderr_capture), STDERR_FILENO) >= 0);

    int n = bn_tokenize(tokenizer, text, tokens, max_tokens);

    fflush(stderr);
    assert(dup2(saved_stderr, STDERR_FILENO) >= 0);
    close(saved_stderr);

    assert(n == -1);
    assert(fseek(stderr_capture, 0, SEEK_SET) == 0);
    char captured[256];
    size_t n_read = fread(captured, 1, sizeof(captured) - 1, stderr_capture);
    captured[n_read] = '\0';
    assert(strstr(captured, expected_fragment) != NULL);

    fclose(stderr_capture);
}

static void assert_bitnet_tokenize_invalid_logged(bitnet_ctx_t *ctx,
                                                  const char *text,
                                                  int *tokens,
                                                  int max_tokens,
                                                  const char *expected_fragment) {
    FILE *stderr_capture = tmpfile();
    assert(stderr_capture);

    fflush(stderr);
    int saved_stderr = dup(STDERR_FILENO);
    assert(saved_stderr >= 0);
    assert(dup2(fileno(stderr_capture), STDERR_FILENO) >= 0);

    int n = bitnet_tokenize(ctx, text, tokens, max_tokens);

    fflush(stderr);
    assert(dup2(saved_stderr, STDERR_FILENO) >= 0);
    close(saved_stderr);

    assert(n == -1);
    assert(fseek(stderr_capture, 0, SEEK_SET) == 0);
    char captured[256];
    size_t n_read = fread(captured, 1, sizeof(captured) - 1, stderr_capture);
    captured[n_read] = '\0';
    assert(strstr(captured, expected_fragment) != NULL);

    fclose(stderr_capture);
}

static void assert_forward_invalid_token_logged(const int *tokens, int n_tokens,
                                                const char *expected_fragment) {
    tiny_fixture_t fx;
    init_tiny_fixture(&fx, 4);

    FILE *stderr_capture = tmpfile();
    assert(stderr_capture);

    fflush(stderr);
    int saved_stderr = dup(STDERR_FILENO);
    assert(saved_stderr >= 0);
    assert(dup2(fileno(stderr_capture), STDERR_FILENO) >= 0);

    float *logits = bitnet_forward(&fx.ctx, tokens, n_tokens, true);

    fflush(stderr);
    assert(dup2(saved_stderr, STDERR_FILENO) >= 0);
    close(saved_stderr);

    assert(logits == NULL);
    assert(fx.ctx.kv_len == 0);

    assert(fseek(stderr_capture, 0, SEEK_SET) == 0);
    char captured[256];
    size_t n_read = fread(captured, 1, sizeof(captured) - 1, stderr_capture);
    captured[n_read] = '\0';
    assert(strstr(captured, expected_fragment) != NULL);

    fclose(stderr_capture);
    free_tiny_fixture(&fx);
}

static void test_tokenize_rejects_null_text(void) {
    tiny_fixture_t fx;
    init_tiny_fixture(&fx, 4);

    int tokens[4] = {0};

    printf("Test 1: Reject NULL tokenize text... ");
    assert_bn_tokenize_invalid_logged(&fx.tokenizer, NULL, tokens, 4,
                                      "tokenize: text is NULL");
    assert_bitnet_tokenize_invalid_logged(&fx.ctx, NULL, tokens, 4,
                                          "tokenize: text is NULL");
    printf("OK\n");

    free_tiny_fixture(&fx);
}

static void test_tokenize_rejects_null_tokens_with_capacity(void) {
    tiny_fixture_t fx;
    init_tiny_fixture(&fx, 4);

    printf("Test 2: Reject NULL token buffer with positive capacity... ");
    assert_bn_tokenize_invalid_logged(&fx.tokenizer, "abc", NULL, 1,
                                      "tokenize: tokens is NULL with positive capacity");
    assert_bitnet_tokenize_invalid_logged(&fx.ctx, "abc", NULL, 1,
                                          "tokenize: tokens is NULL with positive capacity");
    printf("OK\n");

    free_tiny_fixture(&fx);
}

static void test_tokenize_rejects_negative_capacity(void) {
    tiny_fixture_t fx;
    init_tiny_fixture(&fx, 4);

    int tokens[1] = {0};

    printf("Test 3: Reject negative tokenize capacity... ");
    assert_bn_tokenize_invalid_logged(&fx.tokenizer, "abc", tokens, -1,
                                      "tokenize: max_tokens must be >= 0");
    assert_bitnet_tokenize_invalid_logged(&fx.ctx, "abc", tokens, -1,
                                          "tokenize: max_tokens must be >= 0");
    printf("OK\n");

    free_tiny_fixture(&fx);
}

static void test_invalid_token_id(void) {
    printf("Test 4: Reject invalid token id... ");
    int bad = 3;
    assert_forward_invalid_token_logged(&bad, 1, "tokens[0]=3");
    printf("OK\n");
}

static void test_negative_token_id(void) {
    int tokens[] = {1, -1};

    printf("Test 5: Reject negative token id before embedding lookup... ");
    assert_forward_invalid_token_logged(tokens, 2, "tokens[1]=-1");
    printf("OK\n");
}

static void test_forward_rejects_full_context(void) {
    tiny_fixture_t fx;
    init_tiny_fixture(&fx, 1);
    fx.ctx.kv_len = 1;

    int token = 1;
    float *logits = bitnet_forward(&fx.ctx, &token, 1, true);

    printf("Test 6: Reject full KV cache in forward... ");
    assert(logits == NULL);
    assert(fx.ctx.kv_len == 1);
    printf("OK\n");

    free_tiny_fixture(&fx);
}

static void test_forward_rejects_overflowing_append(void) {
    tiny_fixture_t fx;
    init_tiny_fixture(&fx, 2);
    fx.ctx.kv_len = 1;

    int tokens[] = {1, 1};
    float *logits = bitnet_forward(&fx.ctx, tokens, 2, true);

    printf("Test 7: Reject batched KV cache overflow in forward... ");
    assert(logits == NULL);
    assert(fx.ctx.kv_len == 1);
    printf("OK\n");

    free_tiny_fixture(&fx);
}

static void test_forward_overflow_leaves_cache_untouched(void) {
    cache_guard_fixture_t fx;
    init_cache_guard_fixture(&fx, 2);
    fx.ctx.kv_len = 2;

    size_t cache_floats = (size_t)fx.ctx.n_ctx *
                          (size_t)fx.model.n_head_kv *
                          (size_t)fx.model.n_embd_head;
    size_t total_floats = cache_floats + 16;
    float *k_before = (float *)malloc(total_floats * sizeof(float));
    float *v_before = (float *)malloc(total_floats * sizeof(float));
    assert(k_before);
    assert(v_before);
    memcpy(k_before, fx.ctx.k_cache, total_floats * sizeof(float));
    memcpy(v_before, fx.ctx.v_cache, total_floats * sizeof(float));

    int token = 1;
    float *logits = bitnet_forward(&fx.ctx, &token, 1, false);

    printf("Test 8: Reject cache overflow before touching KV buffers... ");
    assert(logits == NULL);
    assert(fx.ctx.kv_len == 2);
    assert(memcmp(fx.ctx.k_cache, k_before, total_floats * sizeof(float)) == 0);
    assert(memcmp(fx.ctx.v_cache, v_before, total_floats * sizeof(float)) == 0);
    printf("OK\n");

    free(k_before);
    free(v_before);
    free_cache_guard_fixture(&fx);
}

static void test_generate_fails_on_context_exhaustion(void) {
    tiny_fixture_t fx;
    init_tiny_fixture(&fx, 2);

    int prompt[] = {0};
    gen_capture_t capture;
    memset(&capture, 0, sizeof(capture));

    int generated = bitnet_generate(&fx.ctx, prompt, 1, 2, collect_cb, &capture);

    printf("Test 9: Fail generation cleanly at context limit... ");
    assert(generated == -1);
    assert(capture.count == 1);
    assert(capture.last_token == 1);
    assert(strcmp(capture.text, "B") == 0);
    assert(fx.ctx.kv_len == 2);
    printf("OK\n");

    free_tiny_fixture(&fx);
}

static void test_generate_empty_prompt_returns_zero(void) {
    tiny_fixture_t fx;
    init_tiny_fixture(&fx, 2);

    gen_capture_t capture;
    memset(&capture, 0, sizeof(capture));

    int generated = bitnet_generate(&fx.ctx, NULL, 0, 2, collect_cb, &capture);

    printf("Test 10: Return without sampling for empty prompt... ");
    assert(generated == 0);
    assert(capture.count == 0);
    assert(fx.ctx.kv_len == 0);
    printf("OK\n");

    free_tiny_fixture(&fx);
}

static void test_bn_sample_rejects_null_sampler(void) {
    float logits[] = {1.0f, 2.0f};
    printf("Test 11: bn_sample rejects NULL sampler... ");
    assert(bn_sample(NULL, logits, 2) == -1);
    printf("OK\n");
}

static void test_bn_sample_rejects_null_logits(void) {
    bn_sampler_t s;
    bn_sampler_init(&s, 0.0f, 1, 1.0f);
    printf("Test 12: bn_sample rejects NULL logits... ");
    assert(bn_sample(&s, NULL, 2) == -1);
    printf("OK\n");
    free(s.pairs_buf);
}

static void test_bn_sample_rejects_non_positive_vocab(void) {
    bn_sampler_t s;
    bn_sampler_init(&s, 0.0f, 1, 1.0f);
    float logits[] = {1.0f};
    printf("Test 13: bn_sample rejects n_vocab <= 0... ");
    assert(bn_sample(&s, logits, 0) == -1);
    assert(bn_sample(&s, logits, -5) == -1);
    printf("OK\n");
    free(s.pairs_buf);
}

static void test_bitnet_sample_token_rejects_null_ctx(void) {
    float logits[] = {1.0f};
    printf("Test 14: bitnet_sample_token rejects NULL ctx... ");
    assert(bitnet_sample_token(NULL, logits) == -1);
    printf("OK\n");
}

static void test_bitnet_sample_token_rejects_null_model(void) {
    bitnet_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.model = NULL;
    float logits[] = {1.0f};
    printf("Test 15: bitnet_sample_token rejects NULL model... ");
    assert(bitnet_sample_token(&ctx, logits) == -1);
    printf("OK\n");
}

static void test_bitnet_sample_token_rejects_null_logits(void) {
    tiny_fixture_t fx;
    init_tiny_fixture(&fx, 4);
    printf("Test 16: bitnet_sample_token rejects NULL logits... ");
    assert(bitnet_sample_token(&fx.ctx, NULL) == -1);
    printf("OK\n");
    free_tiny_fixture(&fx);
}

static void test_bn_detokenize_rejects_null_tokenizer(void) {
    int tokens[] = {0, 1};
    printf("Test 18: bn_detokenize rejects NULL tokenizer... ");
    assert(bn_detokenize(NULL, tokens, 2) == NULL);
    printf("OK\n");
}

static void test_bitnet_detokenize_rejects_null_ctx(void) {
    int tokens[] = {0};
    printf("Test 19: bitnet_detokenize rejects NULL ctx... ");
    assert(bitnet_detokenize(NULL, tokens, 1) == NULL);
    printf("OK\n");
}

static void test_bitnet_detokenize_rejects_null_tokenizer(void) {
    bitnet_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.tokenizer = NULL;
    int tokens[] = {0};
    printf("Test 20: bitnet_detokenize rejects NULL tokenizer in ctx... ");
    assert(bitnet_detokenize(&ctx, tokens, 1) == NULL);
    printf("OK\n");
}

static void test_bn_token_bos_rejects_null(void) {
    printf("Test 21: bn_token_bos rejects NULL tokenizer... ");
    assert(bn_token_bos(NULL) == -1);
    printf("OK\n");
}

static void test_bn_token_eos_rejects_null(void) {
    printf("Test 22: bn_token_eos rejects NULL tokenizer... ");
    assert(bn_token_eos(NULL) == -1);
    printf("OK\n");
}

static void test_bn_token_text_rejects_null(void) {
    printf("Test 23: bn_token_text rejects NULL tokenizer... ");
    const char *result = bn_token_text(NULL, 0);
    assert(result != NULL);
    assert(result[0] == '\0');
    printf("OK\n");
}

static void test_bn_sample_growth_reuses_existing_buffer(void) {
    bn_sampler_t s;
    bn_sampler_init(&s, 1.0f, 0, 1.0f);
    bn_sampler_seed(&s, 42);

    /* Allocate pairs buffer via a first sample call (temp>0 uses pairs). */
    float logits3[] = {0.0f, 2.0f, 1.0f};
    int result = bn_sample(&s, logits3, 3);
    assert(result >= 0 && result < 3);
    assert(s.pairs_buf != NULL);
    assert(s.pairs_cap == 3);

    void *buf_after_growth = s.pairs_buf;

    /* Second call with same or smaller vocab must reuse the buffer. */
    float logits2[] = {1.0f, 0.0f};
    result = bn_sample(&s, logits2, 2);
    assert(result >= 0 && result < 2);
    assert(s.pairs_buf == buf_after_growth);
    assert(s.pairs_cap == 3);

    printf("Test 17: bn_sample growth reuses existing buffer when capacity sufficient... OK\n");

    free(s.pairs_buf);
}

static void init_ffn_norm_fixture(ffn_norm_fixture_t *fx, int n_ctx) {
    const int n_vocab = 3;
    const int n_embd = 128;
    const int n_ff = 256;
    const int kv_dim = n_embd;
    const int cache_floats = n_ctx * kv_dim;

    memset(fx, 0, sizeof(*fx));

    fx->token_embd = (float *)calloc((size_t)n_vocab * n_embd, sizeof(float));
    fx->output_norm = (float *)calloc((size_t)n_embd, sizeof(float));
    fx->ffn_sub_norm = (float *)calloc((size_t)n_ff, sizeof(float));
    fx->logits_buf = (float *)calloc((size_t)n_vocab, sizeof(float));
    fx->model.layers = calloc(1, sizeof(fx->model.layers[0]));
    fx->k_cache_buf = (float *)calloc((size_t)cache_floats, sizeof(float));
    fx->v_cache_buf = (float *)calloc((size_t)cache_floats, sizeof(float));
    assert(fx->token_embd && fx->output_norm && fx->ffn_sub_norm);
    assert(fx->logits_buf && fx->model.layers);
    assert(fx->k_cache_buf && fx->v_cache_buf);

    for (int i = 0; i < n_vocab * n_embd; i++) fx->token_embd[i] = 1.0f;
    for (int i = 0; i < n_embd; i++) fx->output_norm[i] = 1.0f;
    for (int i = 0; i < n_ff; i++) fx->ffn_sub_norm[i] = 1.0f;

    int8_t *zeros_ee = (int8_t *)calloc((size_t)n_embd * n_embd, sizeof(int8_t));
    int8_t *zeros_fe = (int8_t *)calloc((size_t)n_ff * n_embd, sizeof(int8_t));
    int8_t *zeros_ef = (int8_t *)calloc((size_t)n_embd * n_ff, sizeof(int8_t));
    assert(zeros_ee && zeros_fe && zeros_ef);
    fx->zero_embd_embd = pack_test_weights(zeros_ee, n_embd, n_embd);
    fx->zero_ff_embd = pack_test_weights(zeros_fe, n_ff, n_embd);
    fx->zero_embd_ff = pack_test_weights(zeros_ef, n_embd, n_ff);
    free(zeros_ee);
    free(zeros_fe);
    free(zeros_ef);

    fx->model.n_vocab = n_vocab;
    fx->model.n_embd = n_embd;
    fx->model.n_layer = 1;
    fx->model.n_head = 1;
    fx->model.n_head_kv = 1;
    fx->model.n_ff = n_ff;
    fx->model.n_ctx = n_ctx;
    fx->model.n_embd_head = n_embd;
    fx->model.rope_freq_base = 10000.0f;
    fx->model.rms_norm_eps = 1e-5f;
    fx->model.token_embd = fx->token_embd;
    fx->model.output_norm = fx->output_norm;
    fx->model.output = NULL;
    fx->model.output_is_i2s = false;

    fx->model.layers[0].attn_norm = fx->output_norm;
    fx->model.layers[0].attn_q = fx->zero_embd_embd;
    fx->model.layers[0].attn_k = fx->zero_embd_embd;
    fx->model.layers[0].attn_v = fx->zero_embd_embd;
    fx->model.layers[0].attn_q_scale = 1.0f;
    fx->model.layers[0].attn_k_scale = 1.0f;
    fx->model.layers[0].attn_v_scale = 1.0f;
    fx->model.layers[0].attn_sub_norm = fx->output_norm;
    fx->model.layers[0].attn_output = fx->zero_embd_embd;
    fx->model.layers[0].attn_output_scale = 1.0f;
    fx->model.layers[0].ffn_norm = fx->output_norm;
    fx->model.layers[0].ffn_gate = fx->zero_ff_embd;
    fx->model.layers[0].ffn_up = fx->zero_ff_embd;
    fx->model.layers[0].ffn_gate_scale = 1.0f;
    fx->model.layers[0].ffn_up_scale = 1.0f;
    fx->model.layers[0].ffn_sub_norm = fx->ffn_sub_norm;
    fx->model.layers[0].ffn_down = fx->zero_embd_ff;
    fx->model.layers[0].ffn_down_scale = 1.0f;
    fx->model.layers[0].attn_q_wscale = 1.0f;
    fx->model.layers[0].attn_k_wscale = 1.0f;
    fx->model.layers[0].attn_v_wscale = 1.0f;
    fx->model.layers[0].attn_output_wscale = 1.0f;
    fx->model.layers[0].ffn_gate_wscale = 1.0f;
    fx->model.layers[0].ffn_up_wscale = 1.0f;
    fx->model.layers[0].ffn_down_wscale = 1.0f;

    fx->ctx.model = &fx->model;
    fx->ctx.n_ctx = n_ctx;
    fx->ctx.n_threads = 1;
    fx->ctx.gemv = bn_i2s_gemv_scalar;
    fx->ctx.logits_buf = fx->logits_buf;
    fx->ctx.logits_cap = n_vocab;
    fx->ctx.k_cache = fx->k_cache_buf;
    fx->ctx.v_cache = fx->v_cache_buf;
    fx->ctx.scratch = bn_arena_create(1u << 20);
    assert(fx->ctx.scratch.base);
}

static void free_ffn_norm_fixture(ffn_norm_fixture_t *fx) {
    bn_arena_free(&fx->ctx.scratch);
    free(fx->model.layers);
    free(fx->k_cache_buf);
    free(fx->v_cache_buf);
    free(fx->zero_embd_embd);
    free(fx->zero_ff_embd);
    free(fx->zero_embd_ff);
    free(fx->ffn_sub_norm);
    free(fx->logits_buf);
    free(fx->output_norm);
    free(fx->token_embd);
}

static void test_ffn_sub_norm_uses_n_ff_dimension(void) {
    ffn_norm_fixture_t fx;
    init_ffn_norm_fixture(&fx, 4);

    int token = 0;
    float *logits = bitnet_forward(&fx.ctx, &token, 1, false);

    printf("Test 24: ffn_sub_norm indexed with n_ff dimension (n_ff != n_embd)... ");
    assert(logits != NULL);
    printf("OK\n");

    free_ffn_norm_fixture(&fx);
}

int main(void) {
    printf("=== Forward Guard Tests ===\n\n");

    test_tokenize_rejects_null_text();
    test_tokenize_rejects_null_tokens_with_capacity();
    test_tokenize_rejects_negative_capacity();
    test_invalid_token_id();
    test_negative_token_id();
    test_forward_rejects_full_context();
    test_forward_rejects_overflowing_append();
    test_forward_overflow_leaves_cache_untouched();
    test_generate_fails_on_context_exhaustion();
    test_generate_empty_prompt_returns_zero();
    test_bn_sample_rejects_null_sampler();
    test_bn_sample_rejects_null_logits();
    test_bn_sample_rejects_non_positive_vocab();
    test_bitnet_sample_token_rejects_null_ctx();
    test_bitnet_sample_token_rejects_null_model();
    test_bitnet_sample_token_rejects_null_logits();
    test_bn_sample_growth_reuses_existing_buffer();
    test_bn_detokenize_rejects_null_tokenizer();
    test_bitnet_detokenize_rejects_null_ctx();
    test_bitnet_detokenize_rejects_null_tokenizer();
    test_bn_token_bos_rejects_null();
    test_bn_token_eos_rejects_null();
    test_bn_token_text_rejects_null();
    test_ffn_sub_norm_uses_n_ff_dimension();

    printf("\n=== All forward guard tests passed ===\n");
    return 0;
}
