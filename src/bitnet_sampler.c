#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

static inline uint64_t bn_rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t bn_xoshiro_next(uint64_t *s) {
    uint64_t result = bn_rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = bn_rotl(s[3], 45);
    return result;
}

static float bn_rand_float(uint64_t *s) {
    uint64_t v = bn_xoshiro_next(s);
    return (float)(v >> 11) * 0x1.0p-53f;
}

void bn_sampler_init(bn_sampler_t *s, float temp, int top_k, float top_p) {
    s->temperature = temp;
    s->top_k = top_k;
    s->top_p = top_p;
    s->pairs_buf = NULL;
    s->pairs_cap = 0;
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        ssize_t r = read(fd, s->rng_state, sizeof(s->rng_state));
        (void)r;
        close(fd);
    } else {
        s->rng_state[0] = 0x12345678deadbeefULL;
        s->rng_state[1] = 0xfeedface01020304ULL;
        s->rng_state[2] = 0xabcdef0011223344ULL;
        s->rng_state[3] = 0x99887766aabbccddULL;
    }
}

void bn_sampler_seed(bn_sampler_t *s, uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        s->rng_state[i] = z ^ (z >> 31);
    }
}

typedef struct {
    float value;
    int   index;
} bn_logit_pair_t;

static int bn_cmp_desc(const void *a, const void *b) {
    float va = ((const bn_logit_pair_t *)a)->value;
    float vb = ((const bn_logit_pair_t *)b)->value;
    if (va > vb) return -1;
    if (va < vb) return 1;
    return 0;
}

int bn_sample(bn_sampler_t *s, float *logits, int n_vocab) {
    if (s->temperature < 1e-6f) {
        int best = 0;
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    float inv_temp = 1.0f / s->temperature;
    for (int i = 0; i < n_vocab; i++) logits[i] *= inv_temp;

    float mx = logits[0];
    for (int i = 1; i < n_vocab; i++)
        if (logits[i] > mx) mx = logits[i];

    float sum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        logits[i] = expf(logits[i] - mx);
        sum += logits[i];
    }
    for (int i = 0; i < n_vocab; i++) logits[i] /= sum;

    int k = (s->top_k > 0 && s->top_k < n_vocab) ? s->top_k : n_vocab;

    if (s->pairs_cap < n_vocab) {
        free(s->pairs_buf);
        s->pairs_buf = malloc((size_t)n_vocab * sizeof(bn_logit_pair_t));
        s->pairs_cap = n_vocab;
    }
    bn_logit_pair_t *pairs = (bn_logit_pair_t *)s->pairs_buf;
    for (int i = 0; i < n_vocab; i++) {
        pairs[i].value = logits[i];
        pairs[i].index = i;
    }
    qsort(pairs, (size_t)n_vocab, sizeof(bn_logit_pair_t), bn_cmp_desc);

    float cum = 0.0f;
    int np = k;
    if (s->top_p > 0.0f && s->top_p < 1.0f) {
        for (int i = 0; i < k; i++) {
            cum += pairs[i].value;
            if (cum >= s->top_p) {
                np = i + 1;
                break;
            }
        }
    }

    sum = 0.0f;
    for (int i = 0; i < np; i++) sum += pairs[i].value;

    float r = bn_rand_float(s->rng_state) * sum;
    float acc = 0.0f;
    int result = pairs[0].index;
    for (int i = 0; i < np; i++) {
        acc += pairs[i].value;
        if (acc >= r) {
            result = pairs[i].index;
            break;
        }
    }

    return result;
}
