#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static uint32_t byte_to_cp[256];
static uint8_t  cp_to_byte_map[512];
static pthread_once_t byte_tables_once = PTHREAD_ONCE_INIT;

static const char *bn_gguf_type_name(uint32_t type) {
    switch (type) {
    case BN_GGUF_TYPE_UINT8:   return "uint8";
    case BN_GGUF_TYPE_INT8:    return "int8";
    case BN_GGUF_TYPE_UINT16:  return "uint16";
    case BN_GGUF_TYPE_INT16:   return "int16";
    case BN_GGUF_TYPE_UINT32:  return "uint32";
    case BN_GGUF_TYPE_INT32:   return "int32";
    case BN_GGUF_TYPE_FLOAT32: return "float32";
    case BN_GGUF_TYPE_BOOL:    return "bool";
    case BN_GGUF_TYPE_STRING:  return "string";
    case BN_GGUF_TYPE_ARRAY:   return "array";
    case BN_GGUF_TYPE_UINT64:  return "uint64";
    case BN_GGUF_TYPE_INT64:   return "int64";
    case BN_GGUF_TYPE_FLOAT64: return "float64";
    default:                   return "unknown";
    }
}

static void bn_init_byte_tables_once(void) {
    memset(cp_to_byte_map, 0, sizeof(cp_to_byte_map));

    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 0x21 && b <= 0x7E) ||
            (b >= 0xA1 && b <= 0xAC) ||
            (b >= 0xAE && b <= 0xFF)) {
            byte_to_cp[b] = (uint32_t)b;
        } else {
            byte_to_cp[b] = (uint32_t)(256 + n);
            n++;
        }
        cp_to_byte_map[byte_to_cp[b]] = (uint8_t)b;
    }
}

static void bn_init_byte_tables(void) {
    if (pthread_once(&byte_tables_once, bn_init_byte_tables_once) != 0) {
        abort();
    }
}

static int bn_cp_to_utf8(uint32_t cp, char *out) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    } else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000) {
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

static int bn_is_utf8_cont(uint8_t c) {
    return (c & 0xC0) == 0x80;
}

static int bn_utf8_to_cp(const char *s, int len, uint32_t *cp, int *advance) {
    uint8_t c = (uint8_t)s[0];
    if (c < 0x80) {
        *cp = c;
        *advance = 1;
        return 1;
    } else if (c >= 0xC2 && c < 0xE0) {
        if (len < 2 || !bn_is_utf8_cont((uint8_t)s[1])) return 0;
        *cp = ((uint32_t)(c & 0x1F) << 6) | ((uint32_t)(s[1] & 0x3F));
        *advance = 2;
        return 1;
    } else if (c < 0xF0) {
        uint8_t c1;
        if (len < 3 ||
            !bn_is_utf8_cont((uint8_t)s[1]) ||
            !bn_is_utf8_cont((uint8_t)s[2])) {
            return 0;
        }
        c1 = (uint8_t)s[1];
        if ((c == 0xE0 && c1 < 0xA0) || (c == 0xED && c1 >= 0xA0)) return 0;
        *cp = ((uint32_t)(c & 0x0F) << 12) |
              ((uint32_t)(s[1] & 0x3F) << 6) |
              ((uint32_t)(s[2] & 0x3F));
        *advance = 3;
        return 1;
    }
    if (c >= 0xF5) return 0;
    if (len < 4 ||
        !bn_is_utf8_cont((uint8_t)s[1]) ||
        !bn_is_utf8_cont((uint8_t)s[2]) ||
        !bn_is_utf8_cont((uint8_t)s[3])) {
        return 0;
    }
    if ((c == 0xF0 && (uint8_t)s[1] < 0x90) ||
        (c == 0xF4 && (uint8_t)s[1] >= 0x90)) {
        return 0;
    }
    *cp = ((uint32_t)(c & 0x07) << 18) |
          ((uint32_t)(s[1] & 0x3F) << 12) |
          ((uint32_t)(s[2] & 0x3F) << 6) |
          ((uint32_t)(s[3] & 0x3F));
    *advance = 4;
    return 1;
}

static char *bn_bytes_to_gpt2(const char *input, int len, int *out_len) {
    bn_init_byte_tables();
    char *out = (char *)malloc((size_t)len * 3 + 1);
    if (!out) { *out_len = 0; return NULL; }
    int pos = 0;
    for (int i = 0; i < len; i++) {
        uint32_t cp = byte_to_cp[(uint8_t)input[i]];
        pos += bn_cp_to_utf8(cp, out + pos);
    }
    out[pos] = '\0';
    *out_len = pos;
    return out;
}

static char *bn_gpt2_to_bytes(const char *input, int len, int *out_len) {
    bn_init_byte_tables();
    char *out = (char *)malloc((size_t)len + 1);
    if (!out) { *out_len = 0; return NULL; }
    int pos = 0;
    int i = 0;
    while (i < len) {
        int adv;
        uint32_t cp;
        if (!bn_utf8_to_cp(input + i, len - i, &cp, &adv)) {
            out[pos++] = input[i++];
            continue;
        }
        if (cp < 512) {
            out[pos++] = (char)cp_to_byte_map[cp];
        } else {
            memcpy(out + pos, input + i, (size_t)adv);
            pos += adv;
        }
        i += adv;
    }
    out[pos] = '\0';
    *out_len = pos;
    return out;
}

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
        uint64_t hash;
        int      rank;
        int      next;
    } *mt;
    int *mt_buckets;
    int  mt_count;

    int bos_id;
    int eos_id;
};

static uint64_t bn_hash_str(const char *s, int len) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < len; i++) {
        h ^= (uint8_t)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t bn_hash_pair(const char *a, int alen, const char *b, int blen) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < alen; i++) { h ^= (uint8_t)a[i]; h *= 1099511628211ULL; }
    h ^= 0xff; h *= 1099511628211ULL;
    for (int i = 0; i < blen; i++) { h ^= (uint8_t)b[i]; h *= 1099511628211ULL; }
    return h;
}

static int bn_ht_init(bn_tokenizer_t *t, int cap) {
    if (cap <= 0) { t->ht_size = 0; return 0; }
    t->ht_size = cap;
    t->ht = (struct bn_ht_entry *)calloc((size_t)cap, sizeof(t->ht[0]));
    t->ht_buckets = (int *)malloc((size_t)cap * sizeof(int));
    if (!t->ht || !t->ht_buckets) {
        free(t->ht); t->ht = NULL;
        free(t->ht_buckets); t->ht_buckets = NULL;
        t->ht_size = 0;
        return 0;
    }
    for (int i = 0; i < cap; i++) t->ht_buckets[i] = -1;
    t->ht_count = 0;
    return 1;
}

static void bn_ht_insert(bn_tokenizer_t *t, const char *key, int len, int val) {
    uint64_t h = bn_hash_str(key, len);
    int bucket = (int)(h % (uint64_t)t->ht_size);
    int idx = t->ht_count++;
    t->ht[idx].key = (char *)malloc((size_t)len + 1);
    memcpy(t->ht[idx].key, key, (size_t)len);
    t->ht[idx].key[len] = '\0';
    t->ht[idx].val = val;
    t->ht[idx].next = t->ht_buckets[bucket];
    t->ht_buckets[bucket] = idx;
}

static int bn_ht_lookup(bn_tokenizer_t *t, const char *key, int len) {
    if (t->ht_size == 0) return -1;
    uint64_t h = bn_hash_str(key, len);
    int bucket = (int)(h % (uint64_t)t->ht_size);
    for (int idx = t->ht_buckets[bucket]; idx >= 0; idx = t->ht[idx].next) {
        if ((int)strlen(t->ht[idx].key) == len &&
            memcmp(t->ht[idx].key, key, (size_t)len) == 0) {
            return t->ht[idx].val;
        }
    }
    return -1;
}

static int bn_mt_init(bn_tokenizer_t *t, int cap) {
    if (cap <= 0) { t->mt_size = 0; return 0; }
    t->mt_size = cap;
    t->mt = (struct bn_mt_entry *)calloc((size_t)cap, sizeof(t->mt[0]));
    t->mt_buckets = (int *)malloc((size_t)cap * sizeof(int));
    if (!t->mt || !t->mt_buckets) {
        free(t->mt); t->mt = NULL;
        free(t->mt_buckets); t->mt_buckets = NULL;
        t->mt_size = 0;
        return 0;
    }
    for (int i = 0; i < cap; i++) t->mt_buckets[i] = -1;
    t->mt_count = 0;
    return 1;
}

static void bn_mt_insert(bn_tokenizer_t *t, const char *a, int alen,
                          const char *b, int blen, int rank) {
    uint64_t h = bn_hash_pair(a, alen, b, blen);
    int bucket = (int)(h % (uint64_t)t->mt_size);
    int idx = t->mt_count++;
    t->mt[idx].hash = h;
    t->mt[idx].rank = rank;
    t->mt[idx].next = t->mt_buckets[bucket];
    t->mt_buckets[bucket] = idx;
}

static int bn_mt_lookup(bn_tokenizer_t *t, const char *a, int alen,
                         const char *b, int blen) {
    if (t->mt_size == 0) return -1;
    uint64_t h = bn_hash_pair(a, alen, b, blen);
    int bucket = (int)(h % (uint64_t)t->mt_size);
    for (int idx = t->mt_buckets[bucket]; idx >= 0; idx = t->mt[idx].next) {
        if (t->mt[idx].hash == h) return t->mt[idx].rank;
    }
    return -1;
}

typedef struct {
    int  prev;
    int  next;
    const char *text;
    int  len;
} bn_symbol_t;

typedef struct {
    int left;
    int right;
    int rank;
} bn_bigram_t;

typedef struct {
    bn_bigram_t *data;
    int size;
    int cap;
} bn_heap_t;

static int bn_heap_push(bn_heap_t *h, bn_bigram_t bg) {
    if (h->size >= h->cap) {
        int new_cap = h->cap ? h->cap * 2 : 64;
        bn_bigram_t *tmp = (bn_bigram_t *)realloc(h->data, (size_t)new_cap * sizeof(bn_bigram_t));
        if (!tmp) return 0;
        h->data = tmp;
        h->cap = new_cap;
    }
    int i = h->size++;
    h->data[i] = bg;
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (h->data[parent].rank <= h->data[i].rank) break;
        bn_bigram_t tmp = h->data[i];
        h->data[i] = h->data[parent];
        h->data[parent] = tmp;
        i = parent;
    }
    return 1;
}

static bn_bigram_t bn_heap_pop(bn_heap_t *h) {
    bn_bigram_t top = h->data[0];
    h->data[0] = h->data[--h->size];
    int i = 0;
    for (;;) {
        int l = 2*i+1, r = 2*i+2, smallest = i;
        if (l < h->size && h->data[l].rank < h->data[smallest].rank) smallest = l;
        if (r < h->size && h->data[r].rank < h->data[smallest].rank) smallest = r;
        if (smallest == i) break;
        bn_bigram_t tmp = h->data[i];
        h->data[i] = h->data[smallest];
        h->data[smallest] = tmp;
        i = smallest;
    }
    return top;
}

static int bn_utf8_len(uint8_t c) {
    if (c < 0x80) return 1;
    if (c < 0xE0) return 2;
    if (c < 0xF0) return 3;
    return 4;
}

static int bn_bpe_encode(bn_tokenizer_t *t, const char *text, int text_len,
                          int *output, int max_out)
{
    if (text_len == 0) return 0;

    int n_chars = 0;
    int pos = 0;
    while (pos < text_len) {
        int clen = bn_utf8_len((uint8_t)text[pos]);
        if (pos + clen > text_len) break;
        n_chars++;
        pos += clen;
    }

    if (n_chars == 0) return 0;

    if (n_chars == 1) {
        int id = bn_ht_lookup(t, text, text_len);
        if (id >= 0 && max_out > 0) {
            output[0] = id;
            return 1;
        }
        int n = 0;
        for (int i = 0; i < text_len && n < max_out; i++) {
            char buf[16];
            int blen = snprintf(buf, sizeof(buf), "<0x%02X>", (uint8_t)text[i]);
            int bid = bn_ht_lookup(t, buf, blen);
            if (bid >= 0) output[n++] = bid;
        }
        return n;
    }

    bn_symbol_t *syms = (bn_symbol_t *)malloc((size_t)n_chars * 2 * sizeof(bn_symbol_t));
    if (!syms) return 0;
    int n_syms = n_chars;
    pos = 0;
    for (int i = 0; i < n_chars; i++) {
        int clen = bn_utf8_len((uint8_t)text[pos]);
        syms[i].text = text + pos;
        syms[i].len  = clen;
        syms[i].prev = i - 1;
        syms[i].next = (i + 1 < n_chars) ? i + 1 : -1;
        pos += clen;
    }

    bn_heap_t heap = {0};
    for (int i = 0; i < n_chars - 1; i++) {
        int j = syms[i].next;
        if (j < 0) continue;
        int rank = bn_mt_lookup(t, syms[i].text, syms[i].len,
                                 syms[j].text, syms[j].len);
        if (rank >= 0) {
            bn_bigram_t bg = { .left = i, .right = j, .rank = rank };
            bn_heap_push(&heap, bg);
        }
    }

    while (heap.size > 0) {
        bn_bigram_t bg = bn_heap_pop(&heap);
        bn_symbol_t *left  = &syms[bg.left];
        bn_symbol_t *right = &syms[bg.right];

        if (left->len == 0 || right->len == 0) continue;
        if (left->next != bg.right) continue;

        left->len += right->len;
        right->len = 0;
        left->next = right->next;
        if (right->next >= 0)
            syms[right->next].prev = bg.left;

        if (left->prev >= 0) {
            bn_symbol_t *prev = &syms[left->prev];
            int rank = bn_mt_lookup(t, prev->text, prev->len,
                                     left->text, left->len);
            if (rank >= 0) {
                bn_bigram_t nbg = { .left = left->prev, .right = bg.left, .rank = rank };
                bn_heap_push(&heap, nbg);
            }
        }

        if (left->next >= 0) {
            bn_symbol_t *nxt = &syms[left->next];
            int rank = bn_mt_lookup(t, left->text, left->len,
                                     nxt->text, nxt->len);
            if (rank >= 0) {
                bn_bigram_t nbg = { .left = bg.left, .right = left->next, .rank = rank };
                bn_heap_push(&heap, nbg);
            }
        }
    }

    int n_out = 0;
    int cur = -1;
    for (int i = 0; i < n_syms; i++) {
        if (syms[i].len > 0 && (i == 0 || syms[i].prev < 0 || syms[syms[i].prev].len == 0)) {
            cur = i;
            break;
        }
    }

    while (cur >= 0 && n_out < max_out) {
        if (syms[cur].len > 0) {
            int id = bn_ht_lookup(t, syms[cur].text, syms[cur].len);
            if (id >= 0) {
                output[n_out++] = id;
            } else {
                for (int b = 0; b < syms[cur].len && n_out < max_out; b++) {
                    char buf[16];
                    int blen = snprintf(buf, sizeof(buf), "<0x%02X>",
                                        (uint8_t)syms[cur].text[b]);
                    int bid = bn_ht_lookup(t, buf, blen);
                    if (bid >= 0) output[n_out++] = bid;
                }
            }
        }
        cur = syms[cur].next;
    }

    free(heap.data);
    free(syms);
    return n_out;
}

static int bn_is_letter(uint8_t c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80;
}

static int bn_is_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static int bn_is_ws(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

typedef struct { int off; int len; } bn_span_t;

static int bn_pre_tokenize(const char *text, int text_len,
                           bn_span_t *spans, int max_spans)
{
    int n = 0;
    int i = 0;

    while (i < text_len && n < max_spans) {
        uint8_t c = (uint8_t)text[i];

        if (c == '\'' && i + 1 < text_len) {
            uint8_t c2 = (uint8_t)text[i+1];
            int clen = 0;
            if (c2 == 's' || c2 == 't' || c2 == 'm' || c2 == 'd' ||
                c2 == 'S' || c2 == 'T' || c2 == 'M' || c2 == 'D') {
                clen = 2;
            } else if (i + 2 < text_len) {
                char low2 = (char)(c2 | 0x20);
                char low3 = (char)((uint8_t)text[i+2] | 0x20);
                if (low2 == 'r' && low3 == 'e') clen = 3;
                else if (low2 == 'v' && low3 == 'e') clen = 3;
                else if (low2 == 'l' && low3 == 'l') clen = 3;
            }
            if (clen > 0) {
                spans[n].off = i;
                spans[n].len = clen;
                n++;
                i += clen;
                continue;
            }
        }

        if (c == ' ' && i + 1 < text_len && bn_is_letter((uint8_t)text[i+1])) {
            int start = i;
            i++;
            while (i < text_len && bn_is_letter((uint8_t)text[i])) {
                int clen = bn_utf8_len((uint8_t)text[i]);
                i += clen;
            }
            spans[n].off = start;
            spans[n].len = i - start;
            n++;
            continue;
        }

        if (bn_is_letter(c)) {
            int start = i;
            while (i < text_len && bn_is_letter((uint8_t)text[i])) {
                int clen = bn_utf8_len((uint8_t)text[i]);
                i += clen;
            }
            spans[n].off = start;
            spans[n].len = i - start;
            n++;
            continue;
        }

        if (c == ' ' && i + 1 < text_len && bn_is_digit((uint8_t)text[i+1])) {
            int start = i;
            i++;
            int digit_count = 0;
            while (i < text_len && bn_is_digit((uint8_t)text[i]) && digit_count < 3) {
                i++;
                digit_count++;
            }
            spans[n].off = start;
            spans[n].len = i - start;
            n++;
            continue;
        }

        if (bn_is_digit(c)) {
            int start = i;
            int digit_count = 0;
            while (i < text_len && bn_is_digit((uint8_t)text[i]) && digit_count < 3) {
                i++;
                digit_count++;
            }
            spans[n].off = start;
            spans[n].len = i - start;
            n++;
            continue;
        }

        if (bn_is_ws(c)) {
            int start = i;
            while (i < text_len && bn_is_ws((uint8_t)text[i])) i++;
            spans[n].off = start;
            spans[n].len = i - start;
            n++;
            continue;
        }

        if (c == ' ' && i + 1 < text_len &&
            !bn_is_letter((uint8_t)text[i+1]) &&
            !bn_is_digit((uint8_t)text[i+1]) &&
            !bn_is_ws((uint8_t)text[i+1])) {
            int start = i;
            i++;
            while (i < text_len &&
                   !bn_is_letter((uint8_t)text[i]) &&
                   !bn_is_digit((uint8_t)text[i]) &&
                   !bn_is_ws((uint8_t)text[i])) {
                i++;
            }
            spans[n].off = start;
            spans[n].len = i - start;
            n++;
            continue;
        }

        {
            int start = i;
            int clen = bn_utf8_len(c);
            i += clen;
            spans[n].off = start;
            spans[n].len = i - start;
            n++;
        }
    }
    return n;
}

bn_tokenizer_t *bn_tokenizer_create(bn_gguf_t *g) {
    bn_tokenizer_t *t = (bn_tokenizer_t *)calloc(1, sizeof(bn_tokenizer_t));
    if (!t) return NULL;

    bn_gguf_kv_t *kv_tokens = bn_gguf_find_kv(g, "tokenizer.ggml.tokens");
    if (!kv_tokens || kv_tokens->type != BN_GGUF_TYPE_ARRAY) {
        fprintf(stderr, "tokenizer: missing tokens array\n");
        free(t);
        return NULL;
    }
    if (kv_tokens->val.arr.type != BN_GGUF_TYPE_STRING) {
        fprintf(stderr,
                "tokenizer: tokenizer.ggml.tokens must be an array of string, got array of %s\n",
                bn_gguf_type_name(kv_tokens->val.arr.type));
        free(t);
        return NULL;
    }

    t->n_vocab = (int)kv_tokens->val.arr.len;
    if (t->n_vocab <= 0) {
        fprintf(stderr, "tokenizer: token vocabulary is empty\n");
        free(t);
        return NULL;
    }
    char **token_strs = (char **)kv_tokens->val.arr.data;

    t->vocab = (char **)malloc((size_t)t->n_vocab * sizeof(char *));
    t->vocab_len = (int *)malloc((size_t)t->n_vocab * sizeof(int));
    if (!t->vocab || !t->vocab_len) {
        free(t->vocab);
        free(t->vocab_len);
        free(t);
        return NULL;
    }

    int ht_cap = t->n_vocab * 4;
    if (!bn_ht_init(t, ht_cap)) {
        free(t->vocab);
        free(t->vocab_len);
        free(t);
        return NULL;
    }

    for (int i = 0; i < t->n_vocab; i++) {
        int slen = (int)strlen(token_strs[i]);
        t->vocab[i] = (char *)malloc((size_t)slen + 1);
        if (!t->vocab[i]) {
            t->n_vocab = i;
            bn_tokenizer_free(t);
            return NULL;
        }
        memcpy(t->vocab[i], token_strs[i], (size_t)slen + 1);
        t->vocab_len[i] = slen;
        bn_ht_insert(t, token_strs[i], slen, i);
    }

    bn_gguf_kv_t *kv_merges = bn_gguf_find_kv(g, "tokenizer.ggml.merges");
    if (kv_merges && kv_merges->type != BN_GGUF_TYPE_ARRAY) {
        fprintf(stderr,
                "tokenizer: tokenizer.ggml.merges must be an array of string, got %s\n",
                bn_gguf_type_name(kv_merges->type));
        bn_tokenizer_free(t);
        return NULL;
    }
    if (kv_merges && kv_merges->val.arr.type != BN_GGUF_TYPE_STRING) {
        fprintf(stderr,
                "tokenizer: tokenizer.ggml.merges must be an array of string, got array of %s\n",
                bn_gguf_type_name(kv_merges->val.arr.type));
        bn_tokenizer_free(t);
        return NULL;
    }
    if (kv_merges && kv_merges->type == BN_GGUF_TYPE_ARRAY) {
        t->n_merges = (int)kv_merges->val.arr.len;
        char **merge_strs = (char **)kv_merges->val.arr.data;

        if (t->n_merges > 0) {
            t->merges = (bn_merge_t *)malloc((size_t)t->n_merges * sizeof(bn_merge_t));
            if (!t->merges) {
                bn_tokenizer_free(t);
                return NULL;
            }
            int mt_cap = t->n_merges * 4;
            if (!bn_mt_init(t, mt_cap)) {
                bn_tokenizer_free(t);
                return NULL;
            }
        }

        for (int i = 0; i < t->n_merges; i++) {
            char *s = merge_strs[i];
            char *space = strchr(s, ' ');
            if (!space) continue;
            int left_len = (int)(space - s);
            int right_len = (int)strlen(space + 1);

            t->merges[i].left  = (char *)malloc((size_t)left_len + 1);
            t->merges[i].right = (char *)malloc((size_t)right_len + 1);
            memcpy(t->merges[i].left, s, (size_t)left_len);
            t->merges[i].left[left_len] = '\0';
            memcpy(t->merges[i].right, space + 1, (size_t)right_len);
            t->merges[i].right[right_len] = '\0';
            t->merges[i].rank = i;

            bn_mt_insert(t, t->merges[i].left, left_len,
                         t->merges[i].right, right_len, i);
        }
    }

    bn_gguf_kv_t *kv_bos = bn_gguf_find_kv(g, "tokenizer.ggml.bos_token_id");
    bn_gguf_kv_t *kv_eos = bn_gguf_find_kv(g, "tokenizer.ggml.eos_token_id");
    if (kv_bos && kv_bos->type != BN_GGUF_TYPE_UINT32) {
        fprintf(stderr,
                "tokenizer: tokenizer.ggml.bos_token_id must have type uint32, got %s\n",
                bn_gguf_type_name(kv_bos->type));
        bn_tokenizer_free(t);
        return NULL;
    }
    if (kv_eos && kv_eos->type != BN_GGUF_TYPE_UINT32) {
        fprintf(stderr,
                "tokenizer: tokenizer.ggml.eos_token_id must have type uint32, got %s\n",
                bn_gguf_type_name(kv_eos->type));
        bn_tokenizer_free(t);
        return NULL;
    }
    t->bos_id = kv_bos ? (int)kv_bos->val.u32 : 128000;
    t->eos_id = kv_eos ? (int)kv_eos->val.u32 : 128001;

    fprintf(stderr, "tokenizer: loaded %d tokens, %d merges, bos=%d, eos=%d\n",
            t->n_vocab, t->n_merges, t->bos_id, t->eos_id);
    return t;
}

void bn_tokenizer_free(bn_tokenizer_t *t) {
    if (!t) return;
    for (int i = 0; i < t->n_vocab; i++) free(t->vocab[i]);
    free(t->vocab);
    free(t->vocab_len);
    for (int i = 0; i < t->n_merges; i++) {
        free(t->merges[i].left);
        free(t->merges[i].right);
    }
    free(t->merges);
    for (int i = 0; i < t->ht_count; i++) free(t->ht[i].key);
    free(t->ht);
    free(t->ht_buckets);
    free(t->mt);
    free(t->mt_buckets);
    free(t);
}

int bn_tokenize(bn_tokenizer_t *t, const char *text,
                int *tokens, int max_tokens)
{
    if (!t) {
        fprintf(stderr, "tokenize: tokenizer is NULL\n");
        return -1;
    }
    if (!text) {
        fprintf(stderr, "tokenize: text is NULL\n");
        return -1;
    }
    if (max_tokens < 0) {
        fprintf(stderr, "tokenize: max_tokens must be >= 0\n");
        return -1;
    }
    if (max_tokens > 0 && !tokens) {
        fprintf(stderr, "tokenize: tokens is NULL with positive capacity\n");
        return -1;
    }

    int text_len = (int)strlen(text);
    int n_out = 0;

    if (n_out < max_tokens) tokens[n_out++] = t->bos_id;

    int gpt2_len;
    char *gpt2_text = bn_bytes_to_gpt2(text, text_len, &gpt2_len);
    if (!gpt2_text) return -1;

    int max_spans = gpt2_len + 1;
    if (max_spans < 16) max_spans = 16;
    bn_span_t *spans = (bn_span_t *)malloc((size_t)max_spans * sizeof(bn_span_t));
    if (!spans) { free(gpt2_text); return -1; }
    int n_spans = bn_pre_tokenize(gpt2_text, gpt2_len, spans, max_spans);

    for (int i = 0; i < n_spans && n_out < max_tokens; i++) {
        int added = bn_bpe_encode(t, gpt2_text + spans[i].off, spans[i].len,
                                   tokens + n_out, max_tokens - n_out);
        n_out += added;
    }

    free(spans);
    free(gpt2_text);
    return n_out;
}

char *bn_detokenize(bn_tokenizer_t *t, const int *tokens, int n) {
    if (!t) return NULL;
    int total = 0;
    for (int i = 0; i < n; i++) {
        int id = tokens[i];
        if (id >= 0 && id < t->n_vocab) {
            total += t->vocab_len[id];
        }
    }

    char *gpt2 = (char *)malloc((size_t)total + 1);
    if (!gpt2) return NULL;
    int pos = 0;
    for (int i = 0; i < n; i++) {
        int id = tokens[i];
        if (id >= 0 && id < t->n_vocab) {
            memcpy(gpt2 + pos, t->vocab[id], (size_t)t->vocab_len[id]);
            pos += t->vocab_len[id];
        }
    }
    gpt2[pos] = '\0';

    int out_len;
    char *out = bn_gpt2_to_bytes(gpt2, pos, &out_len);
    free(gpt2);
    return out;
}

int bn_token_bos(bn_tokenizer_t *t) { return t ? t->bos_id : -1; }
int bn_token_eos(bn_tokenizer_t *t) { return t ? t->eos_id : -1; }

const char *bn_token_text(bn_tokenizer_t *t, int id) {
    if (!t) return "";
    if (id < 0 || id >= t->n_vocab) return "";
    static _Thread_local char decoded_buf[256];
    const char *tok = t->vocab[id];
    int tok_len = t->vocab_len[id];
    int out_len;
    char *dec = bn_gpt2_to_bytes(tok, tok_len, &out_len);
    if (!dec) { decoded_buf[0] = '\0'; return decoded_buf; }
    if (out_len >= 256) out_len = 255;
    memcpy(decoded_buf, dec, (size_t)out_len);
    decoded_buf[out_len] = '\0';
    free(dec);
    return decoded_buf;
}
