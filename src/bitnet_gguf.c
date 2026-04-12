#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

typedef struct {
    const uint8_t *base;
    const uint8_t *ptr;
    const uint8_t *end;
} bn_reader_t;

static inline bool bn_read_raw(bn_reader_t *r, void *dst, size_t n) {
    if (r->ptr + n > r->end) return false;
    memcpy(dst, r->ptr, n);
    r->ptr += n;
    return true;
}

static inline size_t bn_reader_remaining(const bn_reader_t *r) {
    return (size_t)(r->end - r->ptr);
}

static inline bool bn_read_u8(bn_reader_t *r, uint8_t *v) {
    return bn_read_raw(r, v, 1);
}
static inline bool bn_read_u16(bn_reader_t *r, uint16_t *v) {
    return bn_read_raw(r, v, 2);
}
static inline bool bn_read_u32(bn_reader_t *r, uint32_t *v) {
    return bn_read_raw(r, v, 4);
}
static inline bool bn_read_u64(bn_reader_t *r, uint64_t *v) {
    return bn_read_raw(r, v, 8);
}
static inline bool bn_read_i8(bn_reader_t *r, int8_t *v) {
    return bn_read_raw(r, v, 1);
}
static inline bool bn_read_i16(bn_reader_t *r, int16_t *v) {
    return bn_read_raw(r, v, 2);
}
static inline bool bn_read_i32(bn_reader_t *r, int32_t *v) {
    return bn_read_raw(r, v, 4);
}
static inline bool bn_read_i64(bn_reader_t *r, int64_t *v) {
    return bn_read_raw(r, v, 8);
}
static inline bool bn_read_f32(bn_reader_t *r, float *v) {
    return bn_read_raw(r, v, 4);
}
static inline bool bn_read_f64(bn_reader_t *r, double *v) {
    return bn_read_raw(r, v, 8);
}
static inline bool bn_read_bool(bn_reader_t *r, bool *v) {
    uint8_t b;
    if (!bn_read_u8(r, &b)) return false;
    *v = (b != 0);
    return true;
}

static char *bn_read_string(bn_reader_t *r) {
    uint64_t len;
    if (!bn_read_u64(r, &len)) return NULL;
    if (r->ptr + len > r->end) return NULL;
    char *s = (char *)malloc(len + 1);
    if (!s) return NULL;
    memcpy(s, r->ptr, len);
    s[len] = '\0';
    r->ptr += len;
    return s;
}

static size_t bn_gguf_type_size(uint32_t type) {
    switch (type) {
    case BN_GGUF_TYPE_UINT8:   return 1;
    case BN_GGUF_TYPE_INT8:    return 1;
    case BN_GGUF_TYPE_UINT16:  return 2;
    case BN_GGUF_TYPE_INT16:   return 2;
    case BN_GGUF_TYPE_UINT32:  return 4;
    case BN_GGUF_TYPE_INT32:   return 4;
    case BN_GGUF_TYPE_FLOAT32: return 4;
    case BN_GGUF_TYPE_BOOL:    return 1;
    case BN_GGUF_TYPE_UINT64:  return 8;
    case BN_GGUF_TYPE_INT64:   return 8;
    case BN_GGUF_TYPE_FLOAT64: return 8;
    default: return 0;
    }
}

static bool bn_mul_overflow_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) return true;
    *out = a * b;
    return false;
}

static bool bn_ggml_tensor_nbytes(const bn_gguf_tensor_t *t, size_t *nbytes) {
    size_t n_elem = 1;

    for (uint32_t d = 0; d < t->n_dims; d++) {
        if (t->ne[d] > SIZE_MAX) return false;
        if (bn_mul_overflow_size(n_elem, (size_t)t->ne[d], &n_elem)) return false;
    }

    switch (t->type) {
    case BN_GGML_TYPE_F32:
        return !bn_mul_overflow_size(n_elem, sizeof(float), nbytes);
    case BN_GGML_TYPE_F16:
        return !bn_mul_overflow_size(n_elem, sizeof(uint16_t), nbytes);
    case BN_GGML_TYPE_I2_S: {
        size_t packed_bytes;
        if ((n_elem % 4) != 0) return false;
        packed_bytes = n_elem / 4;
        if (packed_bytes > SIZE_MAX - sizeof(float)) return false;
        *nbytes = packed_bytes + sizeof(float);
        return true;
    }
    default:
        return false;
    }
}

static void bn_log_kv_read_failure(const bn_reader_t *r, const bn_gguf_kv_t *kv,
                                   const char *op, const char *why) {
    fprintf(stderr, "bn_read_kv_value: key '%s' type %u failed to read %s: %s "
                    "(remaining=%zu bytes)\n",
            kv->key ? kv->key : "<null>", kv->type, op, why,
            bn_reader_remaining(r));
}

static bool bn_read_kv_value(bn_reader_t *r, bn_gguf_kv_t *kv) {
    switch (kv->type) {
    case BN_GGUF_TYPE_UINT8:
        if (!bn_read_u8(r, &kv->val.u8)) {
            bn_log_kv_read_failure(r, kv, "uint8 value", "unexpected end of input");
            return false;
        }
        return true;
    case BN_GGUF_TYPE_INT8:
        if (!bn_read_i8(r, &kv->val.i8)) {
            bn_log_kv_read_failure(r, kv, "int8 value", "unexpected end of input");
            return false;
        }
        return true;
    case BN_GGUF_TYPE_UINT16:
        if (!bn_read_u16(r, &kv->val.u16)) {
            bn_log_kv_read_failure(r, kv, "uint16 value", "unexpected end of input");
            return false;
        }
        return true;
    case BN_GGUF_TYPE_INT16:
        if (!bn_read_i16(r, &kv->val.i16)) {
            bn_log_kv_read_failure(r, kv, "int16 value", "unexpected end of input");
            return false;
        }
        return true;
    case BN_GGUF_TYPE_UINT32:
        if (!bn_read_u32(r, &kv->val.u32)) {
            bn_log_kv_read_failure(r, kv, "uint32 value", "unexpected end of input");
            return false;
        }
        return true;
    case BN_GGUF_TYPE_INT32:
        if (!bn_read_i32(r, &kv->val.i32)) {
            bn_log_kv_read_failure(r, kv, "int32 value", "unexpected end of input");
            return false;
        }
        return true;
    case BN_GGUF_TYPE_FLOAT32:
        if (!bn_read_f32(r, &kv->val.f32)) {
            bn_log_kv_read_failure(r, kv, "float32 value", "unexpected end of input");
            return false;
        }
        return true;
    case BN_GGUF_TYPE_BOOL:
        if (!bn_read_bool(r, &kv->val.b)) {
            bn_log_kv_read_failure(r, kv, "bool value", "unexpected end of input");
            return false;
        }
        return true;
    case BN_GGUF_TYPE_UINT64:
        if (!bn_read_u64(r, &kv->val.u64)) {
            bn_log_kv_read_failure(r, kv, "uint64 value", "unexpected end of input");
            return false;
        }
        return true;
    case BN_GGUF_TYPE_INT64:
        if (!bn_read_i64(r, &kv->val.i64)) {
            bn_log_kv_read_failure(r, kv, "int64 value", "unexpected end of input");
            return false;
        }
        return true;
    case BN_GGUF_TYPE_FLOAT64:
        if (!bn_read_f64(r, &kv->val.f64)) {
            bn_log_kv_read_failure(r, kv, "float64 value", "unexpected end of input");
            return false;
        }
        return true;
    case BN_GGUF_TYPE_STRING: {
        char *s = bn_read_string(r);
        if (!s) {
            bn_log_kv_read_failure(r, kv, "string value",
                                   "failed to read string length/data or allocate buffer");
            return false;
        }
        kv->val.str.data = s;
        kv->val.str.len = strlen(s);
        return true;
    }
    case BN_GGUF_TYPE_ARRAY: {
        uint32_t atype;
        uint64_t alen;
        if (!bn_read_u32(r, &atype)) {
            bn_log_kv_read_failure(r, kv, "array element type", "unexpected end of input");
            return false;
        }
        if (!bn_read_u64(r, &alen)) {
            bn_log_kv_read_failure(r, kv, "array length", "unexpected end of input");
            return false;
        }
        kv->val.arr.type = atype;
        kv->val.arr.len  = alen;
        if (atype == BN_GGUF_TYPE_STRING) {
            char **strs = (char **)calloc(alen, sizeof(char *));
            if (!strs) {
                bn_log_kv_read_failure(r, kv, "string array storage", "allocation failed");
                return false;
            }
            for (uint64_t i = 0; i < alen; i++) {
                strs[i] = bn_read_string(r);
                if (!strs[i]) {
                    bn_log_kv_read_failure(r, kv, "string array element",
                                           "failed to read string length/data or allocate buffer");
                    for (uint64_t j = 0; j < i; j++) free(strs[j]);
                    free(strs);
                    return false;
                }
            }
            kv->val.arr.data = strs;
        } else {
            size_t elem_sz = bn_gguf_type_size(atype);
            if (elem_sz == 0) {
                bn_log_kv_read_failure(r, kv, "array element type", "unsupported array element type");
                return false;
            }
            if (alen > SIZE_MAX / elem_sz) {
                bn_log_kv_read_failure(r, kv, "array payload", "size overflow");
                return false;
            }
            size_t total = elem_sz * (size_t)alen;
            if (total > bn_reader_remaining(r)) {
                bn_log_kv_read_failure(r, kv, "array payload", "unexpected end of input");
                return false;
            }
            void *data = malloc(total);
            if (!data) {
                bn_log_kv_read_failure(r, kv, "array payload buffer", "allocation failed");
                return false;
            }
            memcpy(data, r->ptr, total);
            r->ptr += total;
            kv->val.arr.data = data;
        }
        return true;
    }
    default:
        bn_log_kv_read_failure(r, kv, "value", "unsupported value type");
        return false;
    }
}

/**
 * Opens and parses a GGUF model file into a mapped `bn_gguf_t` handle.
 *
 * @param path Path to the GGUF file to open.
 * @return A newly allocated `bn_gguf_t` on success, or `NULL` if the path is
 * invalid, the file cannot be opened or mapped, or the GGUF contents are not
 * supported.
 */
bn_gguf_t *bn_gguf_open(const char *path) {
    if (!path || path[0] == '\0') {
        fprintf(stderr, "bn_gguf_open: invalid path\n");
        return NULL;
    }

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("bn_gguf_open: open");
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("bn_gguf_open: fstat");
        close(fd);
        return NULL;
    }

    size_t file_size = (size_t)st.st_size;
    void *addr = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        perror("bn_gguf_open: mmap");
        close(fd);
        return NULL;
    }

    bn_reader_t reader = {
        .base = (const uint8_t *)addr,
        .ptr  = (const uint8_t *)addr,
        .end  = (const uint8_t *)addr + file_size,
    };

    char magic[4];
    if (!bn_read_raw(&reader, magic, 4) ||
        memcmp(magic, "GGUF", 4) != 0) {
        fprintf(stderr, "bn_gguf_open: bad magic\n");
        goto fail;
    }

    bn_gguf_t *g = (bn_gguf_t *)calloc(1, sizeof(bn_gguf_t));
    if (!g) goto fail;

    g->mmap_addr = addr;
    g->mmap_len  = file_size;
    g->fd        = fd;

    if (!bn_read_u32(&reader, &g->version)) goto fail_g;
    if (g->version < 2 || g->version > 3) {
        fprintf(stderr, "bn_gguf_open: unsupported version %u\n", g->version);
        goto fail_g;
    }

    if (!bn_read_u64(&reader, &g->n_tensors)) goto fail_g;
    if (!bn_read_u64(&reader, &g->n_kv))      goto fail_g;

    g->kvs = (bn_gguf_kv_t *)calloc(g->n_kv, sizeof(bn_gguf_kv_t));
    if (g->n_kv > 0 && !g->kvs) {
        fprintf(stderr, "bn_gguf_open: failed to allocate kv table\n");
        goto fail_g;
    }

    for (uint64_t i = 0; i < g->n_kv; i++) {
        g->kvs[i].key = bn_read_string(&reader);
        if (!g->kvs[i].key) goto fail_g;
        if (!bn_read_u32(&reader, &g->kvs[i].type)) goto fail_g;
        if (!bn_read_kv_value(&reader, &g->kvs[i])) goto fail_g;
    }

    g->alignment = 32;
    bn_gguf_kv_t *align_kv = bn_gguf_find_kv(g, "general.alignment");
    if (align_kv) {
        if (align_kv->type != BN_GGUF_TYPE_UINT32) {
            fprintf(stderr, "bn_gguf_open: general.alignment has wrong type %u (expected UINT32)\n",
                    align_kv->type);
            goto fail_g;
        }
        g->alignment = align_kv->val.u32;
    }

    g->tensors = (bn_gguf_tensor_t *)calloc(g->n_tensors,
                                             sizeof(bn_gguf_tensor_t));
    if (!g->tensors) goto fail_g;

    for (uint64_t i = 0; i < g->n_tensors; i++) {
        bn_gguf_tensor_t *t = &g->tensors[i];
        t->name = bn_read_string(&reader);
        if (!t->name) goto fail_g;
        if (!bn_read_u32(&reader, &t->n_dims)) goto fail_g;
        if (t->n_dims > BN_GGUF_MAX_DIMS) {
            fprintf(stderr, "bn_gguf_open: tensor has too many dimensions (%u > %u)\n",
                    t->n_dims, BN_GGUF_MAX_DIMS);
            goto fail_g;
        }
        for (uint32_t d = 0; d < BN_GGUF_MAX_DIMS; d++) t->ne[d] = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            if (!bn_read_u64(&reader, &t->ne[d])) goto fail_g;
        }
        if (!bn_read_u32(&reader, &t->type)) goto fail_g;
        if (!bn_read_u64(&reader, &t->offset)) goto fail_g;
    }

    size_t header_size = (size_t)(reader.ptr - reader.base);
    if (g->alignment == 0 || (g->alignment & (g->alignment - 1)) != 0) {
        fprintf(stderr, "bn_gguf_open: invalid alignment %u (must be a power of two)\n",
                g->alignment);
        goto fail_g;
    }
    size_t pad = (g->alignment - (header_size % g->alignment)) % g->alignment;
    if (header_size > SIZE_MAX - pad) {
        fprintf(stderr, "bn_gguf_open: data offset overflow\n");
        goto fail_g;
    }
    g->data_offset = header_size + pad;
    if (g->data_offset > file_size) {
        fprintf(stderr, "bn_gguf_open: data section exceeds file size\n");
        goto fail_g;
    }

    for (uint64_t i = 0; i < g->n_tensors; i++) {
        size_t tensor_size;
        size_t tensor_begin;

        if (!bn_ggml_tensor_nbytes(&g->tensors[i], &tensor_size)) {
            fprintf(stderr, "bn_gguf_open: tensor '%s' has invalid shape/type size\n",
                    g->tensors[i].name);
            goto fail_g;
        }
        if (g->tensors[i].offset > SIZE_MAX) {
            fprintf(stderr, "bn_gguf_open: tensor '%s' offset overflow\n",
                    g->tensors[i].name);
            goto fail_g;
        }
        if (g->tensors[i].offset % g->alignment != 0) {
            fprintf(stderr, "bn_gguf_open: tensor '%s' offset %lu is not aligned to %u\n",
                    g->tensors[i].name, (unsigned long)g->tensors[i].offset,
                    g->alignment);
            goto fail_g;
        }

        tensor_begin = (size_t)g->data_offset + (size_t)g->tensors[i].offset;
        if ((size_t)g->data_offset > SIZE_MAX - (size_t)g->tensors[i].offset) {
            fprintf(stderr, "bn_gguf_open: tensor '%s' data range overflow\n",
                    g->tensors[i].name);
            goto fail_g;
        }
        if (tensor_size > file_size - tensor_begin) {
            fprintf(stderr, "bn_gguf_open: tensor '%s' data range exceeds file size\n",
                    g->tensors[i].name);
            goto fail_g;
        }
        g->tensors[i].data = (uint8_t *)addr + tensor_begin;
    }

    fprintf(stderr, "bn_gguf: loaded %s (v%u, %lu tensors, %lu kvs)\n",
            path, g->version,
            (unsigned long)g->n_tensors, (unsigned long)g->n_kv);
    return g;

fail_g:
    if (g) {
        if (g->kvs) {
            for (uint64_t i = 0; i < g->n_kv; i++) {
                free(g->kvs[i].key);
                if (g->kvs[i].type == BN_GGUF_TYPE_STRING) {
                    free(g->kvs[i].val.str.data);
                } else if (g->kvs[i].type == BN_GGUF_TYPE_ARRAY) {
                    if (g->kvs[i].val.arr.type == BN_GGUF_TYPE_STRING) {
                        char **strs = (char **)g->kvs[i].val.arr.data;
                        for (uint64_t j = 0; j < g->kvs[i].val.arr.len; j++)
                            free(strs[j]);
                    }
                    free(g->kvs[i].val.arr.data);
                }
            }
            free(g->kvs);
        }
        if (g->tensors) {
            for (uint64_t i = 0; i < g->n_tensors; i++)
                free(g->tensors[i].name);
            free(g->tensors);
        }
        free(g);
    }
fail:
    munmap(addr, file_size);
    close(fd);
    return NULL;
}

void bn_gguf_close(bn_gguf_t *g) {
    if (!g) return;
    for (uint64_t i = 0; i < g->n_kv; i++) {
        free(g->kvs[i].key);
        if (g->kvs[i].type == BN_GGUF_TYPE_STRING) {
            free(g->kvs[i].val.str.data);
        } else if (g->kvs[i].type == BN_GGUF_TYPE_ARRAY) {
            if (g->kvs[i].val.arr.type == BN_GGUF_TYPE_STRING) {
                char **strs = (char **)g->kvs[i].val.arr.data;
                for (uint64_t j = 0; j < g->kvs[i].val.arr.len; j++)
                    free(strs[j]);
            }
            free(g->kvs[i].val.arr.data);
        }
    }
    free(g->kvs);
    for (uint64_t i = 0; i < g->n_tensors; i++)
        free(g->tensors[i].name);
    free(g->tensors);
    if (g->mmap_addr) munmap(g->mmap_addr, g->mmap_len);
    if (g->fd >= 0) close(g->fd);
    free(g);
}

bn_gguf_kv_t *bn_gguf_find_kv(bn_gguf_t *g, const char *key) {
    for (uint64_t i = 0; i < g->n_kv; i++) {
        if (strcmp(g->kvs[i].key, key) == 0) return &g->kvs[i];
    }
    return NULL;
}

const char *bn_gguf_get_str(bn_gguf_t *g, const char *key) {
    bn_gguf_kv_t *kv = bn_gguf_find_kv(g, key);
    if (!kv) return NULL;
    if (kv->type != BN_GGUF_TYPE_STRING) return NULL;
    return kv->val.str.data;
}

uint32_t bn_gguf_get_u32(bn_gguf_t *g, const char *key) {
    bn_gguf_kv_t *kv = bn_gguf_find_kv(g, key);
    if (!kv || kv->type != BN_GGUF_TYPE_UINT32) return 0;
    return kv->val.u32;
}

float bn_gguf_get_f32(bn_gguf_t *g, const char *key) {
    bn_gguf_kv_t *kv = bn_gguf_find_kv(g, key);
    if (!kv || kv->type != BN_GGUF_TYPE_FLOAT32) return 0.0f;
    return kv->val.f32;
}

bn_gguf_tensor_t *bn_gguf_find_tensor(bn_gguf_t *g, const char *name) {
    if (g->n_tensors == 0) return NULL;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    }
    return NULL;
}
