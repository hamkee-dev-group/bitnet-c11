#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

bn_arena_t bn_arena_create(size_t size) {
    bn_arena_t a;
    a.base = (uint8_t *)calloc(1, size);
    a.size = a.base ? size : 0;
    a.used = 0;
    return a;
}

void bn_arena_free(bn_arena_t *a) {
    free(a->base);
    a->base = NULL;
    a->size = 0;
    a->used = 0;
}

void *bn_arena_alloc(bn_arena_t *a, size_t size) {
    size_t align = 64;
    size_t offset = (a->used + align - 1) & ~(align - 1);
    if (offset + size > a->size) {
        fprintf(stderr, "bn_arena: out of memory (need %zu, have %zu)\n",
                offset + size, a->size);
        return NULL;
    }
    void *ptr = a->base + offset;
    a->used = offset + size;
    return ptr;
}

void bn_arena_reset(bn_arena_t *a) {
    a->used = 0;
}
