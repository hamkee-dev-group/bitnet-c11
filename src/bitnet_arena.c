#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
 * Creates a zero-initialized arena backed by a contiguous heap allocation.
 *
 * Parameters:
 *   size - Number of bytes to reserve for arena allocations.
 *
 * Returns:
 *   A new arena. If allocation fails, the returned arena has a NULL base
 *   pointer, a size of 0, and used set to 0.
 */
bn_arena_t bn_arena_create(size_t size) {
    bn_arena_t a;
    a.base = NULL;
    if (size > 0 && posix_memalign((void **)&a.base, 64, size) == 0) {
        memset(a.base, 0, size);
    }
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
    if (a == NULL) return NULL;

    size_t align = 64;

    /* Detect overflow in alignment rounding: a->used + align - 1 */
    if (a->used > SIZE_MAX - (align - 1))
        return NULL;
    size_t offset = (a->used + align - 1) & ~(align - 1);

    /* Detect overflow in offset + size */
    if (size > SIZE_MAX - offset || offset + size > a->size) {
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
