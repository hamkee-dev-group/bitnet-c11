#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

static void test_alloc_null_arena_returns_null(void) {
    printf("Test: NULL arena pointer returns NULL...\n");
    assert(bn_arena_alloc(NULL, 64) == NULL);
    printf("  OK\n");
}

static void test_allocations_are_64_byte_aligned(void) {
    bn_arena_t arena = bn_arena_create(256);
    void *p1;
    void *p2;

    printf("Test: allocations are 64-byte aligned...\n");
    assert(arena.base != NULL);

    p1 = bn_arena_alloc(&arena, 1);
    p2 = bn_arena_alloc(&arena, 1);

    assert(p1 != NULL);
    assert(p2 != NULL);
    assert(((uintptr_t)p1 % 64) == 0);
    assert(((uintptr_t)p2 % 64) == 0);
    assert((uintptr_t)p2 - (uintptr_t)p1 == 64);

    bn_arena_free(&arena);
    printf("  OK\n");
}

static void test_reset_allows_reuse_from_start(void) {
    bn_arena_t arena = bn_arena_create(256);
    void *p1;
    void *p2;

    printf("Test: reset reuses arena from the start...\n");
    assert(arena.base != NULL);

    p1 = bn_arena_alloc(&arena, 32);
    assert(p1 != NULL);
    assert(arena.used == 32);

    bn_arena_reset(&arena);
    assert(arena.used == 0);

    p2 = bn_arena_alloc(&arena, 32);
    assert(p2 != NULL);
    assert(p2 == p1);
    assert(arena.used == 32);

    bn_arena_free(&arena);
    printf("  OK\n");
}

static void test_oom_returns_null_without_advancing_used(void) {
    bn_arena_t arena = bn_arena_create(96);
    void *p1;
    size_t used_before_oom;

    printf("Test: OOM returns NULL without advancing used...\n");
    assert(arena.base != NULL);

    p1 = bn_arena_alloc(&arena, 32);
    assert(p1 != NULL);
    assert(arena.used == 32);

    used_before_oom = arena.used;
    assert(bn_arena_alloc(&arena, 64) == NULL);
    assert(arena.used == used_before_oom);

    bn_arena_free(&arena);
    printf("  OK\n");
}

static void test_alignment_overflow_returns_null(void) {
    bn_arena_t arena;
    size_t used_before;

    printf("Test: alignment rounding overflow returns NULL...\n");

    /* Fabricate an arena with used near SIZE_MAX so that
       used + 63 would wrap size_t. */
    arena.base = (uint8_t *)(uintptr_t)0x1000; /* fake, never dereferenced */
    arena.size = SIZE_MAX;
    arena.used = SIZE_MAX - 10; /* + 63 wraps */

    used_before = arena.used;
    assert(bn_arena_alloc(&arena, 1) == NULL);
    assert(arena.used == used_before);
    printf("  OK\n");
}

static void test_offset_plus_size_overflow_returns_null(void) {
    bn_arena_t arena;
    size_t used_before;

    printf("Test: offset + size overflow returns NULL...\n");

    /* Fabricate an arena where alignment succeeds (used is already aligned)
       but offset + size wraps size_t. */
    arena.base = (uint8_t *)(uintptr_t)0x1000; /* fake, never dereferenced */
    arena.size = SIZE_MAX;
    arena.used = 64; /* already 64-aligned, so offset == 64 */

    used_before = arena.used;
    assert(bn_arena_alloc(&arena, SIZE_MAX) == NULL);
    assert(arena.used == used_before);
    printf("  OK\n");
}

int main(void) {
    printf("=== Arena Tests ===\n\n");
    test_alloc_null_arena_returns_null();
    test_allocations_are_64_byte_aligned();
    test_reset_allows_reuse_from_start();
    test_oom_returns_null_without_advancing_used();
    test_alignment_overflow_returns_null();
    test_offset_plus_size_overflow_returns_null();
    printf("\n=== All arena tests passed ===\n");
    return 0;
}
