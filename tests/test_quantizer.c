#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static void test_balanced_range(void) {
    printf("Test: balanced range quantization...\n");

    const float input[] = { -1.0f, -0.5f, 0.0f, 0.5f, 1.0f };
    const int8_t expected[] = { -127, -64, 0, 64, 127 };
    int8_t output[5];
    float scale = 0.0f;
    int32_t sum = 0;

    bn_quantize_acts(input, output, 5, &scale, &sum);

    assert(fabsf(scale - 127.0f) < 1e-6f);
    for (int i = 0; i < 5; i++) {
        assert(output[i] == expected[i]);
    }
    assert(sum == 0);

    printf("  OK\n");
}

static void test_zero_input_uses_min_scale_floor(void) {
    printf("Test: zero inputs keep floor scale and zero outputs...\n");

    const float input[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    int8_t output[4];
    float scale = 0.0f;
    int32_t sum = -1;

    bn_quantize_acts(input, output, 4, &scale, &sum);

    assert(fabsf(scale - 12700000.0f) < 1.0f);
    for (int i = 0; i < 4; i++) {
        assert(output[i] == 0);
    }
    assert(sum == 0);

    printf("  OK\n");
}

static void test_small_values_use_min_scale_floor(void) {
    printf("Test: very small values quantize from minimum scale floor...\n");

    const float input[] = { 1.0e-6f, -1.0e-6f, 3.0e-6f };
    const int8_t expected[] = { 13, -13, 38 };
    int8_t output[3];
    float scale = 0.0f;
    int32_t sum = 0;

    bn_quantize_acts(input, output, 3, &scale, &sum);

    assert(fabsf(scale - 12700000.0f) < 1.0f);
    for (int i = 0; i < 3; i++) {
        assert(output[i] == expected[i]);
    }
    assert(sum == 38);

    printf("  OK\n");
}

static void test_empty_input(void) {
    printf("Test: empty input leaves outputs untouched and sum zero...\n");

    int8_t output[] = { 11, -7 };
    float scale = 0.0f;
    int32_t sum = -1;

    bn_quantize_acts(NULL, output, 0, &scale, &sum);

    assert(fabsf(scale - 12700000.0f) < 1.0f);
    assert(sum == 0);
    assert(output[0] == 11);
    assert(output[1] == -7);

    printf("  OK\n");
}

int main(void) {
    printf("=== Quantizer Tests ===\n\n");
    test_balanced_range();
    test_zero_input_uses_min_scale_floor();
    test_small_values_use_min_scale_floor();
    test_empty_input();
    printf("\n=== All quantizer tests passed ===\n");
    return 0;
}
