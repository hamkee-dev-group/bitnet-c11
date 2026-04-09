#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"

#if defined(__AVX2__)
#include <immintrin.h>

#define QK_I2_S 128

static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(a),
        _mm256_extractf128_si256(a, 1));
    const __m128i hi64  = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

void bn_i2s_gemv_avx2(const uint8_t *weights, const int8_t *acts,
                      float *out, int n_rows, int n_cols)
{
    const int nb = n_cols / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int row_bytes = n_cols / 4;

    __m256i mask  = _mm256_set1_epi8(0x03);
    __m256i one16 = _mm256_set1_epi16(1);

    for (int row = 0; row < n_rows; row++) {
        const uint8_t *x = weights + row * row_bytes;
        __m256i accu = _mm256_setzero_si256();

        for (int i = 0; i < group32_num; i++) {
            const uint8_t *px = x + i * 1024;
            const int8_t  *py = acts + i * 4096;
            __m256i accu32 = _mm256_setzero_si256();

            for (int j = 0; j < 32; j++) {
                __m256i xq8_3 = _mm256_loadu_si256((const __m256i *)px);
                __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
                __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
                __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

                xq8_3 = _mm256_and_si256(xq8_3, mask);
                xq8_2 = _mm256_and_si256(xq8_2, mask);
                xq8_1 = _mm256_and_si256(xq8_1, mask);
                xq8_0 = _mm256_and_si256(xq8_0, mask);

                __m256i yq8_0 = _mm256_loadu_si256((const __m256i *)(py));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i *)(py + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i *)(py + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i *)(py + 96));

                xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
                xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
                xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
                xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

                accu32 = _mm256_add_epi16(accu32,
                    _mm256_add_epi16(xq8_0, xq8_1));
                accu32 = _mm256_add_epi16(accu32,
                    _mm256_add_epi16(xq8_2, xq8_3));

                px += 32;
                py += 128;
            }
            accu = _mm256_add_epi32(
                _mm256_madd_epi16(accu32, one16), accu);
        }

        if (la_num > 0) {
            __m256i accula = _mm256_setzero_si256();
            const uint8_t *px = x + group32_num * 1024;
            const int8_t  *py = acts + group32_num * 4096;

            for (int j = 0; j < la_num; j++) {
                __m256i xq8_3 = _mm256_loadu_si256((const __m256i *)px);
                __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
                __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
                __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

                xq8_3 = _mm256_and_si256(xq8_3, mask);
                xq8_2 = _mm256_and_si256(xq8_2, mask);
                xq8_1 = _mm256_and_si256(xq8_1, mask);
                xq8_0 = _mm256_and_si256(xq8_0, mask);

                __m256i yq8_0 = _mm256_loadu_si256((const __m256i *)(py));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i *)(py + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i *)(py + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i *)(py + 96));

                xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
                xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
                xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
                xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

                accula = _mm256_add_epi16(accula,
                    _mm256_add_epi16(xq8_0, xq8_1));
                accula = _mm256_add_epi16(accula,
                    _mm256_add_epi16(xq8_2, xq8_3));

                px += 32;
                py += 128;
            }
            accu = _mm256_add_epi32(accu,
                _mm256_madd_epi16(accula, one16));
        }

        out[row] = (float)hsum_i32_8(accu);
    }
}

#endif
