#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__) && defined(__AVX512VL__)
#include <immintrin.h>

#define QK_I2_S 128

/* AVX-512 + VNNI kernel for I2_S ternary GEMV.
 *
 * Weight layout (matches bn_i2s_gemv_avx2): each 32-byte row-block packs four
 * ternary weights per byte, covering 128 activation positions. Byte p within a
 * 32-byte block holds weights for activation positions {p, p+32, p+64, p+96}:
 *   bits 6-7 -> pos p       (plane w0)
 *   bits 4-5 -> pos p+32    (plane w1)
 *   bits 2-3 -> pos p+64    (plane w2)
 *   bits 0-1 -> pos p+96    (plane w3)
 *
 * Main loop processes 2 blocks (64 weight bytes, 256 activations) per
 * iteration using 512-bit registers. Each 64-lane weight plane spans 2
 * consecutive blocks, so each 512-bit activation vector is assembled from two
 * 256-bit contiguous halves (activations for block 0, block 1) via
 * _mm512_inserti64x4. The maddubs -> add_epi16 -> madd_epi16 chain (and the
 * j==15 accu16-flush bookkeeping) used by the AVX2 kernel is replaced by a
 * single _mm512_dpbusd_epi32, which accumulates directly into int32 with no
 * overflow risk.
 */
void bn_i2s_gemv_avx512(const uint8_t *weights, const int8_t *acts,
                        float *out, int n_rows, int n_cols)
{
    const int nb = n_cols / QK_I2_S;
    const int nb_pairs = nb / 2;
    const int nb_odd = nb & 1;
    const int row_bytes = bn_i2s_row_stride(n_cols);

    const __m512i mask2_512 = _mm512_set1_epi8(0x03);
    const __m256i mask2_256 = _mm256_set1_epi8(0x03);

    for (int row = 0; row < n_rows; row++) {
        const uint8_t *x = weights + (size_t)row * row_bytes;
        const int8_t  *y = acts;

        __m512i accu = _mm512_setzero_si512();

        for (int i = 0; i < nb_pairs; i++) {
            __m512i wp = _mm512_loadu_si512((const __m512i *)x);

            __m512i w3 = _mm512_and_si512(wp, mask2_512);
            __m512i w2 = _mm512_and_si512(_mm512_srli_epi16(wp, 2), mask2_512);
            __m512i w1 = _mm512_and_si512(_mm512_srli_epi16(wp, 4), mask2_512);
            __m512i w0 = _mm512_and_si512(_mm512_srli_epi16(wp, 6), mask2_512);

            /* Each plane's 64 lanes: lo 32 -> block 0, hi 32 -> block 1.
             * Activation offsets below pair plane k with positions
             * {k*32 + p} in block 0 and {128 + k*32 + p} in block 1. */
            __m512i a0 = _mm512_inserti64x4(
                _mm512_castsi256_si512(
                    _mm256_loadu_si256((const __m256i *)(y + 0))),
                _mm256_loadu_si256((const __m256i *)(y + 128)), 1);
            __m512i a1 = _mm512_inserti64x4(
                _mm512_castsi256_si512(
                    _mm256_loadu_si256((const __m256i *)(y + 32))),
                _mm256_loadu_si256((const __m256i *)(y + 160)), 1);
            __m512i a2 = _mm512_inserti64x4(
                _mm512_castsi256_si512(
                    _mm256_loadu_si256((const __m256i *)(y + 64))),
                _mm256_loadu_si256((const __m256i *)(y + 192)), 1);
            __m512i a3 = _mm512_inserti64x4(
                _mm512_castsi256_si512(
                    _mm256_loadu_si256((const __m256i *)(y + 96))),
                _mm256_loadu_si256((const __m256i *)(y + 224)), 1);

            accu = _mm512_dpbusd_epi32(accu, w0, a0);
            accu = _mm512_dpbusd_epi32(accu, w1, a1);
            accu = _mm512_dpbusd_epi32(accu, w2, a2);
            accu = _mm512_dpbusd_epi32(accu, w3, a3);

            x += 64;
            y += 256;
        }

        int32_t result = _mm512_reduce_add_epi32(accu);

        /* One leftover block: 32 weight bytes, 128 activations. */
        if (nb_odd) {
            __m256i wp = _mm256_loadu_si256((const __m256i *)x);

            __m256i w3 = _mm256_and_si256(wp, mask2_256);
            __m256i w2 = _mm256_and_si256(_mm256_srli_epi16(wp, 2), mask2_256);
            __m256i w1 = _mm256_and_si256(_mm256_srli_epi16(wp, 4), mask2_256);
            __m256i w0 = _mm256_and_si256(_mm256_srli_epi16(wp, 6), mask2_256);

            __m256i a0 = _mm256_loadu_si256((const __m256i *)(y +  0));
            __m256i a1 = _mm256_loadu_si256((const __m256i *)(y + 32));
            __m256i a2 = _mm256_loadu_si256((const __m256i *)(y + 64));
            __m256i a3 = _mm256_loadu_si256((const __m256i *)(y + 96));

            __m256i a256 = _mm256_setzero_si256();
            a256 = _mm256_dpbusd_epi32(a256, w0, a0);
            a256 = _mm256_dpbusd_epi32(a256, w1, a1);
            a256 = _mm256_dpbusd_epi32(a256, w2, a2);
            a256 = _mm256_dpbusd_epi32(a256, w3, a3);

            __m128i sum128 = _mm_add_epi32(
                _mm256_castsi256_si128(a256),
                _mm256_extractf128_si256(a256, 1));
            const __m128i hi64  = _mm_unpackhi_epi64(sum128, sum128);
            const __m128i sum64 = _mm_add_epi32(hi64, sum128);
            const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
            result += _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));

            x += 32;
            y += 128;
        }

        /* Scalar tail for n_cols % QK_I2_S. */
        int tail = n_cols - nb * QK_I2_S;
        if (tail > 0) {
            int cols0 = tail >= 32  ? 32 : tail;
            int cols1 = tail >= 64  ? 32 : (tail > 32  ? tail - 32  : 0);
            int cols2 = tail >= 96  ? 32 : (tail > 64  ? tail - 64  : 0);
            int cols3 = tail >= 128 ? 32 : (tail > 96  ? tail - 96  : 0);

            for (int j = 0; j < 32; j++) {
                uint8_t b = x[j];
                int8_t w_0 = (int8_t)((b >> 6) & 0x03);
                int8_t w_1 = (int8_t)((b >> 4) & 0x03);
                int8_t w_2 = (int8_t)((b >> 2) & 0x03);
                int8_t w_3 = (int8_t)((b >> 0) & 0x03);

                if (j < cols0) result += w_0 * (int32_t)y[0*32 + j];
                if (j < cols1) result += w_1 * (int32_t)y[1*32 + j];
                if (j < cols2) result += w_2 * (int32_t)y[2*32 + j];
                if (j < cols3) result += w_3 * (int32_t)y[3*32 + j];
            }
        }

        out[row] = (float)result;
    }
}

#endif /* __AVX512F__ && __AVX512VNNI__ && __AVX512VL__ */
