#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"

void bn_i2s_gemv_scalar(const uint8_t *weights, const int8_t *acts,
                        float *out, int n_rows, int n_cols)
{
    int row_bytes = n_cols / 4;

    for (int row = 0; row < n_rows; row++) {
        const uint8_t *w = weights + row * row_bytes;
        int32_t acc = 0;
        int col = 0;

        while (col + 128 <= n_cols) {
            for (int j = 0; j < 32; j++) {
                uint8_t b = w[j];
                int8_t w0 = (int8_t)((b >> 6) & 0x03);
                int8_t w1 = (int8_t)((b >> 4) & 0x03);
                int8_t w2 = (int8_t)((b >> 2) & 0x03);
                int8_t w3 = (int8_t)((b >> 0) & 0x03);

                acc += w0 * (int32_t)acts[col + 0*32 + j];
                acc += w1 * (int32_t)acts[col + 1*32 + j];
                acc += w2 * (int32_t)acts[col + 2*32 + j];
                acc += w3 * (int32_t)acts[col + 3*32 + j];
            }
            w   += 32;
            col += 128;
        }

        if (col < n_cols) {
            int remain = n_cols - col;
            int cols0 = remain >= 32  ? 32 : remain;
            int cols1 = remain >= 64  ? 32 : (remain > 32  ? remain - 32  : 0);
            int cols2 = remain >= 96  ? 32 : (remain > 64  ? remain - 64  : 0);
            int cols3 = remain >= 128 ? 32 : (remain > 96  ? remain - 96  : 0);

            for (int j = 0; j < 32; j++) {
                uint8_t b = w[j];
                int8_t w0 = (int8_t)((b >> 6) & 0x03);
                int8_t w1 = (int8_t)((b >> 4) & 0x03);
                int8_t w2 = (int8_t)((b >> 2) & 0x03);
                int8_t w3 = (int8_t)((b >> 0) & 0x03);

                if (j < cols0) acc += w0 * (int32_t)acts[col + 0*32 + j];
                if (j < cols1) acc += w1 * (int32_t)acts[col + 1*32 + j];
                if (j < cols2) acc += w2 * (int32_t)acts[col + 2*32 + j];
                if (j < cols3) acc += w3 * (int32_t)acts[col + 3*32 + j];
            }
        }

        out[row] = (float)acc;
    }
}
