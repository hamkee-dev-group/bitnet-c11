#define _POSIX_C_SOURCE 200809L

#include "bitnet.h"
#include <math.h>

void bn_quantize_acts(const float *src, int8_t *dst, int n,
                      float *scale_out, int32_t *sum_out)
{
    double mx = 1e-5;
    for (int i = 0; i < n; i++) {
        double a = fabs((double)src[i]);
        if (a > mx) mx = a;
    }

    float s = 127.0f / (float)mx;
    *scale_out = s;

    int32_t sum = 0;
    for (int i = 0; i < n; i++) {
        int v = (int)lrintf(src[i] * s);
        if (v >  127) v =  127;
        if (v < -128) v = -128;
        dst[i] = (int8_t)v;
        sum += v;
    }
    *sum_out = sum;
}

static const float map2bit[4] = { -1.0f, 0.0f, 1.0f, 0.0f };

void bn_dequant_i2s(const uint8_t *packed, float *dst, int n, float scale) {
    int done = 0;
    const uint8_t *p = packed;
    while (done < n) {
        int blk = (n - done >= 128) ? 128 : (n - done);
        int cols0 = blk >= 32  ? 32 : blk;
        int cols1 = blk >= 64  ? 32 : (blk > 32  ? blk - 32  : 0);
        int cols2 = blk >= 96  ? 32 : (blk > 64  ? blk - 64  : 0);
        int cols3 = blk >= 128 ? 32 : (blk > 96  ? blk - 96  : 0);

        for (int gp = 0; gp < 32 && gp < cols0; gp++) {
            uint8_t b = p[gp];
            uint8_t c0 = (b >> 6) & 0x03;
            uint8_t c1 = (b >> 4) & 0x03;
            uint8_t c2 = (b >> 2) & 0x03;
            uint8_t c3 = (b >> 0) & 0x03;

            if (gp < cols0) dst[done + 0*32 + gp] = scale * map2bit[c0];
            if (gp < cols1) dst[done + 1*32 + gp] = scale * map2bit[c1];
            if (gp < cols2) dst[done + 2*32 + gp] = scale * map2bit[c2];
            if (gp < cols3) dst[done + 3*32 + gp] = scale * map2bit[c3];
        }

        p += 32;
        done += blk;
    }
}
