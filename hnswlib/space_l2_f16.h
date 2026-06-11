#pragma once
#include "hnswlib.h"

namespace hnswlib {

static inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x03FF;

    if (exponent == 0) {
        if (mantissa == 0) {
            uint32_t result = sign;
            float f;
            memcpy(&f, &result, sizeof(f));
            return f;
        }
        exponent = 1;
        while (!(mantissa & 0x0400)) {
            mantissa <<= 1;
            exponent--;
        }
        mantissa &= 0x03FF;
        exponent = exponent + (127 - 15);
        uint32_t result = sign | (exponent << 23) | (mantissa << 13);
        float f;
        memcpy(&f, &result, sizeof(f));
        return f;
    } else if (exponent == 31) {
        uint32_t result = sign | 0x7F800000 | (mantissa << 13);
        float f;
        memcpy(&f, &result, sizeof(f));
        return f;
    }

    exponent = exponent + (127 - 15);
    uint32_t result = sign | (exponent << 23) | (mantissa << 13);
    float f;
    memcpy(&f, &result, sizeof(f));
    return f;
}

static inline uint16_t float_to_half(float value) {
    uint32_t f;
    memcpy(&f, &value, sizeof(f));

    uint32_t sign = (f >> 16) & 0x8000;
    int32_t exponent = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = f & 0x007FFFFF;

    if (exponent <= 0) {
        if (exponent < -10) {
            return (uint16_t)sign;
        }
        mantissa = (mantissa | 0x00800000) >> (1 - exponent);
        if (mantissa & 0x00001000)
            mantissa += 0x00002000;
        return (uint16_t)(sign | (mantissa >> 13));
    } else if (exponent == 0xFF - (127 - 15)) {
        if (mantissa == 0) {
            return (uint16_t)(sign | 0x7C00);
        } else {
            mantissa >>= 13;
            return (uint16_t)(sign | 0x7C00 | mantissa | (mantissa == 0));
        }
    }

    if (mantissa & 0x00001000) {
        mantissa += 0x00002000;
        if (mantissa & 0x00800000) {
            mantissa = 0;
            exponent++;
        }
    }

    if (exponent > 30) {
        return (uint16_t)(sign | 0x7C00);
    }

    return (uint16_t)(sign | (exponent << 10) | (mantissa >> 13));
}

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static bool F16CCapable() {
    int cpuInfo[4];
    cpuid(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];
    if (nIds >= 1) {
        cpuid(cpuInfo, 1, 0);
        return (cpuInfo[2] & ((int)1 << 29)) != 0;
    }
    return false;
}
#endif

static float L2SqrF16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *)pVect1v;
    uint16_t *pVect2 = (uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float v1 = half_to_float(pVect1[i]);
        float v2 = half_to_float(pVect2[i]);
        float t = v1 - v2;
        res += t * t;
    }
    return res;
}

#if defined(USE_AVX512) && defined(__F16C__)

static float L2SqrF16SIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *)pVect1v;
    uint16_t *pVect2 = (uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);
    size_t qty16 = qty >> 4;

    const uint16_t *pEnd1 = pVect1 + (qty16 << 4);

    __m512 sum = _mm512_setzero_ps();

    while (pVect1 < pEnd1) {
        __m256i h1_lo = _mm256_loadu_si256((__m256i const *)pVect1);
        __m256i h2_lo = _mm256_loadu_si256((__m256i const *)pVect2);

        __m128i h1_lo_128 = _mm256_castsi256_si128(h1_lo);
        __m128i h1_hi_128 = _mm256_extracti128_si256(h1_lo, 1);
        __m128i h2_lo_128 = _mm256_castsi256_si128(h2_lo);
        __m128i h2_hi_128 = _mm256_extracti128_si256(h2_lo, 1);

        __m256 f1_lo = _mm256_cvtph_ps(h1_lo_128);
        __m256 f1_hi = _mm256_cvtph_ps(h1_hi_128);
        __m256 f2_lo = _mm256_cvtph_ps(h2_lo_128);
        __m256 f2_hi = _mm256_cvtph_ps(h2_hi_128);

        __m512 v1 = _mm512_insertf32x8(_mm512_castps256_ps512(f1_lo), f1_hi, 1);
        __m512 v2 = _mm512_insertf32x8(_mm512_castps256_ps512(f2_lo), f2_hi, 1);

        __m512 diff = _mm512_sub_ps(v1, v2);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));

        pVect1 += 16;
        pVect2 += 16;
    }

    float PORTABLE_ALIGN64 TmpRes[16];
    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7] +
                TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] + TmpRes[13] + TmpRes[14] + TmpRes[15];

    return res;
}

#endif

#if defined(USE_AVX) && defined(__F16C__)

static float L2SqrF16SIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *)pVect1v;
    uint16_t *pVect2 = (uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const uint16_t *pEnd1 = pVect1 + (qty16 << 4);

    __m256 sum = _mm256_setzero_ps();

    while (pVect1 < pEnd1) {
        __m128i h1 = _mm_loadu_si128((__m128i const *)pVect1);
        __m128i h2 = _mm_loadu_si128((__m128i const *)pVect2);
        __m256 f1 = _mm256_cvtph_ps(h1);
        __m256 f2 = _mm256_cvtph_ps(h2);
        __m256 diff = _mm256_sub_ps(f1, f2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        h1 = _mm_loadu_si128((__m128i const *)(pVect1 + 8));
        h2 = _mm_loadu_si128((__m128i const *)(pVect2 + 8));
        f1 = _mm256_cvtph_ps(h1);
        f2 = _mm256_cvtph_ps(h2);
        diff = _mm256_sub_ps(f1, f2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        pVect1 += 16;
        pVect2 += 16;
    }

    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
}

#endif

#if defined(USE_SSE) && defined(__F16C__)

static float L2SqrF16SIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *)pVect1v;
    uint16_t *pVect2 = (uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty4 = qty >> 2;

    const uint16_t *pEnd1 = pVect1 + (qty4 << 2);

    __m128 sum = _mm_setzero_ps();

    while (pVect1 < pEnd1) {
        __m128i h1 = _mm_loadl_epi64((__m128i const *)pVect1);
        __m128i h2 = _mm_loadl_epi64((__m128i const *)pVect2);
        __m128 f1 = _mm_cvtph_ps(h1);
        __m128 f2 = _mm_cvtph_ps(h2);
        __m128 diff = _mm_sub_ps(f1, f2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        pVect1 += 4;
        pVect2 += 4;
    }

    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

#endif

#if defined(USE_SSE) && defined(__F16C__)
static DISTFUNC<float> L2SqrF16SIMD16Ext = L2SqrF16SIMD4ExtSSE;

static float L2SqrF16SIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *)qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = L2SqrF16SIMD16Ext(pVect1v, pVect2v, &qty16);
    uint16_t *pVect1 = (uint16_t *)pVect1v + qty16;
    uint16_t *pVect2 = (uint16_t *)pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = L2SqrF16(pVect1, pVect2, &qty_left);
    return (res + res_tail);
}

static float L2SqrF16SIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *)qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = L2SqrF16SIMD4ExtSSE(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    uint16_t *pVect1 = (uint16_t *)pVect1v + qty4;
    uint16_t *pVect2 = (uint16_t *)pVect2v + qty4;
    float res_tail = L2SqrF16(pVect1, pVect2, &qty_left);

    return (res + res_tail);
}
#endif

#if defined(USE_NEON)

static float L2SqrF16SIMD16ExtNEON(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *)pVect1v;
    uint16_t *pVect2 = (uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);
    size_t qty16 = qty >> 4;

    const uint16_t *pEnd1 = pVect1 + (qty16 << 4);

#if defined(__ARM_FEATURE_FP16_FML) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    // subtract in f16 (at most 0.5 ulp rounding on the difference), then
    // square-accumulate with widening f16 multiply-add directly into f32
    float32x4_t sum0 = vdupq_n_f32(0);
    float32x4_t sum1 = vdupq_n_f32(0);

    while (pVect1 < pEnd1) {
        float16x8_t d = vsubq_f16(vreinterpretq_f16_u16(vld1q_u16(pVect1)), vreinterpretq_f16_u16(vld1q_u16(pVect2)));
        sum0 = vfmlalq_low_f16(sum0, d, d);
        sum1 = vfmlalq_high_f16(sum1, d, d);

        d = vsubq_f16(vreinterpretq_f16_u16(vld1q_u16(pVect1 + 8)), vreinterpretq_f16_u16(vld1q_u16(pVect2 + 8)));
        sum0 = vfmlalq_low_f16(sum0, d, d);
        sum1 = vfmlalq_high_f16(sum1, d, d);

        pVect1 += 16;
        pVect2 += 16;
    }

    return vaddvq_f32(vaddq_f32(sum0, sum1));
#else
    float32x4_t sum = vdupq_n_f32(0);

    while (pVect1 < pEnd1) {
        float16x8_t h1 = vreinterpretq_f16_u16(vld1q_u16(pVect1));
        float16x8_t h2 = vreinterpretq_f16_u16(vld1q_u16(pVect2));

        float32x4_t v1 = vcvt_f32_f16(vget_low_f16(h1));
        float32x4_t v2 = vcvt_f32_f16(vget_low_f16(h2));
        float32x4_t diff = vsubq_f32(v1, v2);
        sum = vfmaq_f32(sum, diff, diff);

        v1 = vcvt_f32_f16(vget_high_f16(h1));
        v2 = vcvt_f32_f16(vget_high_f16(h2));
        diff = vsubq_f32(v1, v2);
        sum = vfmaq_f32(sum, diff, diff);

        h1 = vreinterpretq_f16_u16(vld1q_u16(pVect1 + 8));
        h2 = vreinterpretq_f16_u16(vld1q_u16(pVect2 + 8));

        v1 = vcvt_f32_f16(vget_low_f16(h1));
        v2 = vcvt_f32_f16(vget_low_f16(h2));
        diff = vsubq_f32(v1, v2);
        sum = vfmaq_f32(sum, diff, diff);

        v1 = vcvt_f32_f16(vget_high_f16(h1));
        v2 = vcvt_f32_f16(vget_high_f16(h2));
        diff = vsubq_f32(v1, v2);
        sum = vfmaq_f32(sum, diff, diff);

        pVect1 += 16;
        pVect2 += 16;
    }

    return vaddvq_f32(sum);
#endif
}

static float L2SqrF16SIMD4ExtNEON(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *)pVect1v;
    uint16_t *pVect2 = (uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);
    size_t qty4 = qty >> 2;

    const uint16_t *pEnd1 = pVect1 + (qty4 << 2);

#if defined(__ARM_FEATURE_FP16_FML) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    float32x2_t sum = vdup_n_f32(0);

    while (pVect1 < pEnd1) {
        float16x4_t d = vsub_f16(vreinterpret_f16_u16(vld1_u16(pVect1)), vreinterpret_f16_u16(vld1_u16(pVect2)));
        sum = vfmlal_low_f16(sum, d, d);
        sum = vfmlal_high_f16(sum, d, d);

        pVect1 += 4;
        pVect2 += 4;
    }

    return vaddv_f32(sum);
#else
    float32x4_t sum = vdupq_n_f32(0);

    while (pVect1 < pEnd1) {
        float32x4_t v1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(pVect1)));
        float32x4_t v2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(pVect2)));
        float32x4_t diff = vsubq_f32(v1, v2);
        sum = vfmaq_f32(sum, diff, diff);

        pVect1 += 4;
        pVect2 += 4;
    }

    return vaddvq_f32(sum);
#endif
}

static float L2SqrF16SIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *)qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = L2SqrF16SIMD16ExtNEON(pVect1v, pVect2v, &qty16);
    uint16_t *pVect1 = (uint16_t *)pVect1v + qty16;
    uint16_t *pVect2 = (uint16_t *)pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = L2SqrF16(pVect1, pVect2, &qty_left);
    return (res + res_tail);
}

static float L2SqrF16SIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *)qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = L2SqrF16SIMD4ExtNEON(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    uint16_t *pVect1 = (uint16_t *)pVect1v + qty4;
    uint16_t *pVect2 = (uint16_t *)pVect2v + qty4;
    float res_tail = L2SqrF16(pVect1, pVect2, &qty_left);

    return (res + res_tail);
}

#endif

class L2Float16Space : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    L2Float16Space(size_t dim) {
        fstdistfunc_ = L2SqrF16;
#if defined(USE_SSE) && defined(__F16C__)
        if (F16CCapable()) {
#if defined(USE_AVX512)
            if (AVX512Capable())
                L2SqrF16SIMD16Ext = L2SqrF16SIMD16ExtAVX512;
            else if (AVXCapable())
                L2SqrF16SIMD16Ext = L2SqrF16SIMD16ExtAVX;
#elif defined(USE_AVX)
            if (AVXCapable())
                L2SqrF16SIMD16Ext = L2SqrF16SIMD16ExtAVX;
#endif

            if (dim % 16 == 0)
                fstdistfunc_ = L2SqrF16SIMD16Ext;
            else if (dim % 4 == 0)
                fstdistfunc_ = L2SqrF16SIMD4ExtSSE;
            else if (dim > 16)
                fstdistfunc_ = L2SqrF16SIMD16ExtResiduals;
            else if (dim > 4)
                fstdistfunc_ = L2SqrF16SIMD4ExtResiduals;
        }
#elif defined(USE_NEON)
        if (dim % 16 == 0)
            fstdistfunc_ = L2SqrF16SIMD16ExtNEON;
        else if (dim % 4 == 0)
            fstdistfunc_ = L2SqrF16SIMD4ExtNEON;
        else if (dim > 16)
            fstdistfunc_ = L2SqrF16SIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = L2SqrF16SIMD4ExtResiduals;
#endif
        dim_ = dim;
        data_size_ = dim * sizeof(uint16_t);
    }

    size_t get_data_size() { return data_size_; }

    DISTFUNC<float> get_dist_func() { return fstdistfunc_; }

    void *get_dist_func_param() { return &dim_; }

    ~L2Float16Space() {}
};

}  // namespace hnswlib
