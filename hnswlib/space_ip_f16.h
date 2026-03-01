#pragma once
#include "hnswlib.h"

namespace hnswlib {

static float
InnerProductF16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *) pVect1v;
    uint16_t *pVect2 = (uint16_t *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        res += half_to_float(pVect1[i]) * half_to_float(pVect2[i]);
    }
    return res;
}

static float
InnerProductDistanceF16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductF16(pVect1v, pVect2v, qty_ptr);
}

#if defined(USE_AVX512) && defined(__F16C__)

static float
InnerProductF16SIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *) pVect1v;
    uint16_t *pVect2 = (uint16_t *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4;

    const uint16_t *pEnd1 = pVect1 + (qty16 << 4);

    __m512 sum = _mm512_setzero_ps();

    while (pVect1 < pEnd1) {
        __m256i h1_lo = _mm256_loadu_si256((__m256i const*)pVect1);
        __m256i h2_lo = _mm256_loadu_si256((__m256i const*)pVect2);

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

        sum = _mm512_add_ps(sum, _mm512_mul_ps(v1, v2));

        pVect1 += 16;
        pVect2 += 16;
    }

    float PORTABLE_ALIGN64 TmpRes[16];
    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
            TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
            TmpRes[13] + TmpRes[14] + TmpRes[15];
    return res;
}

static float
InnerProductDistanceF16SIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductF16SIMD16ExtAVX512(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_AVX) && defined(__F16C__)

static float
InnerProductF16SIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *) pVect1v;
    uint16_t *pVect2 = (uint16_t *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const uint16_t *pEnd1 = pVect1 + (qty16 << 4);

    __m256 sum = _mm256_setzero_ps();

    while (pVect1 < pEnd1) {
        __m128i h1 = _mm_loadu_si128((__m128i const*)pVect1);
        __m128i h2 = _mm_loadu_si128((__m128i const*)pVect2);
        __m256 f1 = _mm256_cvtph_ps(h1);
        __m256 f2 = _mm256_cvtph_ps(h2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(f1, f2));

        h1 = _mm_loadu_si128((__m128i const*)(pVect1 + 8));
        h2 = _mm_loadu_si128((__m128i const*)(pVect2 + 8));
        f1 = _mm256_cvtph_ps(h1);
        f2 = _mm256_cvtph_ps(h2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(f1, f2));

        pVect1 += 16;
        pVect2 += 16;
    }

    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
}

static float
InnerProductDistanceF16SIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductF16SIMD16ExtAVX(pVect1v, pVect2v, qty_ptr);
}

static float
InnerProductF16SIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *) pVect1v;
    uint16_t *pVect2 = (uint16_t *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];

    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;

    const uint16_t *pEnd1 = pVect1 + 16 * qty16;
    const uint16_t *pEnd2 = pVect1 + 4 * qty4;

    __m256 sum256 = _mm256_setzero_ps();

    while (pVect1 < pEnd1) {
        __m128i h1 = _mm_loadu_si128((__m128i const*)pVect1);
        __m128i h2 = _mm_loadu_si128((__m128i const*)pVect2);
        __m256 f1 = _mm256_cvtph_ps(h1);
        __m256 f2 = _mm256_cvtph_ps(h2);
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(f1, f2));

        h1 = _mm_loadu_si128((__m128i const*)(pVect1 + 8));
        h2 = _mm_loadu_si128((__m128i const*)(pVect2 + 8));
        f1 = _mm256_cvtph_ps(h1);
        f2 = _mm256_cvtph_ps(h2);
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(f1, f2));

        pVect1 += 16;
        pVect2 += 16;
    }

    __m128 sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

    while (pVect1 < pEnd2) {
        __m128i h1 = _mm_loadl_epi64((__m128i const*)pVect1);
        __m128i h2 = _mm_loadl_epi64((__m128i const*)pVect2);
        __m128 f1 = _mm_cvtph_ps(h1);
        __m128 f2 = _mm_cvtph_ps(h2);
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(f1, f2));

        pVect1 += 4;
        pVect2 += 4;
    }

    _mm_store_ps(TmpRes, sum_prod);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

static float
InnerProductDistanceF16SIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductF16SIMD4ExtAVX(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_SSE) && defined(__F16C__)

static float
InnerProductF16SIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *) pVect1v;
    uint16_t *pVect2 = (uint16_t *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];

    size_t qty4 = qty / 4;

    const uint16_t *pEnd1 = pVect1 + 4 * qty4;

    __m128 sum_prod = _mm_setzero_ps();

    while (pVect1 < pEnd1) {
        __m128i h1 = _mm_loadl_epi64((__m128i const*)pVect1);
        __m128i h2 = _mm_loadl_epi64((__m128i const*)pVect2);
        __m128 f1 = _mm_cvtph_ps(h1);
        __m128 f2 = _mm_cvtph_ps(h2);
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(f1, f2));

        pVect1 += 4;
        pVect2 += 4;
    }

    _mm_store_ps(TmpRes, sum_prod);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

static float
InnerProductDistanceF16SIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductF16SIMD4ExtSSE(pVect1v, pVect2v, qty_ptr);
}

static float
InnerProductF16SIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *) pVect1v;
    uint16_t *pVect2 = (uint16_t *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];

    size_t qty16 = qty / 16;

    const uint16_t *pEnd1 = pVect1 + 16 * qty16;

    __m128 sum_prod = _mm_setzero_ps();

    while (pVect1 < pEnd1) {
        __m128i h1 = _mm_loadl_epi64((__m128i const*)pVect1);
        __m128i h2 = _mm_loadl_epi64((__m128i const*)pVect2);
        __m128 f1 = _mm_cvtph_ps(h1);
        __m128 f2 = _mm_cvtph_ps(h2);
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(f1, f2));

        h1 = _mm_loadl_epi64((__m128i const*)(pVect1 + 4));
        h2 = _mm_loadl_epi64((__m128i const*)(pVect2 + 4));
        f1 = _mm_cvtph_ps(h1);
        f2 = _mm_cvtph_ps(h2);
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(f1, f2));

        h1 = _mm_loadl_epi64((__m128i const*)(pVect1 + 8));
        h2 = _mm_loadl_epi64((__m128i const*)(pVect2 + 8));
        f1 = _mm_cvtph_ps(h1);
        f2 = _mm_cvtph_ps(h2);
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(f1, f2));

        h1 = _mm_loadl_epi64((__m128i const*)(pVect1 + 12));
        h2 = _mm_loadl_epi64((__m128i const*)(pVect2 + 12));
        f1 = _mm_cvtph_ps(h1);
        f2 = _mm_cvtph_ps(h2);
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(f1, f2));

        pVect1 += 16;
        pVect2 += 16;
    }

    _mm_store_ps(TmpRes, sum_prod);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

static float
InnerProductDistanceF16SIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductF16SIMD16ExtSSE(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_SSE) && defined(__F16C__)
static DISTFUNC<float> InnerProductF16SIMD16Ext = InnerProductF16SIMD16ExtSSE;
static DISTFUNC<float> InnerProductF16SIMD4Ext = InnerProductF16SIMD4ExtSSE;
static DISTFUNC<float> InnerProductDistanceF16SIMD16Ext = InnerProductDistanceF16SIMD16ExtSSE;
static DISTFUNC<float> InnerProductDistanceF16SIMD4Ext = InnerProductDistanceF16SIMD4ExtSSE;

static float
InnerProductDistanceF16SIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = InnerProductF16SIMD16Ext(pVect1v, pVect2v, &qty16);
    uint16_t *pVect1 = (uint16_t *) pVect1v + qty16;
    uint16_t *pVect2 = (uint16_t *) pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = InnerProductF16(pVect1, pVect2, &qty_left);
    return 1.0f - (res + res_tail);
}

static float
InnerProductDistanceF16SIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = InnerProductF16SIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    uint16_t *pVect1 = (uint16_t *) pVect1v + qty4;
    uint16_t *pVect2 = (uint16_t *) pVect2v + qty4;
    float res_tail = InnerProductF16(pVect1, pVect2, &qty_left);

    return 1.0f - (res + res_tail);
}
#endif

#if defined(USE_NEON)

static float
InnerProductF16SIMD16ExtNEON(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *) pVect1v;
    uint16_t *pVect2 = (uint16_t *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4;

    const uint16_t *pEnd1 = pVect1 + (qty16 << 4);

    float32x4_t sum = vdupq_n_f32(0);

    while (pVect1 < pEnd1) {
        float16x8_t h1 = vreinterpretq_f16_u16(vld1q_u16(pVect1));
        float16x8_t h2 = vreinterpretq_f16_u16(vld1q_u16(pVect2));

        sum = vfmaq_f32(sum, vcvt_f32_f16(vget_low_f16(h1)), vcvt_f32_f16(vget_low_f16(h2)));
        sum = vfmaq_f32(sum, vcvt_f32_f16(vget_high_f16(h1)), vcvt_f32_f16(vget_high_f16(h2)));

        h1 = vreinterpretq_f16_u16(vld1q_u16(pVect1 + 8));
        h2 = vreinterpretq_f16_u16(vld1q_u16(pVect2 + 8));

        sum = vfmaq_f32(sum, vcvt_f32_f16(vget_low_f16(h1)), vcvt_f32_f16(vget_low_f16(h2)));
        sum = vfmaq_f32(sum, vcvt_f32_f16(vget_high_f16(h1)), vcvt_f32_f16(vget_high_f16(h2)));

        pVect1 += 16;
        pVect2 += 16;
    }

    return vaddvq_f32(sum);
}

static float
InnerProductDistanceF16SIMD16ExtNEON(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductF16SIMD16ExtNEON(pVect1v, pVect2v, qty_ptr);
}

static float
InnerProductF16SIMD4ExtNEON(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *) pVect1v;
    uint16_t *pVect2 = (uint16_t *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2;

    const uint16_t *pEnd1 = pVect1 + (qty4 << 2);

    float32x4_t sum = vdupq_n_f32(0);

    while (pVect1 < pEnd1) {
        float32x4_t v1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(pVect1)));
        float32x4_t v2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(pVect2)));
        sum = vfmaq_f32(sum, v1, v2);

        pVect1 += 4;
        pVect2 += 4;
    }

    return vaddvq_f32(sum);
}

static float
InnerProductDistanceF16SIMD4ExtNEON(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductF16SIMD4ExtNEON(pVect1v, pVect2v, qty_ptr);
}

static float
InnerProductDistanceF16SIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = InnerProductF16SIMD16ExtNEON(pVect1v, pVect2v, &qty16);
    uint16_t *pVect1 = (uint16_t *) pVect1v + qty16;
    uint16_t *pVect2 = (uint16_t *) pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = InnerProductF16(pVect1, pVect2, &qty_left);
    return 1.0f - (res + res_tail);
}

static float
InnerProductDistanceF16SIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = InnerProductF16SIMD4ExtNEON(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    uint16_t *pVect1 = (uint16_t *) pVect1v + qty4;
    uint16_t *pVect2 = (uint16_t *) pVect2v + qty4;
    float res_tail = InnerProductF16(pVect1, pVect2, &qty_left);

    return 1.0f - (res + res_tail);
}

#endif

class InnerProductFloat16Space : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    InnerProductFloat16Space(size_t dim) {
        fstdistfunc_ = InnerProductDistanceF16;
#if defined(USE_SSE) && defined(__F16C__)
        if (F16CCapable()) {
    #if defined(USE_AVX512)
            if (AVX512Capable()) {
                InnerProductF16SIMD16Ext = InnerProductF16SIMD16ExtAVX512;
                InnerProductDistanceF16SIMD16Ext = InnerProductDistanceF16SIMD16ExtAVX512;
            } else if (AVXCapable()) {
                InnerProductF16SIMD16Ext = InnerProductF16SIMD16ExtAVX;
                InnerProductDistanceF16SIMD16Ext = InnerProductDistanceF16SIMD16ExtAVX;
            }
    #elif defined(USE_AVX)
            if (AVXCapable()) {
                InnerProductF16SIMD16Ext = InnerProductF16SIMD16ExtAVX;
                InnerProductDistanceF16SIMD16Ext = InnerProductDistanceF16SIMD16ExtAVX;
            }
    #endif
    #if defined(USE_AVX)
            if (AVXCapable()) {
                InnerProductF16SIMD4Ext = InnerProductF16SIMD4ExtAVX;
                InnerProductDistanceF16SIMD4Ext = InnerProductDistanceF16SIMD4ExtAVX;
            }
    #endif

            if (dim % 16 == 0)
                fstdistfunc_ = InnerProductDistanceF16SIMD16Ext;
            else if (dim % 4 == 0)
                fstdistfunc_ = InnerProductDistanceF16SIMD4Ext;
            else if (dim > 16)
                fstdistfunc_ = InnerProductDistanceF16SIMD16ExtResiduals;
            else if (dim > 4)
                fstdistfunc_ = InnerProductDistanceF16SIMD4ExtResiduals;
        }
#elif defined(USE_NEON)
        if (dim % 16 == 0)
            fstdistfunc_ = InnerProductDistanceF16SIMD16ExtNEON;
        else if (dim % 4 == 0)
            fstdistfunc_ = InnerProductDistanceF16SIMD4ExtNEON;
        else if (dim > 16)
            fstdistfunc_ = InnerProductDistanceF16SIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = InnerProductDistanceF16SIMD4ExtResiduals;
#endif
        dim_ = dim;
        data_size_ = dim * sizeof(uint16_t);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~InnerProductFloat16Space() {}
};

}  // namespace hnswlib
