#pragma once
#include "hnswlib.h"

namespace hnswlib {

static float InnerProductBF16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *)pVect1v;
    uint16_t *pVect2 = (uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        res += bfloat16_to_float(pVect1[i]) * bfloat16_to_float(pVect2[i]);
    }
    return res;
}

static float InnerProductDistanceBF16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductBF16(pVect1v, pVect2v, qty_ptr);
}

#if defined(USE_AVX512)

static float InnerProductBF16SIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *)pVect1v;
    uint16_t *pVect2 = (uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);
    size_t qty16 = qty >> 4;

    const uint16_t *pEnd1 = pVect1 + (qty16 << 4);

    __m512 sum = _mm512_setzero_ps();

    while (pVect1 < pEnd1) {
        __m256i h1 = _mm256_loadu_si256((__m256i const *)pVect1);
        __m256i h2 = _mm256_loadu_si256((__m256i const *)pVect2);

        __m512 v1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h1), 16));
        __m512 v2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h2), 16));

        sum = _mm512_add_ps(sum, _mm512_mul_ps(v1, v2));

        pVect1 += 16;
        pVect2 += 16;
    }

    float PORTABLE_ALIGN64 TmpRes[16];
    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7] +
                TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] + TmpRes[13] + TmpRes[14] + TmpRes[15];
    return res;
}

static float InnerProductDistanceBF16SIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductBF16SIMD16ExtAVX512(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_AVX)

static float InnerProductBF16SIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
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
        __m256 f1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h1), 16));
        __m256 f2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h2), 16));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(f1, f2));

        h1 = _mm_loadu_si128((__m128i const *)(pVect1 + 8));
        h2 = _mm_loadu_si128((__m128i const *)(pVect2 + 8));
        f1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h1), 16));
        f2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h2), 16));
        sum = _mm256_add_ps(sum, _mm256_mul_ps(f1, f2));

        pVect1 += 16;
        pVect2 += 16;
    }

    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
}

static float InnerProductDistanceBF16SIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductBF16SIMD16ExtAVX(pVect1v, pVect2v, qty_ptr);
}

static float InnerProductBF16SIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *)pVect1v;
    uint16_t *pVect2 = (uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];

    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;

    const uint16_t *pEnd1 = pVect1 + 16 * qty16;
    const uint16_t *pEnd2 = pVect1 + 4 * qty4;

    __m256 sum256 = _mm256_setzero_ps();

    while (pVect1 < pEnd1) {
        __m128i h1 = _mm_loadu_si128((__m128i const *)pVect1);
        __m128i h2 = _mm_loadu_si128((__m128i const *)pVect2);
        __m256 f1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h1), 16));
        __m256 f2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h2), 16));
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(f1, f2));

        h1 = _mm_loadu_si128((__m128i const *)(pVect1 + 8));
        h2 = _mm_loadu_si128((__m128i const *)(pVect2 + 8));
        f1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h1), 16));
        f2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h2), 16));
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(f1, f2));

        pVect1 += 16;
        pVect2 += 16;
    }

    __m128 sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
    __m128i zero = _mm_setzero_si128();

    while (pVect1 < pEnd2) {
        __m128i h1 = _mm_loadl_epi64((__m128i const *)pVect1);
        __m128i h2 = _mm_loadl_epi64((__m128i const *)pVect2);
        __m128 f1 = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, h1));
        __m128 f2 = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, h2));
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(f1, f2));

        pVect1 += 4;
        pVect2 += 4;
    }

    _mm_store_ps(TmpRes, sum_prod);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

static float InnerProductDistanceBF16SIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductBF16SIMD4ExtAVX(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_SSE)

static float InnerProductBF16SIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *)pVect1v;
    uint16_t *pVect2 = (uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];

    size_t qty4 = qty / 4;

    const uint16_t *pEnd1 = pVect1 + 4 * qty4;

    __m128 sum_prod = _mm_setzero_ps();
    __m128i zero = _mm_setzero_si128();

    while (pVect1 < pEnd1) {
        __m128i h1 = _mm_loadl_epi64((__m128i const *)pVect1);
        __m128i h2 = _mm_loadl_epi64((__m128i const *)pVect2);
        __m128 f1 = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, h1));
        __m128 f2 = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, h2));
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(f1, f2));

        pVect1 += 4;
        pVect2 += 4;
    }

    _mm_store_ps(TmpRes, sum_prod);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

static float InnerProductDistanceBF16SIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductBF16SIMD4ExtSSE(pVect1v, pVect2v, qty_ptr);
}

static float InnerProductBF16SIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *)pVect1v;
    uint16_t *pVect2 = (uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];

    size_t qty16 = qty / 16;

    const uint16_t *pEnd1 = pVect1 + 16 * qty16;

    __m128 sum_prod = _mm_setzero_ps();
    __m128i zero = _mm_setzero_si128();

    while (pVect1 < pEnd1) {
        __m128i h1 = _mm_loadl_epi64((__m128i const *)pVect1);
        __m128i h2 = _mm_loadl_epi64((__m128i const *)pVect2);
        __m128 f1 = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, h1));
        __m128 f2 = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, h2));
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(f1, f2));

        h1 = _mm_loadl_epi64((__m128i const *)(pVect1 + 4));
        h2 = _mm_loadl_epi64((__m128i const *)(pVect2 + 4));
        f1 = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, h1));
        f2 = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, h2));
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(f1, f2));

        h1 = _mm_loadl_epi64((__m128i const *)(pVect1 + 8));
        h2 = _mm_loadl_epi64((__m128i const *)(pVect2 + 8));
        f1 = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, h1));
        f2 = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, h2));
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(f1, f2));

        h1 = _mm_loadl_epi64((__m128i const *)(pVect1 + 12));
        h2 = _mm_loadl_epi64((__m128i const *)(pVect2 + 12));
        f1 = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, h1));
        f2 = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, h2));
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(f1, f2));

        pVect1 += 16;
        pVect2 += 16;
    }

    _mm_store_ps(TmpRes, sum_prod);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

static float InnerProductDistanceBF16SIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductBF16SIMD16ExtSSE(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_SSE)
static DISTFUNC<float> InnerProductBF16SIMD16Ext = InnerProductBF16SIMD16ExtSSE;
static DISTFUNC<float> InnerProductBF16SIMD4Ext = InnerProductBF16SIMD4ExtSSE;
static DISTFUNC<float> InnerProductDistanceBF16SIMD16Ext = InnerProductDistanceBF16SIMD16ExtSSE;
static DISTFUNC<float> InnerProductDistanceBF16SIMD4Ext = InnerProductDistanceBF16SIMD4ExtSSE;

static float InnerProductDistanceBF16SIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *)qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = InnerProductBF16SIMD16Ext(pVect1v, pVect2v, &qty16);
    uint16_t *pVect1 = (uint16_t *)pVect1v + qty16;
    uint16_t *pVect2 = (uint16_t *)pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = InnerProductBF16(pVect1, pVect2, &qty_left);
    return 1.0f - (res + res_tail);
}

static float InnerProductDistanceBF16SIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *)qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = InnerProductBF16SIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    uint16_t *pVect1 = (uint16_t *)pVect1v + qty4;
    uint16_t *pVect2 = (uint16_t *)pVect2v + qty4;
    float res_tail = InnerProductBF16(pVect1, pVect2, &qty_left);

    return 1.0f - (res + res_tail);
}
#endif

#if defined(USE_NEON)

static float InnerProductBF16SIMD16ExtNEON(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *)pVect1v;
    uint16_t *pVect2 = (uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);
    size_t qty16 = qty >> 4;

    const uint16_t *pEnd1 = pVect1 + (qty16 << 4);

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    // native bf16 dot product: each BFDOT consumes 8 bf16 lanes and
    // accumulates pairwise products into 4 f32 lanes, no explicit widening
    float32x4_t sum0 = vdupq_n_f32(0);
    float32x4_t sum1 = vdupq_n_f32(0);

    while (pVect1 < pEnd1) {
        sum0 = vbfdotq_f32(sum0, vreinterpretq_bf16_u16(vld1q_u16(pVect1)), vreinterpretq_bf16_u16(vld1q_u16(pVect2)));
        sum1 = vbfdotq_f32(sum1, vreinterpretq_bf16_u16(vld1q_u16(pVect1 + 8)),
                           vreinterpretq_bf16_u16(vld1q_u16(pVect2 + 8)));

        pVect1 += 16;
        pVect2 += 16;
    }

    return vaddvq_f32(vaddq_f32(sum0, sum1));
#else
    float32x4_t sum = vdupq_n_f32(0);

    while (pVect1 < pEnd1) {
        uint16x8_t h1 = vld1q_u16(pVect1);
        uint16x8_t h2 = vld1q_u16(pVect2);

        float32x4_t v1_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h1), 16));
        float32x4_t v2_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h2), 16));
        sum = vfmaq_f32(sum, v1_lo, v2_lo);

        float32x4_t v1_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h1), 16));
        float32x4_t v2_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h2), 16));
        sum = vfmaq_f32(sum, v1_hi, v2_hi);

        h1 = vld1q_u16(pVect1 + 8);
        h2 = vld1q_u16(pVect2 + 8);

        v1_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h1), 16));
        v2_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h2), 16));
        sum = vfmaq_f32(sum, v1_lo, v2_lo);

        v1_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h1), 16));
        v2_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h2), 16));
        sum = vfmaq_f32(sum, v1_hi, v2_hi);

        pVect1 += 16;
        pVect2 += 16;
    }

    return vaddvq_f32(sum);
#endif
}

static float InnerProductDistanceBF16SIMD16ExtNEON(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductBF16SIMD16ExtNEON(pVect1v, pVect2v, qty_ptr);
}

static float InnerProductBF16SIMD4ExtNEON(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *)pVect1v;
    uint16_t *pVect2 = (uint16_t *)pVect2v;
    size_t qty = *((size_t *)qty_ptr);
    size_t qty4 = qty >> 2;

    const uint16_t *pEnd1 = pVect1 + (qty4 << 2);

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    float32x2_t sum = vdup_n_f32(0);

    while (pVect1 < pEnd1) {
        sum = vbfdot_f32(sum, vreinterpret_bf16_u16(vld1_u16(pVect1)), vreinterpret_bf16_u16(vld1_u16(pVect2)));

        pVect1 += 4;
        pVect2 += 4;
    }

    return vaddv_f32(sum);
#else
    float32x4_t sum = vdupq_n_f32(0);

    while (pVect1 < pEnd1) {
        float32x4_t v1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(pVect1), 16));
        float32x4_t v2 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(pVect2), 16));
        sum = vfmaq_f32(sum, v1, v2);

        pVect1 += 4;
        pVect2 += 4;
    }

    return vaddvq_f32(sum);
#endif
}

static float InnerProductDistanceBF16SIMD4ExtNEON(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductBF16SIMD4ExtNEON(pVect1v, pVect2v, qty_ptr);
}

static float InnerProductDistanceBF16SIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *)qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = InnerProductBF16SIMD16ExtNEON(pVect1v, pVect2v, &qty16);
    uint16_t *pVect1 = (uint16_t *)pVect1v + qty16;
    uint16_t *pVect2 = (uint16_t *)pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = InnerProductBF16(pVect1, pVect2, &qty_left);
    return 1.0f - (res + res_tail);
}

static float InnerProductDistanceBF16SIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *)qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = InnerProductBF16SIMD4ExtNEON(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    uint16_t *pVect1 = (uint16_t *)pVect1v + qty4;
    uint16_t *pVect2 = (uint16_t *)pVect2v + qty4;
    float res_tail = InnerProductBF16(pVect1, pVect2, &qty_left);

    return 1.0f - (res + res_tail);
}

#endif

class InnerProductBFloat16Space : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    InnerProductBFloat16Space(size_t dim) {
        fstdistfunc_ = InnerProductDistanceBF16;
#if defined(USE_SSE)
#if defined(USE_AVX512)
        if (AVX512Capable()) {
            InnerProductBF16SIMD16Ext = InnerProductBF16SIMD16ExtAVX512;
            InnerProductDistanceBF16SIMD16Ext = InnerProductDistanceBF16SIMD16ExtAVX512;
        } else if (AVXCapable()) {
            InnerProductBF16SIMD16Ext = InnerProductBF16SIMD16ExtAVX;
            InnerProductDistanceBF16SIMD16Ext = InnerProductDistanceBF16SIMD16ExtAVX;
        }
#elif defined(USE_AVX)
        if (AVXCapable()) {
            InnerProductBF16SIMD16Ext = InnerProductBF16SIMD16ExtAVX;
            InnerProductDistanceBF16SIMD16Ext = InnerProductDistanceBF16SIMD16ExtAVX;
        }
#endif
#if defined(USE_AVX)
        if (AVXCapable()) {
            InnerProductBF16SIMD4Ext = InnerProductBF16SIMD4ExtAVX;
            InnerProductDistanceBF16SIMD4Ext = InnerProductDistanceBF16SIMD4ExtAVX;
        }
#endif

        if (dim % 16 == 0)
            fstdistfunc_ = InnerProductDistanceBF16SIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = InnerProductDistanceBF16SIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = InnerProductDistanceBF16SIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = InnerProductDistanceBF16SIMD4ExtResiduals;
#elif defined(USE_NEON)
        if (dim % 16 == 0)
            fstdistfunc_ = InnerProductDistanceBF16SIMD16ExtNEON;
        else if (dim % 4 == 0)
            fstdistfunc_ = InnerProductDistanceBF16SIMD4ExtNEON;
        else if (dim > 16)
            fstdistfunc_ = InnerProductDistanceBF16SIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = InnerProductDistanceBF16SIMD4ExtResiduals;
#endif
        dim_ = dim;
        data_size_ = dim * sizeof(uint16_t);
    }

    size_t get_data_size() { return data_size_; }

    DISTFUNC<float> get_dist_func() { return fstdistfunc_; }

    void *get_dist_func_param() { return &dim_; }

    ~InnerProductBFloat16Space() {}
};

}  // namespace hnswlib
