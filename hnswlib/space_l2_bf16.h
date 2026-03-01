#pragma once
#include "hnswlib.h"

namespace hnswlib {

static inline float bfloat16_to_float(uint16_t h) {
    uint32_t i = (uint32_t)h << 16;
    float f;
    memcpy(&f, &i, sizeof(f));
    return f;
}

static inline uint16_t float_to_bfloat16(float value) {
    uint32_t f;
    memcpy(&f, &value, sizeof(f));
    if ((f & 0x7FFFFFFF) > 0x7F800000) {
        return (uint16_t)((f >> 16) | 1);
    }
    f += 0x7FFF + ((f >> 16) & 1);
    return (uint16_t)(f >> 16);
}

static float
L2SqrBF16(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *) pVect1v;
    uint16_t *pVect2 = (uint16_t *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float v1 = bfloat16_to_float(pVect1[i]);
        float v2 = bfloat16_to_float(pVect2[i]);
        float t = v1 - v2;
        res += t * t;
    }
    return res;
}

#if defined(USE_AVX512)

static float
L2SqrBF16SIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *) pVect1v;
    uint16_t *pVect2 = (uint16_t *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4;

    const uint16_t *pEnd1 = pVect1 + (qty16 << 4);

    __m512 sum = _mm512_setzero_ps();

    while (pVect1 < pEnd1) {
        __m256i h1 = _mm256_loadu_si256((__m256i const*)pVect1);
        __m256i h2 = _mm256_loadu_si256((__m256i const*)pVect2);

        __m512 v1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h1), 16));
        __m512 v2 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h2), 16));

        __m512 diff = _mm512_sub_ps(v1, v2);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));

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

#endif

#if defined(USE_AVX)

static float
L2SqrBF16SIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
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
        __m256 f1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h1), 16));
        __m256 f2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h2), 16));
        __m256 diff = _mm256_sub_ps(f1, f2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        h1 = _mm_loadu_si128((__m128i const*)(pVect1 + 8));
        h2 = _mm_loadu_si128((__m128i const*)(pVect2 + 8));
        f1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h1), 16));
        f2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h2), 16));
        diff = _mm256_sub_ps(f1, f2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        pVect1 += 16;
        pVect2 += 16;
    }

    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
}

#endif

#if defined(USE_SSE)

static float
L2SqrBF16SIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *) pVect1v;
    uint16_t *pVect2 = (uint16_t *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty4 = qty >> 2;

    const uint16_t *pEnd1 = pVect1 + (qty4 << 2);

    __m128 sum = _mm_setzero_ps();
    __m128i zero = _mm_setzero_si128();

    while (pVect1 < pEnd1) {
        __m128i h1 = _mm_loadl_epi64((__m128i const*)pVect1);
        __m128i h2 = _mm_loadl_epi64((__m128i const*)pVect2);
        __m128 f1 = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, h1));
        __m128 f2 = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, h2));
        __m128 diff = _mm_sub_ps(f1, f2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        pVect1 += 4;
        pVect2 += 4;
    }

    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

#endif

#if defined(USE_SSE)
static DISTFUNC<float> L2SqrBF16SIMD16Ext = L2SqrBF16SIMD4ExtSSE;

static float
L2SqrBF16SIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = L2SqrBF16SIMD16Ext(pVect1v, pVect2v, &qty16);
    uint16_t *pVect1 = (uint16_t *) pVect1v + qty16;
    uint16_t *pVect2 = (uint16_t *) pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = L2SqrBF16(pVect1, pVect2, &qty_left);
    return (res + res_tail);
}

static float
L2SqrBF16SIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = L2SqrBF16SIMD4ExtSSE(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    uint16_t *pVect1 = (uint16_t *) pVect1v + qty4;
    uint16_t *pVect2 = (uint16_t *) pVect2v + qty4;
    float res_tail = L2SqrBF16(pVect1, pVect2, &qty_left);

    return (res + res_tail);
}
#endif

#if defined(USE_NEON)

static float
L2SqrBF16SIMD16ExtNEON(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *) pVect1v;
    uint16_t *pVect2 = (uint16_t *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4;

    const uint16_t *pEnd1 = pVect1 + (qty16 << 4);

    float32x4_t sum = vdupq_n_f32(0);

    while (pVect1 < pEnd1) {
        uint16x8_t h1 = vld1q_u16(pVect1);
        uint16x8_t h2 = vld1q_u16(pVect2);

        float32x4_t v1 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h1), 16));
        float32x4_t v2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h2), 16));
        float32x4_t diff = vsubq_f32(v1, v2);
        sum = vfmaq_f32(sum, diff, diff);

        v1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h1), 16));
        v2 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h2), 16));
        diff = vsubq_f32(v1, v2);
        sum = vfmaq_f32(sum, diff, diff);

        h1 = vld1q_u16(pVect1 + 8);
        h2 = vld1q_u16(pVect2 + 8);

        v1 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h1), 16));
        v2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h2), 16));
        diff = vsubq_f32(v1, v2);
        sum = vfmaq_f32(sum, diff, diff);

        v1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h1), 16));
        v2 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h2), 16));
        diff = vsubq_f32(v1, v2);
        sum = vfmaq_f32(sum, diff, diff);

        pVect1 += 16;
        pVect2 += 16;
    }

    return vaddvq_f32(sum);
}

static float
L2SqrBF16SIMD4ExtNEON(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    uint16_t *pVect1 = (uint16_t *) pVect1v;
    uint16_t *pVect2 = (uint16_t *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2;

    const uint16_t *pEnd1 = pVect1 + (qty4 << 2);

    float32x4_t sum = vdupq_n_f32(0);

    while (pVect1 < pEnd1) {
        float32x4_t v1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(pVect1), 16));
        float32x4_t v2 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(pVect2), 16));
        float32x4_t diff = vsubq_f32(v1, v2);
        sum = vfmaq_f32(sum, diff, diff);

        pVect1 += 4;
        pVect2 += 4;
    }

    return vaddvq_f32(sum);
}

static float
L2SqrBF16SIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = L2SqrBF16SIMD16ExtNEON(pVect1v, pVect2v, &qty16);
    uint16_t *pVect1 = (uint16_t *) pVect1v + qty16;
    uint16_t *pVect2 = (uint16_t *) pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = L2SqrBF16(pVect1, pVect2, &qty_left);
    return (res + res_tail);
}

static float
L2SqrBF16SIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = L2SqrBF16SIMD4ExtNEON(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    uint16_t *pVect1 = (uint16_t *) pVect1v + qty4;
    uint16_t *pVect2 = (uint16_t *) pVect2v + qty4;
    float res_tail = L2SqrBF16(pVect1, pVect2, &qty_left);

    return (res + res_tail);
}

#endif

class L2BFloat16Space : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    L2BFloat16Space(size_t dim) {
        fstdistfunc_ = L2SqrBF16;
#if defined(USE_SSE)
    #if defined(USE_AVX512)
        if (AVX512Capable())
            L2SqrBF16SIMD16Ext = L2SqrBF16SIMD16ExtAVX512;
        else if (AVXCapable())
            L2SqrBF16SIMD16Ext = L2SqrBF16SIMD16ExtAVX;
    #elif defined(USE_AVX)
        if (AVXCapable())
            L2SqrBF16SIMD16Ext = L2SqrBF16SIMD16ExtAVX;
    #endif

        if (dim % 16 == 0)
            fstdistfunc_ = L2SqrBF16SIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = L2SqrBF16SIMD4ExtSSE;
        else if (dim > 16)
            fstdistfunc_ = L2SqrBF16SIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = L2SqrBF16SIMD4ExtResiduals;
#elif defined(USE_NEON)
        if (dim % 16 == 0)
            fstdistfunc_ = L2SqrBF16SIMD16ExtNEON;
        else if (dim % 4 == 0)
            fstdistfunc_ = L2SqrBF16SIMD4ExtNEON;
        else if (dim > 16)
            fstdistfunc_ = L2SqrBF16SIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = L2SqrBF16SIMD4ExtResiduals;
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

    ~L2BFloat16Space() {}
};

}  // namespace hnswlib
