#pragma once
#include "hnswlib.h"
#include <math.h> 

namespace hnswlib {

static float 
CosineSimilarity(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    float dotProduct = 0.0;
    float normA = 0.0;
    float normB = 0.0;
    
    const float *vect1 = (const float *) pVect1;
    const float *vect2 = (const float *) pVect2;

    for (size_t i = 0; i < qty; ++i) {
        dotProduct += vect1[i] * vect2[i];
        normA += vect1[i] * vect1[i];
        normB += vect2[i] * vect2[i];
    }

    float denominator = sqrt(normA) * sqrt(normB);

    if (denominator == 0.0f)
        return 0.0f;

    return dotProduct / denominator;
}

static float
ThetaDistance(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    float theta = acosf(CosineSimilarity(pVect1, pVect2, qty_ptr));
    return theta;
}

class ThetaSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    ThetaSpace(size_t dim) {
        fstdistfunc_ = ThetaDistance;
        dim_ = dim;
        data_size_ = dim * sizeof(float);
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

~ThetaSpace() {}
};

}  // namespace hnswlib
