#pragma once
#include "hnswlib.h"
#include <math.h> 

#define PI 3.14159265358979323846 /* pi */

namespace hnswlib {

static float
AngularDistance(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    return 2.0f * (acos(CosineSimilarity(pVect1, pVect2, qty_ptr)) / PI);
}

class AngularSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    AngularSpace(size_t dim) {
        fstdistfunc_ = AngularDistance;
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

~AngularSpace() {}
};

}  // namespace hnswlib
