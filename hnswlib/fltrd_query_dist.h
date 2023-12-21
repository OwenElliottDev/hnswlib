#pragma once
#include "hnswlib.h"

namespace hnswlib {
    template<typename dist_t>
    static float FltrdQueryDistance(const void *pVect1, const void *pVect2, const void *fltr_vec, const void *qty_ptr, const float threshold, DISTFUNC<dist_t> distance_func) {
        float d_q_v = distance_func(pVect1, pVect2, qty_ptr);
        float d_q_f = distance_func(fltr_vec, pVect2, qty_ptr);

        if (d_q_f > (-1.0 * threshold)) {
            d_q_f = 0.0;
        }
        
        return d_q_v + d_q_f;
    }
}