#pragma once
#include "hnswlib.h"

#include <cmath>
#include <stdexcept>

namespace hnswlib {

// Great-circle distance between two (latitude, longitude) points given in
// degrees, computed with the haversine formula and returned in kilometers.
static const float GEO_DEG_TO_RAD = 0.017453292519943295f;  // pi / 180
static const float GEO_EARTH_MEAN_RADIUS_KM = 6371.0088f;

static float
GeoDegreesDistance(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const float *pVect1 = (const float *) pVect1v;
    const float *pVect2 = (const float *) pVect2v;

    float lat1 = pVect1[0] * GEO_DEG_TO_RAD;
    float lon1 = pVect1[1] * GEO_DEG_TO_RAD;
    float lat2 = pVect2[0] * GEO_DEG_TO_RAD;
    float lon2 = pVect2[1] * GEO_DEG_TO_RAD;

    float sin_dlat = std::sin((lat2 - lat1) * 0.5f);
    float sin_dlon = std::sin((lon2 - lon1) * 0.5f);

    float h = sin_dlat * sin_dlat + std::cos(lat1) * std::cos(lat2) * sin_dlon * sin_dlon;
    // clamp to [0, 1] to guard against floating point drift before asin
    h = h < 0.0f ? 0.0f : (h > 1.0f ? 1.0f : h);

    return 2.0f * GEO_EARTH_MEAN_RADIUS_KM * std::asin(std::sqrt(h));
}

class GeoDegreesSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    GeoDegreesSpace(size_t dim) {
        if (dim != 2)
            throw std::runtime_error("GeoDegreesSpace requires dim=2 (latitude, longitude).");
        fstdistfunc_ = GeoDegreesDistance;
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

    ~GeoDegreesSpace() {}
};
}  // namespace hnswlib
