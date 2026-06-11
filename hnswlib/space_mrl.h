#pragma once
#include "hnswlib.h"

#include <stdexcept>

namespace hnswlib {

// Space for Matryoshka (MRL) embeddings: graph construction and traversal use
// only the first scan_dim dimensions of the stored vectors, while vectors are
// stored at full dimensionality so search candidates can be reranked with the
// full-dimension distance (see HierarchicalNSW::searchKnnMrl).
//
// Takes ownership of both inner spaces. The scan space must be constructed
// with the truncated dimensionality and the full space with the full
// dimensionality of the same metric, e.g.
//   MrlSpace(new L2Space(64), new L2Space(512))
class MrlSpace : public SpaceInterface<float> {
    SpaceInterface<float>* scan_space_;
    SpaceInterface<float>* full_space_;

 public:
    MrlSpace(SpaceInterface<float>* scan_space, SpaceInterface<float>* full_space)
        : scan_space_(scan_space), full_space_(full_space) {
        if (scan_space_->get_data_size() >= full_space_->get_data_size()) {
            delete scan_space_;
            delete full_space_;
            throw std::runtime_error("MRL scan dimensionality must be smaller than the full dimensionality.");
        }
    }

    size_t get_data_size() { return full_space_->get_data_size(); }

    DISTFUNC<float> get_dist_func() { return scan_space_->get_dist_func(); }

    void* get_dist_func_param() { return scan_space_->get_dist_func_param(); }

    DISTFUNC<float> get_full_dist_func() { return full_space_->get_dist_func(); }

    void* get_full_dist_func_param() { return full_space_->get_dist_func_param(); }

    ~MrlSpace() {
        delete scan_space_;
        delete full_space_;
    }
};
}  // namespace hnswlib
