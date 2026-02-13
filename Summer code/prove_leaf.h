#pragma once

#include "GKR.h"
#include <vector>

// Result structure for leaf proof generation (VeriDP version - no RNN dependencies)
struct LeafResult {
    struct proof step_proof;               // The main proof for this step
    std::vector<struct proof> transcript;  // All proofs generated in this step
    std::vector<F> witness;                // Witness data
    bool ok{false};                        // Success flag
};
