#pragma once
#include "prove_leaf.h"
#include "proof_serialization.h"
#include <string>
#include <vector>

struct AggregationResult {
  std::string serialized;
  struct proof proof_struct;
  bool ok{false};
};

// Aggregates k leaf proofs into one accumulator string and proof struct.
AggregationResult FA_Aggregate(const std::vector<LeafResult>& children,
                               const std::string& prev_acc);

// Tree-based aggregation: aggregates a batch of proofs (no previous accumulator).
// Used for intermediate tree nodes.
AggregationResult FA_AggregateTreeBatch(const std::vector<LeafResult>& children);