#pragma once
#include <string>
#include <vector>
#include "GKR.h"  // For struct proof
#include "proof_serialization.h"  // For DeserializeProofFromString

struct FinalProof {
  struct proof final_proof;  // The final combined proof structure
  std::string root_acc;      // Serialized representation of the final proof
  std::vector<std::string> transcripts; // per-round transcripts (for debugging/logging)
};

// Finalise the proof by deserializing the accumulator and creating a final proof structure
// The final_acc should be a serialized proof string from FA_Aggregate
// round_transcripts are the transcripts from each aggregation round (for logging)
FinalProof FinaliseProof(const std::string &final_acc,
                         const std::vector<std::string> &round_transcripts,
                         const struct proof *decoded_proof = nullptr);
