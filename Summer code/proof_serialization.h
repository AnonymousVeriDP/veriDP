#pragma once

#include "GKR.h"
#include "types.h"
#include <string>
#include <vector>

// Serialize a proof to a vector of field elements
// Returns the encoded proof data and a transcript describing its structure
struct SerializedProof {
  std::vector<F> data;           // Encoded proof data (field elements)
  std::vector<int> transcript;   // Metadata describing proof structure
  int proof_type;                // Proof type (GKR_PROOF, MATMUL_PROOF, RANGE_PROOF, etc.)
};

// Serialize a proof to field elements + transcript
SerializedProof SerializeProof(const struct proof &P);

// Deserialize a proof from field elements + transcript
// Note: This reconstructs the proof structure, but some fields may be incomplete
// as the encoding functions only serialize essential verification data
struct proof DeserializeProof(const SerializedProof &serialized);

// Serialize a proof to a string (for storage/transmission)
// Uses base64-like encoding of field elements
std::string SerializeProofToString(const struct proof &P);

// Deserialize a proof from a string
struct proof DeserializeProofFromString(const std::string &serialized_str);

// Helper: Convert field element to string representation
std::string FieldElementToString(const F &elem);

// Helper: Convert string to field element
F StringToFieldElement(const std::string &str);

