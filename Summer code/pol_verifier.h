#include "GKR.h"
#include "config_pc.hpp"

// Verification functions for different proof types
void verify_gkr(struct proof P);
void verify_matrix2matrix(struct proof Pr);
void verify_bit_decomposition(struct proof Pr);
void verify_add_proof(struct proof Pr);
struct VerifyBundle {
  std::vector<std::vector<F>> data;
  std::vector<std::vector<int>> transcript;
};

VerifyBundle BuildVerificationBundle(const std::vector<struct proof>& proofs, int num_non_acc_proofs = -1);
proof verify_proof(std::vector<struct proof> P);

// Encoding functions for different proof types
vector<F> encode_gkr_proof(struct proof P);
vector<F> encode_m2m_proof(struct proof P);
vector<F> encode_bit_decomposition(struct proof P);
vector<F> encode_hash_proof(proof P);
vector<F> encode_lookup_proof(layer_proof P);
vector<F> encode_add_proof(struct proof P);

// Transcript functions for different proof types
vector<int> get_gkr_transcript(struct proof P);
vector<int> get_m2m_transcript(struct proof P);
vector<int> get_range_proof_transcript(struct proof P);
vector<int> get_hash_transcript(struct proof P);
vector<int> get_lookup_transcript(layer_proof P);
vector<int> get_add_transcript(struct proof P);
