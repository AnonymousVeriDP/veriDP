#include "ivc_adapter.h"
#include "prove_leaf.h"
#include "proof_serialization.h"
#include "pol_verifier.h"
#include "GKR.h"
#include "proof_utils.h"
#include "polynomial.h"
#include "utils.hpp"
#include "mimc.h"
#include <iostream>
#include <vector>
#include <string> 
#include <sstream>
#include <cassert>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <ctime>

using namespace std;

namespace {

proof PadProofForEncoding(const proof &original) {
  proof padded = original;

  if (padded.q_poly.empty()) {
    padded.q_poly.push_back(quadratic_poly(F(0), F(0), F(0)));
  }

  if (padded.randomness.empty()) {
    padded.randomness.push_back(vector<F>());
  }
  if (padded.randomness.size() < 2) {
    padded.randomness.resize(2);
  }

  if (padded.sig.empty()) {
    padded.sig.push_back(vector<F>{F(0)});
  }

  if (padded.final_claims_v.size() < padded.sig.size()) {
    padded.final_claims_v.resize(padded.sig.size());
  }
  for (size_t i = 0; i < padded.sig.size(); ++i) {
    if (padded.sig[i].empty()) {
      padded.sig[i].push_back(F(0));
    }
    if (padded.final_claims_v[i].size() < padded.sig[i].size()) {
      padded.final_claims_v[i].resize(padded.sig[i].size(), F(0));
    }
  }

  if (padded.vr.size() < padded.sig.size()) {
    padded.vr.resize(padded.sig.size(), F(0));
  }
  if (padded.gr.size() < padded.sig.size()) {
    padded.gr.resize(padded.sig.size(), F(0));
  }

  return padded;
}

// Compute a constant-size hash commitment to a proof (for IVC)
// This produces a fixed-size representation regardless of proof complexity
F ComputeProofCommitment(const proof& p) {
  F commitment = F_ZERO;
  
  // Hash proof type
  commitment = mimc_hash(F(p.type), commitment);
  
  // Hash key scalar values
  commitment = mimc_hash(p.in1, commitment);
  commitment = mimc_hash(p.in2, commitment);
  commitment = mimc_hash(p.out_eval, commitment);
  
  // Hash polynomial coefficients (bounded contribution)
  for (size_t i = 0; i < p.q_poly.size() && i < 16; i++) {
    commitment = mimc_hash(p.q_poly[i].a, commitment);
    commitment = mimc_hash(p.q_poly[i].b, commitment);
    commitment = mimc_hash(p.q_poly[i].c, commitment);
  }
  // Include count to differentiate proofs with different sizes
  commitment = mimc_hash(F(p.q_poly.size()), commitment);
  
  // Hash vr values (final evaluations)
  for (size_t i = 0; i < p.vr.size() && i < 8; i++) {
    commitment = mimc_hash(p.vr[i], commitment);
  }
  commitment = mimc_hash(F(p.vr.size()), commitment);
  
  // Hash first layer of randomness (challenges)
  if (!p.randomness.empty() && !p.randomness[0].empty()) {
    for (size_t i = 0; i < p.randomness[0].size() && i < 8; i++) {
      commitment = mimc_hash(p.randomness[0][i], commitment);
    }
  }
  
  return commitment;
}

// Compute hash commitment from serialized accumulator string
F ComputeAccumulatorCommitment(const std::string& acc_str) {
  F commitment = F_ZERO;
  
  // Hash the string bytes in chunks
  // Use 8-byte chunks, field element constructor handles reduction
  for (size_t i = 0; i < acc_str.size(); i += 8) {
    uint64_t chunk = 0;
    for (size_t j = 0; j < 8 && i + j < acc_str.size(); j++) {
      chunk |= static_cast<uint64_t>(static_cast<unsigned char>(acc_str[i + j])) << (j * 8);
    }
    // F constructor handles modular reduction internally
    commitment = mimc_hash(F(chunk), commitment);
  }
  
  return commitment;
}

}

AggregationResult FA_Aggregate(const std::vector<LeafResult>& children,
                               const std::string& prev_acc) {
  AggregationResult result;
  result.serialized = prev_acc;
  result.proof_struct = proof{};
  result.ok = false;
  if (children.empty()) {
    return result; // Return previous accumulator if no children
  }
  
  // Step 1: Collect all proofs from children (current batch only)
  vector<struct proof> proofs;
  for (size_t i = 0; i < children.size(); ++i) {
    if (!children[i].transcript.empty()) {
      proofs.insert(proofs.end(), children[i].transcript.begin(), children[i].transcript.end());
    } else if (children[i].step_proof.type != 0) {
      proofs.push_back(children[i].step_proof);
    }
  }

  if (proofs.empty()) {
    return result;
  }
  
  // Number of proofs from current batch (NOT including previous accumulator)
  int num_batch_proofs = static_cast<int>(proofs.size());
  
  // Step 2: Compute constant-size commitment to previous accumulator (TRUE IVC)
  // Instead of embedding the full previous proof, we only include a hash commitment
  F prev_acc_commitment = F_ZERO;
  bool has_prev_acc = (!prev_acc.empty() && prev_acc != "ACC_INIT");
  
  if (has_prev_acc) {
    // Compute hash commitment to previous accumulator (constant size!)
    prev_acc_commitment = ComputeAccumulatorCommitment(prev_acc);
  }
  
  // Ensure proofs are properly initialized before encoding
  for (size_t i = 0; i < proofs.size(); ++i) {
    if (proofs[i].type == GKR_PROOF) {
      if (proofs[i].randomness.empty()) {
        proofs[i].randomness.push_back(vector<F>());
      }
      if (proofs[i].randomness.size() < 2) {
        while (proofs[i].randomness.size() < 2) {
          proofs[i].randomness.push_back(vector<F>());
        }
      }
      if (proofs[i].gr.empty() && !proofs[i].vr.empty()) {
        proofs[i].gr.resize(proofs[i].vr.size(), F(0));
      }
      if (proofs[i].final_claims_v.empty() && !proofs[i].sig.empty()) {
        proofs[i].final_claims_v.resize(proofs[i].sig.size());
        for (size_t j = 0; j < proofs[i].sig.size(); ++j) {
          proofs[i].final_claims_v[j].resize(proofs[i].sig[j].size(), F(0));
        }
      }
    }
  }
  
  try {
    // Normalize only the current batch proofs (NOT previous accumulator)
    vector<proof> normalized;
    normalized.reserve(proofs.size());
    for (const auto &p : proofs) {
      normalized.push_back(PadProofForEncoding(p));
    }
    
    // Build verification bundle for current batch ONLY
    VerifyBundle bundle = BuildVerificationBundle(normalized, num_batch_proofs);
    
    // IVC Data Structure (constant size per batch):
    // [batch_proof_data] + [prev_acc_commitment] + [batch_count] + [sentinel]
    std::vector<F> gkr_data;
    for (const auto &chunk : bundle.data) {
      gkr_data.insert(gkr_data.end(), chunk.begin(), chunk.end());
    }
    
    // Add constant-size IVC chain link (instead of full previous proof)
    gkr_data.push_back(prev_acc_commitment);  // Hash of previous accumulator
    gkr_data.push_back(F(has_prev_acc ? 1 : 0));  // Flag: has previous accumulator
    gkr_data.push_back(F(1));  // Sentinel

    // Derive initial randomness using Fiat-Shamir from transcript
    for (const F& elem : gkr_data) {
      mimc_hash(elem, current_randomness);
    }
    
    // Derive randomness deterministically from transcript
    std::vector<F> randomness(10);
    for (int i = 0; i < 10; i++) {
      vector<F> ctx = {current_randomness, F(i), F(10)};
      randomness[i] = mimc_multihash(ctx);
      current_randomness = randomness[i];
    }

    struct proof acc_proof = prove_verification(gkr_data, randomness, bundle.transcript);
    
    // Store the previous accumulator commitment in the proof for verification chain
    // This allows verifier to check IVC chain integrity
    if (acc_proof.eval_point.empty()) {
      acc_proof.eval_point.push_back(prev_acc_commitment);
    } else {
      acc_proof.eval_point[0] = prev_acc_commitment;
    }
    
    std::string accumulator = SerializeProofToString(acc_proof);
    result.serialized = accumulator;
    result.proof_struct = acc_proof;
    result.ok = true;
    return result;
  } catch (const std::exception &e) {
    std::cerr << "[FA_Aggregate] Error generating accumulator: " << e.what() << "\n";
  } catch (...) {
    std::cerr << "[FA_Aggregate] Unknown error generating accumulator\n";
  }

  std::cerr << "[FA_Aggregate] Falling back to previous accumulator without update" << std::endl;
  return result;
}

AggregationResult FA_AggregateTreeBatch(const std::vector<LeafResult>& children) {
  // Tree-based aggregation: no previous accumulator, just aggregate the children
  return FA_Aggregate(children, "ACC_INIT");
}
