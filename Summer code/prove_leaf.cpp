#include "prove_leaf.h"
#include "aux_witness.h"
#include "activation_gkr.h"
#include "sum_matmul_wrapper.h"
#include "proof_utils.h"
#include "quantization.h"
#include "utils.hpp"
#include "pol_verifier.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>

using namespace std;

// Prove a single RNN timestep
// Combines linear operations (matrix-vector products) using ProveSumMatMul
// and non-linear operations (activations) using ProveActivationGKR
// wh_result, wx_result, zy_result: pre-computed matrix-vector products from forward pass (avoids recomputation)
LeafResult ProveLeaf(const ProveParams &p,
                     const TimeStepInput &in,
                     const RNNWeights &weights,
                     const TimeStepOutput &out,
                     const FieldVector &wh_result,
                     const FieldVector &wx_result,
                     const FieldVector &zy_result) {
  // Get dimensions (no validation needed - assume inputs are correct)
  int m = weights.W_h.size();  // hidden size
  int n = in.x_t.size();       // input size
  int k = weights.W_y.size();  // output size
  
  // Step 1: Prove linear operations using ProveSumMatMul (SUMMER strategy)
  // Use pre-computed matrix-vector products from forward pass (no recomputation!)
  cout << "[ProveLeaf] Phase 1: proving linear relations (matmuls)\n";
  
  // Combine linear proofs using SUMMER-style block diagonal
  vector<vector<FieldVector>> matrices_combined{weights.W_h, weights.W_x, weights.W_y};
  vector<FieldVector> vectors_combined{in.h_prev, in.x_t, out.h_t};
  FieldVector combined_result;
  combined_result.reserve(wh_result.size() + wx_result.size() + zy_result.size());
  combined_result.insert(combined_result.end(), wh_result.begin(), wh_result.end());
  combined_result.insert(combined_result.end(), wx_result.begin(), wx_result.end());
  combined_result.insert(combined_result.end(), zy_result.begin(), zy_result.end());

  SumMatMulResult linear_proof = ProveSumMatMul(matrices_combined, vectors_combined, combined_result);
  
  // Step 2: Prove non-linear operations using ProveActivationGKR
  // 1. h = tanh(a) - tanh activation
  // 2. yHat = softmax(z) - softmax activation
  cout << "[ProveLeaf] Phase 2: proving activation functions\n";
  
  
  // Store quantization bits in local variable
  int num_bits = p.quantization_bits;
  if (num_bits <= 0) num_bits = Q;
  if (num_bits > Q) num_bits = Q;

  auto mask_signed = [&](const FieldVector &values) {
    FieldVector masked(values.size(), F_ZERO);
    const unsigned __int128 modulus =
        (num_bits >= 64) ? 0 : (static_cast<unsigned __int128>(1) << num_bits);
    const unsigned __int128 mask =
        (num_bits >= 64) ? (~static_cast<unsigned __int128>(0))
                         : (modulus - 1);
    for (size_t i = 0; i < values.size(); ++i) {
      __int128 raw = values[i].toint128();
      unsigned __int128 canonical = 0;
      if (raw >= 0) {
        canonical = static_cast<unsigned __int128>(raw) & mask;
      } else {
        unsigned __int128 abs = static_cast<unsigned __int128>(-raw) & mask;
        canonical = (modulus - abs) & mask;
        if (canonical == modulus) canonical = 0;
      }
      masked[i] = F(static_cast<long long>(canonical));
    }
    return masked;
  };
  
  // Build auxiliary witness for pre-activation values
  cout << "[ProveLeaf] Building auxiliary witness for tanh activation\n";
  FieldVector masked_a = mask_signed(out.a_t);
  AuxWitness aux_a = BuildAuxWitness(masked_a, num_bits);
  
  // Prove tanh activation: h = tanh(a)
  // Use tanh_layer_data directly (more efficient, avoids ExpBatch lookups)
  cout << "[ProveLeaf] Proving tanh activation (GKR circuit)\n";
  ActivationProofs tanh_proofs = ProveActivationGKR(masked_a, aux_a, out.h_t,
                                                    out.tanh_layer_data,
                                                    ACTIVATION_TANH, num_bits);
  
  // Build auxiliary witness for pre-softmax values
  cout << "[ProveLeaf] Building auxiliary witness for softmax activation\n";
  FieldVector masked_z = mask_signed(out.z_t);
  AuxWitness aux_z = BuildAuxWitness(masked_z, num_bits);
  
  // Prove softmax activation: yHat = softmax(z)
  // Use softmax_layer_data directly (more efficient, avoids ExpBatch lookups)
  cout << "[ProveLeaf] Proving softmax activation (GKR circuit)\n";
  ActivationProofs softmax_proofs = ProveActivationGKR(masked_z, aux_z, out.yHat_t,
                                                       out.softmax_layer_data,
                                                       ACTIVATION_SOFTMAX, num_bits);
  
  // Step 3: Combine all proofs into a single leaf proof
  LeafResult result;
  result.transcript.reserve(3);
  result.logup_proofs.reserve(2);
  result.transcript.push_back(linear_proof.proof);
  result.transcript.push_back(tanh_proofs.range);
  result.transcript.push_back(softmax_proofs.range);
  result.logup_proofs.push_back(tanh_proofs.logup);
  result.logup_proofs.push_back(softmax_proofs.logup);
  
  cout << "[ProveLeaf] Phase summary: "
            << result.transcript.size() << " arithmetic proofs, "
            << result.logup_proofs.size() << " GKR proofs\n";
  
  return result;
}
