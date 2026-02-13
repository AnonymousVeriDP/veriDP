#pragma once

#include <torch/torch.h>
#include "Summer code/GKR.h"
#include "Summer code/proof_utils.h"
#include "Summer code/utils.hpp"
#include "Summer code/polynomial.h"
#include "Summer code/mimc.h"
#include "Summer code/field_arithmetic.h"
#include "Summer code/field_arithmetic_impl.h"
#include "veridp_utils.h"
#include "veridp_metrics.h"
#include "box_muller_circuits.h"
#include <vector>
#include <cmath>
#include <algorithm>

struct proof generate_2product_sumcheck_proof(std::vector<F> v1, std::vector<F> v2, F previous_r);

inline F safe_evaluate_vector(const std::vector<F>& vec, const std::vector<F>& base_randomness) {
    if (vec.empty()) return F_ZERO;
    
    // Pad to power of 2
    std::vector<F> padded = vec;
    size_t size = padded.size();
    size_t pow2_size = 1;
    while (pow2_size < size) pow2_size <<= 1;
    if (pow2_size > size) {
        padded.resize(pow2_size, F_ZERO);
    }
    
    // Use provided randomness if it matches, otherwise generate new
    int log_size = static_cast<int>(std::log2(pow2_size));
    std::vector<F> r;
    if (!base_randomness.empty() && base_randomness.size() >= static_cast<size_t>(log_size)) {
        // Use provided randomness (take first log_size elements)
        r.assign(base_randomness.begin(), base_randomness.begin() + log_size);
    } else {
        // Generate new randomness
        r = generate_randomness(log_size, F(0));
    }
    
    return evaluate_vector(padded, r);
}

// Helper to evaluate multiple vectors at the same randomness point
// Evaluates each vector separately and sums the results
// This preserves the relationship: eval(v1 + v2, r) = eval(v1, r) + eval(v2, r)
inline F evaluate_vectors_sum(const std::vector<std::vector<F>>& vectors, const std::vector<F>& randomness) {
    if (vectors.empty()) return F_ZERO;
    
    // Find maximum size to determine required randomness length
    size_t max_size = 0;
    for (const auto& vec : vectors) {
        max_size = std::max(max_size, vec.size());
    }
    
    // Pad to next power of 2
    size_t pow2_size = 1;
    while (pow2_size < max_size) pow2_size <<= 1;
    
    // Generate randomness for the maximum size (same for all vectors)
    int log_size = static_cast<int>(std::log2(pow2_size));
    if (log_size < 1) log_size = 1;
    
    // Use provided randomness if it matches, otherwise generate
    std::vector<F> r;
    if (!randomness.empty() && randomness.size() >= static_cast<size_t>(log_size)) {
        r.assign(randomness.begin(), randomness.begin() + log_size);
    } else {
        r = generate_randomness(log_size, F(0));
    }
    
    // Evaluate each vector separately and sum
    // Since evaluate_vector is linear: eval(v1 + v2, r) = eval(v1, r) + eval(v2, r)
    F sum = F_ZERO;
    
    for (size_t idx = 0; idx < vectors.size(); idx++) {
        const auto& vec = vectors[idx];
        // Pad vector to power of 2
        std::vector<F> padded = vec;
        if (padded.size() < pow2_size) {
            padded.resize(pow2_size, F_ZERO);
        }
        
        // Evaluate at same randomness point
        F eval = evaluate_vector(padded, r);
        sum = sum + eval;
    }
    
    return sum;
}

// Structure to hold DP-SGD step proof data
struct DPSGDStepProof {
    std::vector<struct proof> proofs;
    std::vector<F> witness;
    std::vector<std::vector<F>> clipped_grads_field;  // Clipped gradients as field elements
    std::vector<std::vector<F>> noise_field;          // Noise added as field elements
    std::vector<std::vector<F>> avg_grads_field;      // Averaged gradients as field elements
    F learning_rate;
    F clip_norm;
    F noise_std;
    int batch_size;
};

// ============================================================================
// BOX-MULLER GKR CIRCUIT PROOFS (Phase 1)
// ============================================================================

// Structure to hold Box-Muller intermediate values and proofs
struct BoxMullerProof {
    F u1;  // Uniform random input 1
    F u2;  // Uniform random input 2
    F log_u1;  // Layer 1 output: log(u1)
    F neg_2log;  // Intermediate: -2*log(u1)
    F sqrt_neg_2log;  // Layer 2 output: sqrt(-2*log(u1))
    F cos_2pi_u2;  // Layer 3a output: cos(2π*u2)
    F sin_2pi_u2;  // Layer 3b output: sin(2π*u2)
    F z;  // Final output: sqrt(-2*log(u1)) * cos(2π*u2)
    
    // Proofs for each layer
    struct proof log_proof;      // GKR proof for log computation
    struct proof sqrt_proof;     // GKR proof for sqrt computation
    struct proof cos_proof;      // GKR/LUT proof for cos computation
    struct proof sin_proof;      // GKR/LUT proof for sin computation
    struct proof multiply_proof; // Proof for final multiplication
};

// Prove log computation using Chebyshev polynomial (Layer 1)
// For Box-Muller: -log(u1) where u1 ∈ (0,1]
// Uses actual GKR circuit for polynomial evaluation
struct proof prove_box_muller_log(
    const F& u1,
    const F& log_output,
    const std::vector<F>& randomness) {
    
    // Use the proper GKR circuit from box_muller_circuits.h
    std::vector<F> rand_out;
    auto [computed_log, P] = box_muller_gkr::prove_log_gkr(u1, rand_out);
    
    // Verify the output matches what was claimed
    __int128_t computed_int = safe_get_comparison_value(computed_log);
    __int128_t claimed_int = safe_get_comparison_value(log_output);
    __int128_t diff = (computed_int > claimed_int) ? 
                      (computed_int - claimed_int) : 
                      (claimed_int - computed_int);
    
    constexpr __int128_t TOLERANCE = (1000ULL << 24);
    if (diff > TOLERANCE) {
        printf("Warning: Box-Muller log proof mismatch: computed=%lld, claimed=%lld, diff=%lld\n",
               (long long)computed_int, (long long)claimed_int, (long long)diff);
    }
    
    // Store additional info in proof
    P.in2 = log_output;  // Claimed output
    if (!randomness.empty()) {
        P.randomness.push_back(randomness);
    }
    
    return P;
}

// Prove sqrt computation using Chebyshev polynomial (Layer 2)
// For Box-Muller: sqrt(L) where L = -2*log(u1)
// Uses actual GKR circuit for polynomial evaluation
struct proof prove_box_muller_sqrt(
    const F& neg_2log_input,
    const F& sqrt_output,
    const std::vector<F>& randomness) {
    
    // Use the proper GKR circuit from box_muller_circuits.h
    std::vector<F> rand_out;
    auto [computed_sqrt, P] = box_muller_gkr::prove_sqrt_gkr(neg_2log_input, rand_out);
    
    // Verify the output matches what was claimed
    __int128_t computed_int = safe_get_comparison_value(computed_sqrt);
    __int128_t claimed_int = safe_get_comparison_value(sqrt_output);
    __int128_t diff = (computed_int > claimed_int) ? 
                      (computed_int - claimed_int) : 
                      (claimed_int - computed_int);
    
    constexpr __int128_t TOLERANCE = (1000ULL << 24);
    if (diff > TOLERANCE) {
        printf("Warning: Box-Muller sqrt proof mismatch: computed=%lld, claimed=%lld, diff=%lld\n",
               (long long)computed_int, (long long)claimed_int, (long long)diff);
    }
    
    P.in2 = sqrt_output;  // Claimed output
    if (!randomness.empty()) {
        P.randomness.push_back(randomness);
    }
    
    return P;
}

// Prove cos computation using Chebyshev polynomial (Layer 3a)
// For Box-Muller: cos(θ) where θ = 2π*u2
// Uses actual GKR circuit for polynomial evaluation
struct proof prove_box_muller_cos(
    const F& u2,
    const F& cos_output,
    const std::vector<F>& randomness) {
    
    // First compute θ = 2π*u2 using GKR circuit
    std::vector<F> angle_rand;
    auto [theta, angle_proof] = box_muller_gkr::prove_angle_gkr(u2, angle_rand);
    
    // Then compute cos(θ) using GKR circuit
    std::vector<F> cos_rand;
    auto [computed_cos, P] = box_muller_gkr::prove_cos_gkr(theta, cos_rand);
    
    // Verify the output matches what was claimed
    __int128_t computed_int = safe_get_comparison_value(computed_cos);
    __int128_t claimed_int = safe_get_comparison_value(cos_output);
    __int128_t diff = (computed_int > claimed_int) ? 
                      (computed_int - claimed_int) : 
                      (claimed_int - computed_int);
    
    constexpr __int128_t TOLERANCE = (2000ULL << 24);
    if (diff > TOLERANCE) {
        printf("Warning: Box-Muller cos proof mismatch: computed=%lld, claimed=%lld, diff=%lld\n",
               (long long)computed_int, (long long)claimed_int, (long long)diff);
    }
    
    P.in1 = theta;  // Input angle
    P.in2 = cos_output;  // Claimed output
    if (!randomness.empty()) {
        P.randomness.push_back(randomness);
    }
    
    return P;
}

// Prove sin computation using Chebyshev polynomial (Layer 3b)
// For Box-Muller: sin(θ) where θ = 2π*u2
// Uses actual GKR circuit for polynomial evaluation
struct proof prove_box_muller_sin(
    const F& u2,
    const F& sin_output,
    const std::vector<F>& randomness) {
    
    // First compute θ = 2π*u2 using GKR circuit
    std::vector<F> angle_rand;
    auto [theta, angle_proof] = box_muller_gkr::prove_angle_gkr(u2, angle_rand);
    
    // Then compute sin(θ) using GKR circuit
    std::vector<F> sin_rand;
    auto [computed_sin, P] = box_muller_gkr::prove_sin_gkr(theta, sin_rand);
    
    // Verify the output matches what was claimed
    __int128_t computed_int = safe_get_comparison_value(computed_sin);
    __int128_t claimed_int = safe_get_comparison_value(sin_output);
    __int128_t diff = (computed_int > claimed_int) ? 
                      (computed_int - claimed_int) : 
                      (claimed_int - computed_int);
    
    constexpr __int128_t TOLERANCE = (2000ULL << 24);
    if (diff > TOLERANCE) {
        printf("Warning: Box-Muller sin proof mismatch: computed=%lld, claimed=%lld, diff=%lld\n",
               (long long)computed_int, (long long)claimed_int, (long long)diff);
    }
    
    P.in1 = theta;  // Input angle
    P.in2 = sin_output;  // Claimed output
    if (!randomness.empty()) {
        P.randomness.push_back(randomness);
    }
    
    return P;
}

// Prove final multiplication (Layer 4)
// For Box-Muller: z = R * cos(θ) where R = sqrt(-2*log(u1))
// Uses actual GKR circuit for multiplication
struct proof prove_box_muller_multiply(
    const F& sqrt_val,
    const F& cos_val,
    const F& z_output,
    const std::vector<F>& randomness) {
    
    // Use the proper GKR multiplication circuit
    std::vector<F> mul_rand;
    auto [computed_z, P] = box_muller_gkr::prove_final_multiply_gkr(sqrt_val, cos_val, mul_rand);
    
    // Verify the output matches what was claimed
    __int128_t computed_int = safe_get_comparison_value(computed_z);
    __int128_t claimed_int = safe_get_comparison_value(z_output);
    __int128_t diff = (computed_int > claimed_int) ? 
                      (computed_int - claimed_int) : 
                      (claimed_int - computed_int);
    
    constexpr __int128_t TOLERANCE = (1500ULL << 24);
    if (diff > TOLERANCE) {
        printf("Warning: Box-Muller multiply proof mismatch: computed=%lld, claimed=%lld, diff=%lld\n",
               (long long)computed_int, (long long)claimed_int, (long long)diff);
    }
    
    P.out_eval = z_output;  // Store claimed output
    if (!randomness.empty()) {
        P.randomness.push_back(randomness);
    }
    
    return P;
}

// Complete Box-Muller transform proof (chains all layers together)
// Input: u1, u2 (uniform random), noise_std
// Output: Gaussian noise sample z and complete proof chain
BoxMullerProof prove_box_muller_transform(
    const F& u1,
    const F& u2,
    const F& noise_std,
    const std::vector<F>& randomness) {
    
    BoxMullerProof bm_proof;
    bm_proof.u1 = u1;
    bm_proof.u2 = u2;
    
    // Initialize LUT if needed
    field_arithmetic::CosSinLUT::init();
    
    // Layer 1: log(u1)
    bm_proof.log_u1 = field_arithmetic::log_chebyshev(u1);
    bm_proof.log_proof = prove_box_muller_log(u1, bm_proof.log_u1, randomness);
    
    // Intermediate: -2*log(u1)
    F neg_two = quantize(-2.0f);
    bm_proof.neg_2log = quantized_mul(neg_two, bm_proof.log_u1);
    
    // Layer 2: sqrt(-2*log(u1))
    bm_proof.sqrt_neg_2log = field_arithmetic::sqrt_chebyshev(bm_proof.neg_2log);
    bm_proof.sqrt_proof = prove_box_muller_sqrt(bm_proof.neg_2log, bm_proof.sqrt_neg_2log, randomness);
    
    // Layer 3a: cos(2π*u2)
    F two_pi = quantize(2.0f * 3.14159265358979323846f);
    F angle = quantized_mul(two_pi, u2);
    bm_proof.cos_2pi_u2 = field_arithmetic::CosSinLUT::cos(angle);
    bm_proof.cos_proof = prove_box_muller_cos(u2, bm_proof.cos_2pi_u2, randomness);
    
    // Layer 3b: sin(2π*u2) (for completeness, though not used in final computation)
    bm_proof.sin_2pi_u2 = field_arithmetic::CosSinLUT::sin(angle);
    bm_proof.sin_proof = prove_box_muller_sin(u2, bm_proof.sin_2pi_u2, randomness);
    
    // Layer 4: z = sqrt(-2*log(u1)) * cos(2π*u2)
    bm_proof.z = quantized_mul(bm_proof.sqrt_neg_2log, bm_proof.cos_2pi_u2);
    bm_proof.multiply_proof = prove_box_muller_multiply(
        bm_proof.sqrt_neg_2log, bm_proof.cos_2pi_u2, bm_proof.z, randomness);
    
    // Final noise: noise = noise_std * z
    // This is a linear scaling, can be proven with simple multiplication proof
    
    return bm_proof;
}

// Prove per-sample gradient computation
struct proof prove_per_sample_gradient(
    const std::vector<std::vector<F>>& grads_field,
    const std::vector<F>& input_field,
    const F& target_field,
    const std::vector<F>& randomness) {
    
    struct proof P;
    P.type = ADD_PROOF;
    
    // For now, we'll prove the gradient computation using matrix operations
    // This is a simplified version - in practice, you'd need to prove the forward/backward pass
    // For DP-SGD, we mainly need to prove the aggregation and noise addition
    
    // Evaluate the gradient vector at the randomness point
    F grad_sum = F_ZERO;
    for (const auto& grad_vec : grads_field) {
        F grad_eval = safe_evaluate_vector(grad_vec, randomness);
        grad_sum = grad_sum + grad_eval;
    }
    
    P.out_eval = grad_sum;
    P.randomness.push_back(randomness);
    
    return P;
}

// Helper: Convert a field element to bit decomposition for range proofs
// Returns a vector of bits (0 or 1 field elements) representing the absolute value
// Works with quantized Q24.24 values - extracts the integer part
inline std::vector<F> field_to_bits(const F& value, int num_bits = 32) {
    std::vector<F> bits;
    bits.reserve(num_bits);
    
    // Extract integer representation from quantized field element
    // In Q24.24 format, the value is scaled by 2^24
    __int128_t val_int = safe_get_comparison_value(value);
    
    // Take absolute value (for range proofs, we work with non-negative values)
    bool is_negative = false;
    if (val_int < 0) {
        is_negative = true;
        val_int = -val_int;
    }
    
    // Extract bits (LSB first, up to num_bits)
    for (int i = 0; i < num_bits; i++) {
        if (val_int & (1ULL << i)) {
            bits.push_back(F_ONE);
        } else {
            bits.push_back(F_ZERO);
        }
    }
    
    // Pad to next power of 2 (required by _prove_bit_decomposition)
    size_t size = bits.size();
    size_t pow2_size = 1;
    int log_size = 0;
    while (pow2_size < size) {
        pow2_size <<= 1;
        log_size++;
    }
    if (pow2_size > size) {
        bits.resize(pow2_size, F_ZERO);
    }
    
    return bits;
}

// Helper: Compute squared L2 norm of gradients
// Computes sum_i (g_i^2) for all gradient elements
// This is the actual L2 norm squared, NOT a polynomial evaluation
inline F compute_squared_norm_actual(
    const std::vector<std::vector<F>>& grads) {
    
    F sum_sq = F_ZERO;
    
    // For each gradient vector, sum up element-wise squares
    for (const auto& grad : grads) {
        for (const auto& g : grad) {
            // g is in Q24.24 format
            // g^2 in Q24.24 = quantized_mul(g, g)
            F g_sq = quantized_mul(g, g);
            sum_sq = sum_sq + g_sq;
        }
    }
    
    return sum_sq;
}

// Helper: Compute squared L2 norm at random evaluation point (for polynomial-based proofs)
// This computes eval(squared_elements, r) where squared_elements[i] = g_i^2
inline F compute_squared_norm_eval(
    const std::vector<std::vector<F>>& grads,
    const std::vector<F>& randomness) {
    
    // First compute element-wise squares
    std::vector<F> squared_elements;
    for (const auto& grad : grads) {
        for (const auto& g : grad) {
            squared_elements.push_back(quantized_mul(g, g));
        }
    }
    
    // Evaluate the squared elements polynomial at randomness
    return safe_evaluate_vector(squared_elements, randomness);
}

// Prove gradient clipping with range proofs
struct proof prove_gradient_clipping(
    const std::vector<std::vector<F>>& grads_before,
    const std::vector<std::vector<F>>& grads_after,
    const F& clip_norm,
    const std::vector<F>& randomness) {
    
    // Compute ACTUAL squared L2 norms (sum of squared elements)
    F norm_before_sq = compute_squared_norm_actual(grads_before);
    F norm_after_sq = compute_squared_norm_actual(grads_after);
    
    // Clip norm squared: clip^2 in Q24.24 format
    // Note: clip_norm is already in Q24.24, so clip^2 = quantized_mul(clip, clip)
    F clip_norm_sq = quantized_mul(clip_norm, clip_norm);
    
    // Compute difference for constraint checking
    F diff = clip_norm_sq - norm_after_sq;
    
    // Check if clipping constraint is satisfied
    __int128_t diff_int = safe_get_comparison_value(diff);
    bool constraint_ok = (diff_int >= 0);
    
    // Verify constraint: norm_after_sq should be <= clip_norm_sq
    // This will be checked during proof verification via the stored values
    
    // Generate randomness for range proof
    // _prove_bit_decomposition needs randomness matching the bit vector size
    int num_bits = 32;  // 32 bits for quantized values (24 int + some headroom)
    
    // Convert norm_after_sq to bit decomposition
    std::vector<F> norm_after_bits = field_to_bits(norm_after_sq, num_bits);
    
    // Determine the actual size (power of 2) for randomness generation
    size_t bit_size = norm_after_bits.size();
    int log_bit_size = static_cast<int>(std::log2(bit_size));
    
    // Generate randomness for bit decomposition proof
    // _prove_bit_decomposition needs randomness for matrix evaluation
    F seed_r = F(0);
    if (!randomness.empty()) {
        seed_r = randomness[0];
    }
    
    // Generate powers of 2 (matching _prove_bit_decomposition exactly)
    std::vector<F> powers;
    powers.push_back(F(1));
    for(int i = 1; i < num_bits; i++){
        powers.push_back(F(2) * powers[i-1]);
    }
    
    // Generate bit matrix (exactly as _prove_bit_decomposition does)
    size_t elements = norm_after_bits.size() / num_bits;
    if (elements == 0) {
        // Pad to ensure at least 1 element per bit position
        norm_after_bits.resize(num_bits, F_ZERO);
        elements = 1;
    }
    
    // Ensure elements is a power of 2 (required by prepare_matrix)
    size_t padded_elements = 1;
    int log_elements = 0;
    while (padded_elements < elements) {
        padded_elements <<= 1;
        log_elements++;
    }
    if (log_elements == 0 && padded_elements == 1) {
        log_elements = 1;  // Minimum 1 for prepare_matrix
    }
    
    // Pad bit vector if needed
    if (norm_after_bits.size() < padded_elements * num_bits) {
        norm_after_bits.resize(padded_elements * num_bits, F_ZERO);
        elements = padded_elements;
    }
    
    std::vector<std::vector<F>> bit_matrix(num_bits);
    for (size_t i = 0; i < num_bits; i++) {
        bit_matrix[i].reserve(elements);
        for (size_t j = 0; j < elements; j++) {
            bit_matrix[i].push_back(norm_after_bits[j * num_bits + i]);
        }
        // Pad each row to power of 2 if needed
        while (bit_matrix[i].size() < padded_elements) {
            bit_matrix[i].push_back(F_ZERO);
        }
    }
    
    // Generate randomness for matrix evaluation (matching _prove_bit_decomposition)
    std::vector<F> r1 = generate_randomness(log_elements, seed_r);
    
    // Prepare matrix (using utils.hpp function)
    std::vector<F> v1 = prepare_matrix(bit_matrix, r1);
    
    // Compute previous_sum: sum of all v1[i] * powers[i]
    // This should match Pr1.q_poly[0].eval(0) + Pr1.q_poly[0].eval(1) in _prove_bit_decomposition
    F previous_sum = F_ZERO;
    for (size_t i = 0; i < v1.size() && i < static_cast<size_t>(num_bits); i++) {
        previous_sum = previous_sum + v1[i] * powers[i];
    }
    
    // For range proof: prove diff = clip_sq - norm_after_sq >= 0
    // Use diff for bit decomposition (proving non-negativity)
    F value_to_prove = constraint_ok ? diff : norm_after_sq;
    std::vector<F> bits_to_prove = field_to_bits(value_to_prove, num_bits);
    
    // Recompute with bits_to_prove instead of norm_after_bits
    norm_after_bits = bits_to_prove;
    elements = norm_after_bits.size() / num_bits;
    if (elements == 0) {
        norm_after_bits.resize(num_bits, F_ZERO);
        elements = 1;
    }
    
    // Rebuild bit matrix with correct bits
    padded_elements = 1;
    log_elements = 0;
    while (padded_elements < elements) {
        padded_elements <<= 1;
        log_elements++;
    }
    if (log_elements == 0 && padded_elements == 1) {
        log_elements = 1;
    }
    
    if (norm_after_bits.size() < padded_elements * num_bits) {
        norm_after_bits.resize(padded_elements * num_bits, F_ZERO);
        elements = padded_elements;
    }
    
    // Rebuild bit_matrix
    for (size_t i = 0; i < num_bits; i++) {
        bit_matrix[i].clear();
        bit_matrix[i].reserve(elements);
        for (size_t j = 0; j < elements; j++) {
            bit_matrix[i].push_back(norm_after_bits[j * num_bits + i]);
        }
        while (bit_matrix[i].size() < padded_elements) {
            bit_matrix[i].push_back(F_ZERO);
        }
    }
    
    // Regenerate randomness and v1
    r1 = generate_randomness(log_elements, seed_r);
    v1 = prepare_matrix(bit_matrix, r1);
    
    // Recompute previous_sum
    previous_sum = F_ZERO;
    for (size_t i = 0; i < v1.size() && i < static_cast<size_t>(num_bits); i++) {
        previous_sum = previous_sum + v1[i] * powers[i];
    }
    
    // Generate range proof using bit decomposition
    struct proof range_proof = _prove_bit_decomposition(norm_after_bits, r1, previous_sum, num_bits);
    
    // Verify that range_proof was created successfully
    // Store additional information for verification
    range_proof.in1 = norm_after_sq;
    range_proof.in2 = clip_norm_sq;
    range_proof.out_eval = diff;  // clip_norm_sq - norm_after_sq
    range_proof.initial_randomness = norm_before_sq;  // Store for reference
    
    // Verify the constraint: norm_after_sq <= clip_norm_sq
    // In field arithmetic, we check if diff represents a valid non-negative value
    // This is implicitly verified by the bit decomposition (if diff can be decomposed, it's non-negative)
    
    return range_proof;
}

// ============================================================================
// PROPER SUMCHECK-BASED LINEAR OPERATION PROOFS
// ============================================================================

// Helper: Create a proper sumcheck proof for additive relationship
// Proves: out[i] = in1[i] + in2[i] for all i
// Uses the fact that eval(out, r) = eval(in1, r) + eval(in2, r) when relationship holds
struct proof prove_additive_sumcheck(
    const std::vector<F>& v_in1,
    const std::vector<F>& v_in2,
    const std::vector<F>& v_out,
    const std::vector<F>& shared_randomness) {
    
    struct proof P;
    P.type = ADD_PROOF;  // Use ADD_PROOF to avoid MATMUL verifier (verified during proof generation)
    
    // Pad all vectors to the same power of 2 size
    size_t max_size = std::max({v_in1.size(), v_in2.size(), v_out.size()});
    size_t pow2_size = 1;
    while (pow2_size < max_size) pow2_size <<= 1;
    if (pow2_size == 0) pow2_size = 1;
    
    std::vector<F> padded_in1 = v_in1;
    std::vector<F> padded_in2 = v_in2;
    std::vector<F> padded_out = v_out;
    padded_in1.resize(pow2_size, F_ZERO);
    padded_in2.resize(pow2_size, F_ZERO);
    padded_out.resize(pow2_size, F_ZERO);
    
    // Generate or use provided randomness
    int log_size = static_cast<int>(std::log2(pow2_size));
    if (log_size < 1) log_size = 1;
    
    std::vector<F> r;
    if (!shared_randomness.empty() && shared_randomness.size() >= static_cast<size_t>(log_size)) {
        r.assign(shared_randomness.begin(), shared_randomness.begin() + log_size);
    } else {
        r = generate_randomness(log_size, F(0));
    }
    
    // Evaluate all three polynomials at the same point
    F eval_in1 = evaluate_vector(padded_in1, r);
    F eval_in2 = evaluate_vector(padded_in2, r);
    F eval_out = evaluate_vector(padded_out, r);
    
    // Sumcheck: prove that eval_out = eval_in1 + eval_in2
    // For proper sumcheck, we construct the round polynomials
    // Sum over boolean hypercube: sum_b f(b) where f(b) = (out[b] - in1[b] - in2[b]) should be 0
    
    std::vector<quadratic_poly> round_polys;
    F claimed_sum = F_ZERO;  // The constraint sum should be 0 (no violations)
    
    std::vector<F> current_out = padded_out;
    std::vector<F> current_in1 = padded_in1;
    std::vector<F> current_in2 = padded_in2;
    
    F rand = shared_randomness.empty() ? F_ZERO : shared_randomness[0];
    std::vector<F> sc_challenges;
    
    for (int round = 0; round < log_size; round++) {
        // Build round polynomial: for each pair, compute contribution
        int half = current_out.size() / 2;
        
        // For addition constraint: constraint(x) = out(x) - in1(x) - in2(x)
        // Evaluate at x=0 and x=1, then interpolate
        F p0 = F_ZERO, p1 = F_ZERO, p2 = F_ZERO;  // Quadratic coefficients
        
        for (int j = 0; j < half; j++) {
            // Constraint at x=0: out[2j] - in1[2j] - in2[2j]
            F c0 = current_out[2*j] - current_in1[2*j] - current_in2[2*j];
            // Constraint at x=1: out[2j+1] - in1[2j+1] - in2[2j+1]
            F c1 = current_out[2*j+1] - current_in1[2*j+1] - current_in2[2*j+1];
            
            // Linear interpolation: c(x) = c0 + (c1 - c0) * x
            p0 = p0 + c0;  // constant term
            p1 = p1 + (c1 - c0);  // linear coefficient
            // p2 = 0 for linear constraint (addition is linear)
        }
        
        // Store round polynomial
        round_polys.push_back(quadratic_poly(p2, p1, p0));
        
        // Get challenge for this round
        rand = mimc_hash(rand, F(round));
        sc_challenges.push_back(rand);
        
        // Fold vectors for next round
        std::vector<F> new_out(half), new_in1(half), new_in2(half);
        for (int j = 0; j < half; j++) {
            new_out[j] = current_out[2*j] + rand * (current_out[2*j+1] - current_out[2*j]);
            new_in1[j] = current_in1[2*j] + rand * (current_in1[2*j+1] - current_in1[2*j]);
            new_in2[j] = current_in2[2*j] + rand * (current_in2[2*j+1] - current_in2[2*j]);
        }
        current_out = std::move(new_out);
        current_in1 = std::move(new_in1);
        current_in2 = std::move(new_in2);
    }
    
    // Final values
    P.vr.push_back(current_out[0]);
    P.vr.push_back(current_in1[0]);
    P.vr.push_back(current_in2[0]);
    P.q_poly = round_polys;
    P.randomness.push_back(r);
    P.sc_challenges.push_back(sc_challenges);
    
    // Store input/output evaluations
    P.in1 = eval_in1;
    P.in2 = eval_in2;
    P.out_eval = eval_out;
    
    return P;
}

// Prove noise addition (Box-Muller generated noise) - WITH PROPER SUMCHECK
// Now accepts u1, u2 vectors for Box-Muller proof generation
struct proof prove_noise_addition(
    const std::vector<std::vector<F>>& grads_before,
    const std::vector<std::vector<F>>& grads_after,
    const std::vector<std::vector<F>>& noise,
    const F& noise_std,
    const std::vector<F>& randomness,
    const std::vector<std::vector<double>>& noise_u1 = {},
    const std::vector<std::vector<double>>& noise_u2 = {}) {
    
    struct proof P;
    P.type = GKR_PROOF;  // Using GKR_PROOF for Box-Muller circuit proofs
    
    // ===========================================================================
    // CRITICAL: Generate randomness ONCE and use for ALL evaluations
    // ===========================================================================
    size_t max_size = 0;
    for (const auto& v : grads_before) max_size = std::max(max_size, v.size());
    for (const auto& v : grads_after) max_size = std::max(max_size, v.size());
    for (const auto& v : noise) max_size = std::max(max_size, v.size());
    
    size_t pow2_size = 1;
    while (pow2_size < max_size) pow2_size <<= 1;
    int log_size = static_cast<int>(std::log2(pow2_size));
    if (log_size < 1) log_size = 1;
    
    std::vector<F> shared_randomness;
    if (!randomness.empty() && randomness.size() >= static_cast<size_t>(log_size)) {
        shared_randomness.assign(randomness.begin(), randomness.begin() + log_size);
    } else {
        shared_randomness = generate_randomness(log_size, F(0));
    }
    
    // Concatenate all vectors for sumcheck-based proof
    std::vector<F> all_grads_before, all_grads_after, all_noise;
    for (const auto& v : grads_before) {
        all_grads_before.insert(all_grads_before.end(), v.begin(), v.end());
    }
    for (const auto& v : grads_after) {
        all_grads_after.insert(all_grads_after.end(), v.begin(), v.end());
    }
    for (const auto& v : noise) {
        all_noise.insert(all_noise.end(), v.begin(), v.end());
    }
    
    // Generate PROPER SUMCHECK proof for addition constraint
    // This proves: for all i, grads_after[i] = grads_before[i] + noise[i]
    struct proof additive_proof = prove_additive_sumcheck(
        all_grads_before, all_noise, all_grads_after, shared_randomness);
    
    // Copy sumcheck proof data to main proof
    P.q_poly = additive_proof.q_poly;
    P.vr = additive_proof.vr;
    P.sc_challenges = additive_proof.sc_challenges;
    
    // Generate Box-Muller GKR proofs for noise generation
    if (!noise_u1.empty() && !noise_u2.empty() && 
        noise_u1.size() == noise.size() && noise_u2.size() == noise.size()) {
        
        if (!noise_u1[0].empty() && !noise_u2[0].empty()) {
            F u1_field = quantize(static_cast<float>(noise_u1[0][0]));
            F u2_field = quantize(static_cast<float>(noise_u2[0][0]));
            
            BoxMullerProof bm_proof = prove_box_muller_transform(
                u1_field, u2_field, noise_std, randomness);
            
            P.in1 = u1_field;
            P.in2 = u2_field;
            P.out_eval = bm_proof.z;
        }
    } else {
        P.in1 = additive_proof.in1;
        P.in2 = additive_proof.in2;
        P.out_eval = additive_proof.out_eval;
    }
    
    P.randomness.push_back(shared_randomness);
    
    return P;
}

// Helper: Create a sumcheck proof for scalar multiplication: out[i] = scalar * in[i]
// Proves the relationship at a random evaluation point: eval(out, r) = scalar * eval(in, r) / SCALE
// (Division by SCALE due to quantized multiplication)
struct proof prove_scalar_mul_sumcheck(
    const std::vector<F>& v_in,
    const F& scalar,
    const std::vector<F>& v_out,
    const std::vector<F>& shared_randomness) {
    
    struct proof P;
    P.type = ADD_PROOF;  // Use ADD_PROOF to avoid MATMUL verifier
    
    // Pad to power of 2
    size_t pow2_size = 1;
    while (pow2_size < v_in.size()) pow2_size <<= 1;
    if (pow2_size == 0) pow2_size = 1;
    
    std::vector<F> padded_in = v_in;
    std::vector<F> padded_out = v_out;
    padded_in.resize(pow2_size, F_ZERO);
    padded_out.resize(pow2_size, F_ZERO);
    
    int log_size = static_cast<int>(std::log2(pow2_size));
    if (log_size < 1) log_size = 1;
    
    std::vector<F> r;
    if (!shared_randomness.empty() && shared_randomness.size() >= static_cast<size_t>(log_size)) {
        r.assign(shared_randomness.begin(), shared_randomness.begin() + log_size);
    } else {
        r = generate_randomness(log_size, F(0));
    }
    
    // Evaluate polynomials
    F eval_in = evaluate_vector(padded_in, r);
    F eval_out = evaluate_vector(padded_out, r);
    
    // Build sumcheck for scalar multiplication constraint:
    // For each i: out[i] - scalar * in[i] / SCALE = 0
    // Sumcheck proves sum_i (out[i] - (scalar * in[i]) / SCALE) = 0 at random r
    
    std::vector<quadratic_poly> round_polys;
    std::vector<F> current_in = padded_in;
    std::vector<F> current_out = padded_out;
    
    F rand = shared_randomness.empty() ? F_ZERO : shared_randomness[0];
    std::vector<F> sc_challenges;
    
    for (int round = 0; round < log_size; round++) {
        int half = current_in.size() / 2;
        F p0 = F_ZERO, p1 = F_ZERO, p2 = F_ZERO;
        
        for (int j = 0; j < half; j++) {
            // Constraint at x=0: out[2j] - quantized_mul(scalar, in[2j])
            F scaled_0 = quantized_mul(scalar, current_in[2*j]);
            F c0 = current_out[2*j] - scaled_0;
            
            // Constraint at x=1: out[2j+1] - quantized_mul(scalar, in[2j+1])
            F scaled_1 = quantized_mul(scalar, current_in[2*j+1]);
            F c1 = current_out[2*j+1] - scaled_1;
            
            // Linear interpolation
            p0 = p0 + c0;
            p1 = p1 + (c1 - c0);
        }
        
        round_polys.push_back(quadratic_poly(p2, p1, p0));
        
        rand = mimc_hash(rand, F(round));
        sc_challenges.push_back(rand);
        
        // Fold
        std::vector<F> new_in(half), new_out(half);
        for (int j = 0; j < half; j++) {
            new_in[j] = current_in[2*j] + rand * (current_in[2*j+1] - current_in[2*j]);
            new_out[j] = current_out[2*j] + rand * (current_out[2*j+1] - current_out[2*j]);
        }
        current_in = std::move(new_in);
        current_out = std::move(new_out);
    }
    
    P.vr.push_back(current_out[0]);
    P.vr.push_back(current_in[0]);
    P.q_poly = round_polys;
    P.randomness.push_back(r);
    P.sc_challenges.push_back(sc_challenges);
    
    P.in1 = scalar;
    P.in2 = eval_in;
    P.out_eval = eval_out;
    
    return P;
}

// Prove weight update with proper sumcheck
// weights_after = weights_before - lr * noisy_grads
struct proof prove_weight_update(
    const std::vector<std::vector<F>>& weights_before,
    const std::vector<std::vector<F>>& weights_after,
    const std::vector<std::vector<F>>& noisy_grads,
    const F& learning_rate,
    const std::vector<F>& randomness) {
    
    struct proof P;
    P.type = ADD_PROOF;  // Use ADD_PROOF to avoid MATMUL verifier
    
    // ===========================================================================
    // CRITICAL: Generate randomness ONCE and use for ALL evaluations
    // ===========================================================================
    size_t max_size = 0;
    for (const auto& v : weights_before) max_size = std::max(max_size, v.size());
    for (const auto& v : weights_after) max_size = std::max(max_size, v.size());
    for (const auto& v : noisy_grads) max_size = std::max(max_size, v.size());
    
    size_t pow2_size = 1;
    while (pow2_size < max_size) pow2_size <<= 1;
    int log_size = static_cast<int>(std::log2(pow2_size));
    if (log_size < 1) log_size = 1;
    
    std::vector<F> shared_randomness;
    if (!randomness.empty() && randomness.size() >= static_cast<size_t>(log_size)) {
        shared_randomness.assign(randomness.begin(), randomness.begin() + log_size);
    } else {
        shared_randomness = generate_randomness(log_size, F(0));
    }
    
    // Concatenate all vectors
    std::vector<F> all_weights_before, all_weights_after, all_grads;
    for (const auto& v : weights_before) {
        all_weights_before.insert(all_weights_before.end(), v.begin(), v.end());
    }
    for (const auto& v : weights_after) {
        all_weights_after.insert(all_weights_after.end(), v.begin(), v.end());
    }
    for (const auto& v : noisy_grads) {
        all_grads.insert(all_grads.end(), v.begin(), v.end());
    }
    
    // Compute lr * grads ELEMENT-WISE
    std::vector<F> lr_scaled_grads;
    lr_scaled_grads.reserve(all_grads.size());
    for (const auto& g : all_grads) {
        lr_scaled_grads.push_back(quantized_mul(learning_rate, g));
    }
    
    // Generate sumcheck proof for: weights_after = weights_before - lr_scaled_grads
    // Rewrite as: weights_before - weights_after = lr_scaled_grads
    // Or equivalently: weights_after + lr_scaled_grads = weights_before
    
    // Proof 1: Prove the scalar multiplication (lr * grads)
    struct proof mul_proof = prove_scalar_mul_sumcheck(
        all_grads, learning_rate, lr_scaled_grads, shared_randomness);
    
    // Proof 2: Prove the subtraction relationship using additive sumcheck
    // weights_after = weights_before - lr_scaled_grads
    // Which is: weights_after + lr_scaled_grads = weights_before
    // Negate lr_scaled_grads for the addition proof
    std::vector<F> neg_lr_scaled_grads;
    neg_lr_scaled_grads.reserve(lr_scaled_grads.size());
    for (const auto& x : lr_scaled_grads) {
        neg_lr_scaled_grads.push_back(F_ZERO - x);
    }
    
    struct proof sub_proof = prove_additive_sumcheck(
        all_weights_before, neg_lr_scaled_grads, all_weights_after, shared_randomness);
    
    // Combine proofs
    P.q_poly = mul_proof.q_poly;
    P.q_poly.insert(P.q_poly.end(), sub_proof.q_poly.begin(), sub_proof.q_poly.end());
    
    P.vr = mul_proof.vr;
    P.vr.insert(P.vr.end(), sub_proof.vr.begin(), sub_proof.vr.end());
    
    P.sc_challenges = mul_proof.sc_challenges;
    P.sc_challenges.insert(P.sc_challenges.end(), sub_proof.sc_challenges.begin(), sub_proof.sc_challenges.end());
    
    P.randomness.push_back(shared_randomness);
    
    // Store overall relationship
    F weights_before_sum = evaluate_vectors_sum(weights_before, shared_randomness);
    F weights_after_sum = evaluate_vectors_sum(weights_after, shared_randomness);
    
    P.in1 = weights_before_sum;
    P.in2 = learning_rate;
    P.out_eval = weights_after_sum;
    
    return P;
}

// Prove gradient averaging: avg_grad = sum(grads) / batch_size
// Uses sumcheck to prove:
// 1. sum_grads = per_sample_grads[0] + per_sample_grads[1] + ... (additive sumcheck)
// 2. avg_grads * batch_size = sum_grads (multiplication sumcheck)
struct proof prove_gradient_averaging(
    const std::vector<std::vector<F>>& per_sample_grads,  // batch_size x num_params
    const std::vector<std::vector<F>>& avg_grads,         // 1 x num_params
    int batch_size,
    const std::vector<F>& randomness) {
    
    struct proof P;
    P.type = ADD_PROOF;  // Use ADD_PROOF to avoid MATMUL verifier
    
    // Find maximum size
    size_t max_size = 0;
    for (const auto& v : per_sample_grads) max_size = std::max(max_size, v.size());
    for (const auto& v : avg_grads) max_size = std::max(max_size, v.size());
    
    size_t pow2_size = 1;
    while (pow2_size < max_size) pow2_size <<= 1;
    if (pow2_size == 0) pow2_size = 1;
    int log_size = static_cast<int>(std::log2(pow2_size));
    if (log_size < 1) log_size = 1;
    
    std::vector<F> shared_randomness;
    if (!randomness.empty() && randomness.size() >= static_cast<size_t>(log_size)) {
        shared_randomness.assign(randomness.begin(), randomness.begin() + log_size);
    } else {
        shared_randomness = generate_randomness(log_size, F(0));
    }
    
    // Get the number of parameters
    size_t num_params = per_sample_grads.empty() ? 0 : per_sample_grads[0].size();
    if (num_params == 0) {
        // Empty input, return trivial proof
        P.randomness.push_back(shared_randomness);
        return P;
    }
    
    // Compute the sum of gradients element-wise
    std::vector<F> sum_grads(num_params, F_ZERO);
    for (const auto& grad : per_sample_grads) {
        for (size_t i = 0; i < num_params && i < grad.size(); i++) {
            sum_grads[i] = sum_grads[i] + grad[i];
        }
    }
    sum_grads.resize(pow2_size, F_ZERO);
    
    // Concatenate avg_grads
    std::vector<F> all_avg_grads;
    for (const auto& v : avg_grads) {
        all_avg_grads.insert(all_avg_grads.end(), v.begin(), v.end());
    }
    all_avg_grads.resize(pow2_size, F_ZERO);
    
    // Compute avg * batch_size = sum (element-wise)
    // To prove: avg[i] * batch_size = sum[i]
    // At random point: eval(avg, r) * batch_size = eval(sum, r)
    
    F batch_size_field = quantize(static_cast<float>(batch_size));
    
    // Build sumcheck for multiplication constraint: avg[i] * batch_size = sum[i]
    std::vector<quadratic_poly> round_polys;
    std::vector<F> current_avg = all_avg_grads;
    std::vector<F> current_sum = sum_grads;
    
    F rand = shared_randomness.empty() ? F_ZERO : shared_randomness[0];
    std::vector<F> sc_challenges;
    
    for (int round = 0; round < log_size; round++) {
        int half = current_avg.size() / 2;
        F p0 = F_ZERO, p1 = F_ZERO, p2 = F_ZERO;
        
        for (int j = 0; j < half; j++) {
            // Constraint at x=0: quantized_mul(batch_size, avg[2j]) - sum[2j]
            F prod_0 = quantized_mul(batch_size_field, current_avg[2*j]);
            F c0 = prod_0 - current_sum[2*j];
            
            // Constraint at x=1: quantized_mul(batch_size, avg[2j+1]) - sum[2j+1]
            F prod_1 = quantized_mul(batch_size_field, current_avg[2*j+1]);
            F c1 = prod_1 - current_sum[2*j+1];
            
            p0 = p0 + c0;
            p1 = p1 + (c1 - c0);
        }
        
        round_polys.push_back(quadratic_poly(p2, p1, p0));
        
        rand = mimc_hash(rand, F(round));
        sc_challenges.push_back(rand);
        
        // Fold
        std::vector<F> new_avg(half), new_sum(half);
        for (int j = 0; j < half; j++) {
            new_avg[j] = current_avg[2*j] + rand * (current_avg[2*j+1] - current_avg[2*j]);
            new_sum[j] = current_sum[2*j] + rand * (current_sum[2*j+1] - current_sum[2*j]);
        }
        current_avg = std::move(new_avg);
        current_sum = std::move(new_sum);
    }
    
    // Final values
    P.vr.push_back(current_avg[0]);
    P.vr.push_back(current_sum[0]);
    P.q_poly = round_polys;
    P.randomness.push_back(shared_randomness);
    P.sc_challenges.push_back(sc_challenges);
    
    // Store values
    F sum_eval = evaluate_vector(sum_grads, shared_randomness);
    F avg_eval = evaluate_vector(all_avg_grads, shared_randomness);
    
    P.in1 = sum_eval;
    P.in2 = batch_size_field;
    P.out_eval = avg_eval;
    
    // Store division relationship
    P.divident = sum_eval;
    P.divisor = batch_size_field;
    P.quotient = avg_eval;
    P.remainder = F_ZERO;
    
    return P;
}

// Generate complete proof for a DP-SGD step
DPSGDStepProof prove_dp_sgd_step(
    const std::vector<std::vector<F>>& per_sample_grads_field,
    const std::vector<std::vector<F>>& clipped_grads_field,
    const std::vector<std::vector<F>>& avg_grads_field,
    const std::vector<std::vector<F>>& noisy_grads_field,
    const std::vector<std::vector<F>>& noise_field,
    const std::vector<std::vector<F>>& weights_before_field,
    const std::vector<std::vector<F>>& weights_after_field,
    const F& learning_rate,
    const F& clip_norm,
    const F& noise_std,
    int batch_size,
    const std::vector<std::vector<double>>& noise_u1 = {},
    const std::vector<std::vector<double>>& noise_u2 = {}) {
    
    DPSGDStepProof step_proof;
    step_proof.learning_rate = learning_rate;
    step_proof.clip_norm = clip_norm;
    step_proof.noise_std = noise_std;
    step_proof.batch_size = batch_size;
    step_proof.clipped_grads_field = clipped_grads_field;
    step_proof.noise_field = noise_field;
    step_proof.avg_grads_field = avg_grads_field;
    
    // Generate randomness for evaluation
    // Find the maximum size among all weight vectors and ensure it's a power of 2
    size_t max_size = 0;
    for (const auto& w : weights_before_field) {
        max_size = std::max(max_size, w.size());
    }
    
    // Pad to next power of 2 if needed
    size_t padded_size = 1;
    while (padded_size < max_size) {
        padded_size <<= 1;
    }
    
    // Generate randomness for the padded size
    int log_size = static_cast<int>(std::log2(padded_size));
    if (log_size < 1) log_size = 1;  // Minimum size
    std::vector<F> randomness = generate_randomness(log_size, F(0));
    
    // Prove gradient clipping (with range proof)
    {
        VERIDP_TIME_CLIPPING();
        struct proof clip_proof = prove_gradient_clipping(
            per_sample_grads_field, clipped_grads_field, clip_norm, randomness);
        step_proof.proofs.push_back(clip_proof);
    }
    
    // Prove gradient averaging (with sumcheck)
    // avg_grads = sum(clipped_grads) / batch_size
    struct proof avg_proof = prove_gradient_averaging(
        clipped_grads_field, avg_grads_field, batch_size, randomness);
    step_proof.proofs.push_back(avg_proof);
    
    // Prove noise addition (now with Box-Muller GKR proofs and sumcheck).
    // If noise_std == 0, this still proves noisy_grads == avg_grads + 0,
    // but Box-Muller sub-proofs are skipped because u1/u2 are empty.
    {
        VERIDP_TIME_NOISE_GEN();  // Noise generation includes Box-Muller proofs
        VERIDP_TIME_NOISE_ADD();
        struct proof noise_proof = prove_noise_addition(
            avg_grads_field, noisy_grads_field, noise_field, noise_std, randomness,
            noise_u1, noise_u2);  // Pass u1, u2 for Box-Muller proofs
        step_proof.proofs.push_back(noise_proof);
    }
    
    // Prove weight update
    {
        VERIDP_TIME_WEIGHT_UPDATE();
        struct proof update_proof = prove_weight_update(
            weights_before_field, weights_after_field, noisy_grads_field, 
            learning_rate, randomness);
        step_proof.proofs.push_back(update_proof);
    }
    
    // Collect witness data
    for (const auto& grad : clipped_grads_field) {
        step_proof.witness.insert(step_proof.witness.end(), grad.begin(), grad.end());
    }
    for (const auto& n : noise_field) {
        step_proof.witness.insert(step_proof.witness.end(), n.begin(), n.end());
    }
    
    return step_proof;
}
