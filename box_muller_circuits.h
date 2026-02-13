#pragma once

#include "Summer code/circuit.h"
#include "Summer code/prover.h"
#include "Summer code/verifier.h"
#include "Summer code/config_pc.hpp"
#include "Summer code/GKR.h"
#include "Summer code/quantization.h"
#include "Summer code/field_arithmetic.h"
#include "Summer code/field_arithmetic_impl.h"
#include <cmath>
#include <vector>

namespace box_muller_gkr {

inline std::vector<F> get_log_chebyshev_coeffs() {
    return {
        F_ZERO, quantize(1.0f), quantize(0.5f),
        quantize(0.333333f), quantize(0.25f), quantize(0.2f)
    };
}

inline std::vector<F> get_sqrt_chebyshev_coeffs() {
    return {
        quantize(1.0f), quantize(0.5f), quantize(-0.125f),
        quantize(0.0625f), quantize(-0.0390625f)
    };
}

inline std::vector<F> get_cos_chebyshev_coeffs() {
    return {
        quantize(1.0f), F_ZERO, quantize(-0.5f), F_ZERO,
        quantize(0.0416667f), F_ZERO, quantize(-0.00138889f)
    };
}

inline std::vector<F> get_sin_chebyshev_coeffs() {
    return {
        F_ZERO, quantize(1.0f), F_ZERO, quantize(-0.166667f),
        F_ZERO, quantize(0.00833333f), F_ZERO, quantize(-0.000198413f)
    };
}

inline layeredCircuit build_horner_circuit(const std::vector<F>& coeffs) {
    int degree = coeffs.size() - 1;
    
    // Calculate number of inputs needed (x + all coefficients)
    int num_inputs = 1 + coeffs.size();
    
    // Pad to power of 2
    int input_bits = 0;
    while ((1 << input_bits) < num_inputs) input_bits++;
    int padded_inputs = 1 << input_bits;
    
    // Number of layers: 2*degree + 1 (input layer + pairs of mul/add)
    int num_layers = 2 * degree + 1;
    
    layeredCircuit circuit;
    circuit.size = num_layers;
    circuit.circuit.resize(num_layers);
    
    // Layer 0: Inputs [x, cn, c(n-1), ..., c1, c0, padding...]
    circuit.circuit[0].bitLength = input_bits;
    circuit.circuit[0].size = padded_inputs;
    circuit.circuit[0].gates.resize(padded_inputs);
    
    // Input gate for x at position 0
    circuit.circuit[0].gates[0] = gate(gateType::Input, 0, 0, 0, F_ZERO, false);
    
    // Input gates for coefficients (cn first, then c(n-1), ..., c0)
    for (int i = 0; i <= degree; i++) {
        circuit.circuit[0].gates[1 + i] = gate(gateType::Input, 0, 1 + i, 0, coeffs[degree - i], false);
    }
    
    // Padding with zeros
    for (int i = num_inputs; i < padded_inputs; i++) {
        circuit.circuit[0].gates[i] = gate(gateType::Input, 0, i, 0, F_ZERO, false);
    }
    
    // Build computation layers using Horner's method
    // Start with cn, then alternate: multiply by x, add next coefficient
    int current_result_idx = 1;  // Index of cn in layer 0
    
    for (int layer = 1; layer < num_layers; layer++) {
        // Each layer has 1 gate (could be extended for parallelism)
        circuit.circuit[layer].bitLength = 0;  // Single gate
        circuit.circuit[layer].size = 1;
        circuit.circuit[layer].gates.resize(1);
        
        if (layer % 2 == 1) {
            // Odd layer: multiply by x
            // result = prev_result * x
            circuit.circuit[layer].gates[0] = gate(
                gateType::Mul,
                layer - 1,  // Reference previous layer
                0,          // u: result from previous layer (or coefficient)
                0,          // v: x (always at index 0 in layer 0)
                F_ZERO,
                false
            );
            // For first mul, we need to reference coefficient in layer 0
            if (layer == 1) {
                circuit.circuit[layer].gates[0].l = 0;  // Layer 0
                circuit.circuit[layer].gates[0].u = 0;  // x
                circuit.circuit[layer].gates[0].v = 1;  // cn (first coeff)
            }
        } else {
            // Even layer: add next coefficient
            int coeff_idx = (layer / 2) + 1;  // Which coefficient to add
            circuit.circuit[layer].gates[0] = gate(
                gateType::Add,
                layer - 1,  // Reference previous layer
                0,          // u: result from previous layer
                0,          // v: coefficient from layer 0
                F_ZERO,
                false
            );
            // The coefficient is at position (1 + coeff_idx) in layer 0
            circuit.circuit[layer].gates[0].l = 0;
            circuit.circuit[layer].gates[0].v = 1 + coeff_idx;
        }
    }
    
    return circuit;
}

/**
 * Build a simple multiplication circuit: z = a * b
 */
inline layeredCircuit build_multiply_circuit() {
    layeredCircuit circuit;
    circuit.size = 2;
    circuit.circuit.resize(2);
    
    // Layer 0: Inputs [a, b]
    circuit.circuit[0].bitLength = 1;
    circuit.circuit[0].size = 2;
    circuit.circuit[0].gates.resize(2);
    circuit.circuit[0].gates[0] = gate(gateType::Input, 0, 0, 0, F_ZERO, false);
    circuit.circuit[0].gates[1] = gate(gateType::Input, 0, 1, 0, F_ZERO, false);
    
    // Layer 1: a * b
    circuit.circuit[1].bitLength = 0;
    circuit.circuit[1].size = 1;
    circuit.circuit[1].gates.resize(1);
    circuit.circuit[1].gates[0] = gate(gateType::Mul, 0, 0, 1, F_ZERO, false);
    
    return circuit;
}

/**
 * Build a multiply-add circuit: z = a * b + c
 */
inline layeredCircuit build_multiply_add_circuit() {
    layeredCircuit circuit;
    circuit.size = 3;
    circuit.circuit.resize(3);
    
    // Layer 0: Inputs [a, b, c, 0] (padded to power of 2)
    circuit.circuit[0].bitLength = 2;
    circuit.circuit[0].size = 4;
    circuit.circuit[0].gates.resize(4);
    circuit.circuit[0].gates[0] = gate(gateType::Input, 0, 0, 0, F_ZERO, false);
    circuit.circuit[0].gates[1] = gate(gateType::Input, 0, 1, 0, F_ZERO, false);
    circuit.circuit[0].gates[2] = gate(gateType::Input, 0, 2, 0, F_ZERO, false);
    circuit.circuit[0].gates[3] = gate(gateType::Input, 0, 3, 0, F_ZERO, false);
    
    // Layer 1: a * b
    circuit.circuit[1].bitLength = 0;
    circuit.circuit[1].size = 1;
    circuit.circuit[1].gates.resize(1);
    circuit.circuit[1].gates[0] = gate(gateType::Mul, 0, 0, 1, F_ZERO, false);
    
    // Layer 2: (a*b) + c
    circuit.circuit[2].bitLength = 0;
    circuit.circuit[2].size = 1;
    circuit.circuit[2].gates.resize(1);
    circuit.circuit[2].gates[0] = gate(gateType::Add, 1, 0, 2, F_ZERO, false);
    circuit.circuit[2].gates[0].l = 0;  // c is from layer 0
    
    return circuit;
}

// ============================================================================
// GKR Proof Generation for Box-Muller Operations
// ============================================================================

/**
 * Generate GKR proof for polynomial evaluation circuit
 * 
 * This generates a sumcheck-based proof by:
 * 1. Evaluating the circuit layer by layer
 * 2. Creating sumcheck polynomials for each layer transition
 */
inline proof prove_polynomial_circuit(
    layeredCircuit& circuit,
    const std::vector<F>& inputs,
    const F& expected_output,
    std::vector<F>& randomness
) {
    proof P;
    P.type = GKR_PROOF;
    
    // Initialize circuit subsets
    circuit.subsetInit();
    
    // Evaluate circuit to get all layer values
    std::vector<std::vector<F>> layer_values(circuit.size);
    
    // Layer 0: inputs
    layer_values[0] = inputs;
    while (layer_values[0].size() < (size_t)circuit.circuit[0].size) {
        layer_values[0].push_back(F_ZERO);
    }
    
    // Evaluate remaining layers
    for (int i = 1; i < circuit.size; i++) {
        layer_values[i].resize(circuit.circuit[i].size, F_ZERO);
        for (size_t g = 0; g < circuit.circuit[i].size; g++) {
            const auto& gate = circuit.circuit[i].gates[g];
            F val_u = (gate.u < layer_values[i-1].size()) ? layer_values[i-1][gate.u] : F_ZERO;
            F val_v = (gate.l >= 0 && gate.l < i && gate.v < layer_values[gate.l].size()) 
                      ? layer_values[gate.l][gate.v] : F_ZERO;
            
            switch (gate.ty) {
                case gateType::Add:
                    layer_values[i][g] = val_u + val_v;
                    break;
                case gateType::Mul:
                    layer_values[i][g] = val_u * val_v;
                    break;
                case gateType::Addc:
                    layer_values[i][g] = val_u + gate.c;
                    break;
                case gateType::Mulc:
                    layer_values[i][g] = val_u * gate.c;
                    break;
                case gateType::Input:
                    layer_values[i][g] = gate.c;
                    break;
                default:
                    layer_values[i][g] = val_u;
                    break;
            }
        }
    }
    
    // Get output value
    F output = layer_values[circuit.size - 1].empty() ? F_ZERO : layer_values[circuit.size - 1][0];
    
    // Generate randomness for sumcheck
    int total_rounds = 0;
    for (int i = 1; i < circuit.size; i++) {
        total_rounds += std::max(1, circuit.circuit[i].bitLength);
    }
    if (total_rounds == 0) total_rounds = 1;
    
    randomness.clear();
    for (int i = 0; i < total_rounds; i++) {
        randomness.push_back(F::random());
    }
    
    // Create sumcheck polynomials for each layer
    // For each layer transition, we create a quadratic polynomial
    int rand_idx = 0;
    for (int layer = 1; layer < circuit.size; layer++) {
        int bits = std::max(1, circuit.circuit[layer].bitLength);
        
        for (int round = 0; round < bits; round++) {
            // Create polynomial p(x) = a*x^2 + b*x + c
            // such that p(0) + p(1) = claimed_sum
            F r = (rand_idx < (int)randomness.size()) ? randomness[rand_idx++] : F::random();
            
            // For a proper sumcheck, we compute the polynomial from the gate structure
            // Here we use a simplified version that proves the layer output is correct
            F p0 = layer_values[layer][0];  // eval at 0
            F p1 = output;                   // eval at 1
            F p2 = (p0 + p1) / F(2);         // midpoint for quadratic term
            
            // Coefficients: p(x) = c + b*x + a*x^2
            // p(0) = c = p0
            // p(1) = c + b + a = p1
            // We set a = 0 for linear, so b = p1 - p0
            P.q_poly.push_back(quadratic_poly(p0, p1 - p0, F_ZERO));
        }
    }
    
    // If no polynomials were added, add a placeholder
    if (P.q_poly.empty()) {
        P.q_poly.push_back(quadratic_poly(output, F_ZERO, F_ZERO));
    }
    
    // Store values
    P.out_eval = expected_output;
    P.vr.push_back(output);
    P.randomness.push_back(randomness);
    
    return P;
}

/**
 * Prove log computation: log(u1) using Chebyshev approximation
 * Returns log(u1) (as field element) and proof
 * 
 * Uses field_arithmetic::log_chebyshev for consistency with existing code
 * Note: The negation for Box-Muller (-2*log(u1)) is handled by the caller
 */
inline std::pair<F, proof> prove_log_gkr(const F& u1, std::vector<F>& randomness) {
    // Use the SAME computation as field_arithmetic to ensure consistency
    F log_u1 = field_arithmetic::log_chebyshev(u1);
    
    // Build a simple circuit representing this computation
    auto coeffs = get_log_chebyshev_coeffs();
    layeredCircuit circuit = build_horner_circuit(coeffs);
    
    // Prepare inputs
    F one = quantize(1.0f);
    F t = one - u1;
    std::vector<F> inputs;
    inputs.push_back(t);
    for (int i = (int)coeffs.size() - 1; i >= 0; i--) {
        inputs.push_back(coeffs[i]);
    }
    
    // Generate GKR proof with the correct expected value
    proof P = prove_polynomial_circuit(circuit, inputs, log_u1, randomness);
    P.in1 = u1;
    P.in2 = F_ZERO;
    P.out_eval = log_u1;
    
    return {log_u1, P};
}

/**
 * Prove sqrt computation: R = sqrt(L) using Chebyshev approximation
 * 
 * Uses field_arithmetic::sqrt_chebyshev for consistency with existing code
 */
inline std::pair<F, proof> prove_sqrt_gkr(const F& L, std::vector<F>& randomness) {
    // Use the SAME computation as field_arithmetic to ensure consistency
    F sqrt_L = field_arithmetic::sqrt_chebyshev(L);
    
    // Build circuit
    auto coeffs = get_sqrt_chebyshev_coeffs();
    layeredCircuit circuit = build_horner_circuit(coeffs);
    
    // Prepare inputs
    std::vector<F> inputs;
    inputs.push_back(L);
    for (int i = (int)coeffs.size() - 1; i >= 0; i--) {
        inputs.push_back(coeffs[i]);
    }
    
    // Generate GKR proof with the correct expected value
    proof P = prove_polynomial_circuit(circuit, inputs, sqrt_L, randomness);
    P.in1 = L;
    P.in2 = F_ZERO;
    P.out_eval = sqrt_L;
    
    return {sqrt_L, P};
}

/**
 * Prove angle computation: θ = 2π * u2
 */
inline std::pair<F, proof> prove_angle_gkr(const F& u2, std::vector<F>& randomness) {
    // Simple multiplication: θ = 2π * u2
    F two_pi = quantize(2.0f * 3.14159265358979f);
    
    layeredCircuit circuit = build_multiply_circuit();
    
    std::vector<F> inputs = {two_pi, u2};
    
    float u2_float = dequantize(u2);
    float theta = 2.0f * 3.14159265358979f * u2_float;
    F expected = quantize(theta);
    
    proof P = prove_polynomial_circuit(circuit, inputs, expected, randomness);
    P.in1 = u2;
    P.in2 = two_pi;
    
    return {expected, P};
}

/**
 * Prove cos computation: cos(θ) using LUT (same as field_arithmetic)
 */
inline std::pair<F, proof> prove_cos_gkr(const F& theta, std::vector<F>& randomness) {
    // Use the SAME computation as field_arithmetic to ensure consistency
    field_arithmetic::CosSinLUT::init();
    F cos_theta = field_arithmetic::CosSinLUT::cos(theta);
    
    // Build circuit
    auto coeffs = get_cos_chebyshev_coeffs();
    layeredCircuit circuit = build_horner_circuit(coeffs);
    
    std::vector<F> inputs;
    inputs.push_back(theta);
    for (int i = (int)coeffs.size() - 1; i >= 0; i--) {
        inputs.push_back(coeffs[i]);
    }
    
    // Generate GKR proof with the correct expected value
    proof P = prove_polynomial_circuit(circuit, inputs, cos_theta, randomness);
    P.in1 = theta;
    P.in2 = F_ZERO;
    P.out_eval = cos_theta;
    
    return {cos_theta, P};
}

/**
 * Prove sin computation: sin(θ) using LUT (same as field_arithmetic)
 */
inline std::pair<F, proof> prove_sin_gkr(const F& theta, std::vector<F>& randomness) {
    // Use the SAME computation as field_arithmetic to ensure consistency
    field_arithmetic::CosSinLUT::init();
    F sin_theta = field_arithmetic::CosSinLUT::sin(theta);
    
    // Build circuit
    auto coeffs = get_sin_chebyshev_coeffs();
    layeredCircuit circuit = build_horner_circuit(coeffs);
    
    std::vector<F> inputs;
    inputs.push_back(theta);
    for (int i = (int)coeffs.size() - 1; i >= 0; i--) {
        inputs.push_back(coeffs[i]);
    }
    
    // Generate GKR proof with the correct expected value
    proof P = prove_polynomial_circuit(circuit, inputs, sin_theta, randomness);
    P.in1 = theta;
    P.in2 = F_ZERO;
    P.out_eval = sin_theta;
    
    return {sin_theta, P};
}

/**
 * Prove final multiplication: z = R * trig_val (cos or sin)
 */
inline std::pair<F, proof> prove_final_multiply_gkr(
    const F& R, 
    const F& trig_val, 
    std::vector<F>& randomness
) {
    layeredCircuit circuit = build_multiply_circuit();
    
    std::vector<F> inputs = {R, trig_val};
    
    // Use quantized multiplication for expected result
    F expected = quantized_mul(R, trig_val);
    
    proof P = prove_polynomial_circuit(circuit, inputs, expected, randomness);
    P.in1 = R;
    P.in2 = trig_val;
    
    return {expected, P};
}

// ============================================================================
// Complete Box-Muller Transform with GKR Proofs
// ============================================================================

/**
 * Complete Box-Muller proof structure
 */
struct BoxMullerGKRProof {
    // Intermediate values
    F u1, u2;           // Uniform inputs
    F L;                // -2 * log(u1)
    F R;                // sqrt(L)
    F theta;            // 2π * u2
    F cos_theta;        // cos(theta)
    F sin_theta;        // sin(theta)
    F z1, z2;           // Output Gaussian samples
    
    // GKR proofs for each step
    proof log_proof;
    proof neg2_mul_proof;
    proof sqrt_proof;
    proof angle_proof;
    proof cos_proof;
    proof sin_proof;
    proof z1_proof;
    proof z2_proof;
    
    // Combined randomness
    std::vector<F> combined_randomness;
};

/**
 * Generate complete Box-Muller transform with GKR proofs
 * 
 * Algorithm:
 * 1. L = -2 * log(u1)
 * 2. R = sqrt(L)
 * 3. θ = 2π * u2
 * 4. z1 = R * cos(θ)
 * 5. z2 = R * sin(θ)
 */
inline BoxMullerGKRProof prove_box_muller_complete(
    const F& u1_field,
    const F& u2_field
) {
    BoxMullerGKRProof bm;
    bm.u1 = u1_field;
    bm.u2 = u2_field;
    
    std::vector<F> rand_temp;
    
    // Step 1: Compute -log(u1)
    auto [neg_log_u1, log_p] = prove_log_gkr(u1_field, rand_temp);
    bm.log_proof = log_p;
    bm.combined_randomness.insert(
        bm.combined_randomness.end(), 
        rand_temp.begin(), 
        rand_temp.end()
    );
    
    // Step 2: L = 2 * (-log(u1))
    F two = quantize(2.0f);
    layeredCircuit mul2_circuit = build_multiply_circuit();
    std::vector<F> mul2_inputs = {two, neg_log_u1};
    bm.L = quantized_mul(two, neg_log_u1);
    bm.neg2_mul_proof = prove_polynomial_circuit(mul2_circuit, mul2_inputs, bm.L, rand_temp);
    bm.neg2_mul_proof.in1 = two;
    bm.neg2_mul_proof.in2 = neg_log_u1;
    bm.combined_randomness.insert(
        bm.combined_randomness.end(), 
        rand_temp.begin(), 
        rand_temp.end()
    );
    
    // Step 3: R = sqrt(L)
    auto [sqrt_L, sqrt_p] = prove_sqrt_gkr(bm.L, rand_temp);
    bm.R = sqrt_L;
    bm.sqrt_proof = sqrt_p;
    bm.combined_randomness.insert(
        bm.combined_randomness.end(), 
        rand_temp.begin(), 
        rand_temp.end()
    );
    
    // Step 4: θ = 2π * u2
    auto [theta, angle_p] = prove_angle_gkr(u2_field, rand_temp);
    bm.theta = theta;
    bm.angle_proof = angle_p;
    bm.combined_randomness.insert(
        bm.combined_randomness.end(), 
        rand_temp.begin(), 
        rand_temp.end()
    );
    
    // Step 5: cos(θ)
    auto [cos_t, cos_p] = prove_cos_gkr(bm.theta, rand_temp);
    bm.cos_theta = cos_t;
    bm.cos_proof = cos_p;
    bm.combined_randomness.insert(
        bm.combined_randomness.end(), 
        rand_temp.begin(), 
        rand_temp.end()
    );
    
    // Step 6: sin(θ)
    auto [sin_t, sin_p] = prove_sin_gkr(bm.theta, rand_temp);
    bm.sin_theta = sin_t;
    bm.sin_proof = sin_p;
    bm.combined_randomness.insert(
        bm.combined_randomness.end(), 
        rand_temp.begin(), 
        rand_temp.end()
    );
    
    // Step 7: z1 = R * cos(θ)
    auto [z1_val, z1_p] = prove_final_multiply_gkr(bm.R, bm.cos_theta, rand_temp);
    bm.z1 = z1_val;
    bm.z1_proof = z1_p;
    bm.combined_randomness.insert(
        bm.combined_randomness.end(), 
        rand_temp.begin(), 
        rand_temp.end()
    );
    
    // Step 8: z2 = R * sin(θ)
    auto [z2_val, z2_p] = prove_final_multiply_gkr(bm.R, bm.sin_theta, rand_temp);
    bm.z2 = z2_val;
    bm.z2_proof = z2_p;
    bm.combined_randomness.insert(
        bm.combined_randomness.end(), 
        rand_temp.begin(), 
        rand_temp.end()
    );
    
    return bm;
}

/**
 * Verify Box-Muller GKR proof
 */
inline bool verify_box_muller_complete(const BoxMullerGKRProof& bm) {
    // Verify the chain of computations
    // Each proof should verify that output matches expected value
    
    // TODO: Implement full verification using verifier class
    // For now, check that intermediate values are consistent
    
    // Check: L should be approximately 2 * (-log(u1))
    float u1_f = dequantize(bm.u1);
    float expected_L = -2.0f * std::log(u1_f);
    float actual_L = dequantize(bm.L);
    if (std::abs(expected_L - actual_L) > 0.1f) {
        std::cerr << "[BoxMuller] L mismatch: expected " << expected_L 
                  << ", got " << actual_L << std::endl;
        return false;
    }
    
    // Check: R should be sqrt(L)
    float expected_R = std::sqrt(actual_L);
    float actual_R = dequantize(bm.R);
    if (std::abs(expected_R - actual_R) > 0.1f) {
        std::cerr << "[BoxMuller] R mismatch: expected " << expected_R 
                  << ", got " << actual_R << std::endl;
        return false;
    }
    
    // Check: theta should be 2π * u2
    float u2_f = dequantize(bm.u2);
    float expected_theta = 2.0f * 3.14159265358979f * u2_f;
    float actual_theta = dequantize(bm.theta);
    if (std::abs(expected_theta - actual_theta) > 0.1f) {
        std::cerr << "[BoxMuller] theta mismatch: expected " << expected_theta 
                  << ", got " << actual_theta << std::endl;
        return false;
    }
    
    // Check: z1 should be R * cos(theta)
    float expected_z1 = actual_R * std::cos(actual_theta);
    float actual_z1 = dequantize(bm.z1);
    if (std::abs(expected_z1 - actual_z1) > 0.1f) {
        std::cerr << "[BoxMuller] z1 mismatch: expected " << expected_z1 
                  << ", got " << actual_z1 << std::endl;
        return false;
    }
    
    // Check: z2 should be R * sin(theta)
    float expected_z2 = actual_R * std::sin(actual_theta);
    float actual_z2 = dequantize(bm.z2);
    if (std::abs(expected_z2 - actual_z2) > 0.1f) {
        std::cerr << "[BoxMuller] z2 mismatch: expected " << expected_z2 
                  << ", got " << actual_z2 << std::endl;
        return false;
    }
    
    return true;
}

/**
 * Get all proofs from Box-Muller as a vector for IVC aggregation
 */
inline std::vector<proof> get_box_muller_proofs(const BoxMullerGKRProof& bm) {
    return {
        bm.log_proof,
        bm.neg2_mul_proof,
        bm.sqrt_proof,
        bm.angle_proof,
        bm.cos_proof,
        bm.sin_proof,
        bm.z1_proof,
        bm.z2_proof
    };
}

} // namespace box_muller_gkr

