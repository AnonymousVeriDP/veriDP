#pragma once

#include "config_pc.hpp"
#include <vector>
#include <cmath>
#include <array>

// Forward declarations - implementations in quantization.h
// We'll include quantization.h at the end of this file for implementations

// BLS12-381 scalar field modulus: q = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
// For 256-bit prime field operations
namespace field_arithmetic {

// Q16.16 fixed-point representation (16 integer bits, 16 fractional bits)
constexpr int Q16_INT_BITS = 16;
constexpr int Q16_FRAC_BITS = 16;
constexpr int Q16_TOTAL_BITS = 32;
constexpr u64 Q16_SCALE = 1ULL << Q16_FRAC_BITS;  // 65536
// Note: Q16_SCALE_F cannot be constexpr because F (fieldElement) is not a literal type
// Provide as an inline function to avoid multiple definition issues
inline F get_q16_scale_f() {
    static const F scale = F(Q16_SCALE);
    return scale;
}

// BLS12-381 scalar field modulus (256 bits)
// q = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
// For now, we'll use the existing field element but prepare for 256-bit operations
// In production, you'd use a proper 256-bit big integer library

// ============================================================================
// Chebyshev Polynomial Approximations
// ============================================================================

// Chebyshev polynomial evaluation: T_n(x) = cos(n * arccos(x))
// For approximation: f(x) ≈ Σ c_i * T_i(x)
template<int Degree>
struct ChebyshevApprox {
    std::array<F, Degree + 1> coefficients;
    
    // Evaluate Chebyshev polynomial approximation
    F evaluate(const F& x) const {
        // Map x from [a, b] to [-1, 1] for Chebyshev domain
        // For now, assume x is already in appropriate range
        F result = F_ZERO;
        F T_prev = F_ONE;  // T_0(x) = 1
        F T_curr = x;      // T_1(x) = x
        
        result = result + coefficients[0] * T_prev;
        if constexpr (Degree >= 1) {
            result = result + coefficients[1] * T_curr;
        }
        
        // Recurrence: T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)
        for (int i = 2; i <= Degree; ++i) {
            F T_next = (F(2) * x * T_curr) - T_prev;
            result = result + coefficients[i] * T_next;
            T_prev = T_curr;
            T_curr = T_next;
        }
        
        return result;
    }
};

// Natural logarithm approximation using Chebyshev polynomials (degree 5)
// Approximates log(x) for x in [0.5, 2.0]
// Implementation in field_arithmetic_impl.h (after quantization.h is included)
F log_chebyshev(const F& x);

// Square root approximation using Chebyshev polynomials (degree 4)
// Approximates sqrt(x) for x in [0.25, 4.0]
// Implementation in field_arithmetic_impl.h
F sqrt_chebyshev(const F& x);

// ============================================================================
// Piece-wise Lookup Table for cos/sin
// ============================================================================

// LUT size: 256 entries covering [0, 2π)
constexpr int COS_SIN_LUT_SIZE = 256;
constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 2.0f * PI;

// Precomputed cosine and sine lookup tables
class CosSinLUT {
private:
    static std::array<F, COS_SIN_LUT_SIZE> cos_table;
    static std::array<F, COS_SIN_LUT_SIZE> sin_table;
    static bool initialized;
    
public:
    // Initialization - implementation in field_arithmetic_impl.h
    static void init();
    
    // Get cosine value (x in radians, mapped to [0, 2π))
    // Implementation in field_arithmetic_impl.h
    static F cos(const F& x);
    
    // Get sine value (x in radians, mapped to [0, 2π))
    static F sin(const F& x);
    
    // Linear interpolation for better accuracy
    static F cos_interp(const F& x);
    static F sin_interp(const F& x);
};

// Static member definitions are in field_arithmetic.cpp

// ============================================================================
// Q16.16 Fixed-Point Operations
// ============================================================================

// Convert to Q16.16 format
inline F to_q16_16(float value) {
    int64_t scaled = static_cast<int64_t>(std::round(value * Q16_SCALE));
    return F(scaled);
}

// Convert from Q16.16 format
inline float from_q16_16(const F& value) {
    int64_t scaled = value.toint128();
    return static_cast<float>(scaled) / static_cast<float>(Q16_SCALE);
}

// Q16.16 multiplication: (a * b) / SCALE
inline F q16_16_mul(const F& a, const F& b) {
    // For proper Q16.16 multiplication, we'd need 64-bit intermediate
    // For now, use dequantize/quantize approach
    float a_f = from_q16_16(a);
    float b_f = from_q16_16(b);
    return to_q16_16(a_f * b_f);
}

// Q16.16 division: (a * SCALE) / b
inline F q16_16_div(const F& a, const F& b) {
    if (b == F_ZERO) return F_ZERO;
    float a_f = from_q16_16(a);
    float b_f = from_q16_16(b);
    return to_q16_16(a_f / b_f);
}

} // namespace field_arithmetic

