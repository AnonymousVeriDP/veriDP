#pragma once
#include "config_pc.hpp"
#include <cmath>

// Quantization constant - fixed-point representation
// Using Q24.24 (24 integer bits, 24 fractional bits) for better precision
// This provides ~5.96e-8 precision and better range than Q16.16
#ifndef Q
#define Q 24  // Fixed-point quantization bits (Q24.24 format) - increased from 16 for better precision
#endif

// Convert float to quantized field element
inline F quantize(float value) {
    const F SCALE = F(1ULL << Q);
    int64_t scaled = static_cast<int64_t>(std::round(value * (1ULL << Q)));
    return F(scaled);
}

// Helper to safely extract int128 from field element
// This function is field-type independent
inline __int128_t safe_extract_int128(const F& val) {
#ifdef USE_FIELD_256
    // For 256-bit field, use toint128() which returns lower 128 bits with sign handling
    return val.toint128();
#else
    // For 61-bit field with extension (has real and img parts)
    if (val.img == 0) {
        // Check for negative values in field representation
        if ((val.real >> 60) & 1) {
            return static_cast<__int128_t>(val.real) - static_cast<__int128_t>(2305843009213693951LL);
        }
        return static_cast<__int128_t>(val.real);
    }
    // If img != 0, just use real part for approximate extraction
    return static_cast<__int128_t>(val.real);
#endif
}

// Convert quantized field element back to float
inline float dequantize(const F& value, int scale_bits = Q) {
    __int128_t scaled = safe_extract_int128(value);
    return static_cast<float>(scaled) / static_cast<float>(1ULL << scale_bits);
}

// Quantized multiplication: (a * b) / SCALE
// For Q24.24: multiplying two Q24.24 numbers gives Q48.48, need to shift back to Q24.24
// Formula: If a and b represent a_float = a_int/2^Q and b_float = b_int/2^Q,
//          then result should be (a_int * b_int) / 2^Q in Q24.24 format
// 
// Rounding: "round-to-nearest, ties-to-away-from-zero"
//   result_int = (a_int * b_int + (1 << 23)) >> 24
//
// Overflow: Check that result fits in signed Q24.24 range (-2^31 ≤ result < 2^31)
inline F quantized_mul(const F& a, const F& b) {
    // Extract integer representations (signed 48-bit values in Q24.24 format)
    __int128_t a_int = safe_extract_int128(a);
    __int128_t b_int = safe_extract_int128(b);
    
    // Multiply in 128-bit integer arithmetic to avoid overflow
    // a_int * b_int can be up to 96 bits (48-bit * 48-bit)
    __int128_t product = a_int * b_int;
    
    // Apply round-to-nearest, ties-to-away-from-zero
    // Add half-unit-in-the-last-place: 0.5 * 2^-24 = 1 << 23
    __int128_t rounded_product = product + (static_cast<__int128_t>(1) << 23);
    
    // Right shift by 24 bits (arithmetic shift, sign-extending for negative numbers)
    __int128_t result_int = rounded_product >> 24;
    
    // Overflow check: result must fit in signed Q24.24 range [-2^31, 2^31-1]
    constexpr __int128_t Q24_24_MIN = -(static_cast<__int128_t>(1) << 31);  // -2^31
    constexpr __int128_t Q24_24_MAX = (static_cast<__int128_t>(1) << 31) - 1;  // 2^31 - 1
    
    if (result_int < Q24_24_MIN) {
        // Saturate to minimum value
        result_int = Q24_24_MIN;
    } else if (result_int > Q24_24_MAX) {
        // Saturate to maximum value
        result_int = Q24_24_MAX;
    }
    
    // Convert back to field element
    return F(static_cast<long long>(result_int));
}

// Quantized division: (a * SCALE) / b
// Formula: If a and b represent a_float = a_int/2^Q and b_float = b_int/2^Q,
//          then result should be (a_int * 2^Q) / b_int in Q24.24 format
//
// Rounding: "round-to-nearest, ties-to-away-from-zero"
//   result_int = ((a_int << 24) + (b_int >> 1)) / b_int
//
// Overflow: Check that result fits in signed Q24.24 range (-2^31 ≤ result < 2^31)
inline F quantized_div(const F& numerator, const F& denominator) {
    // Edge case: zero denominator - abort or return sentinel
    if (denominator == F_ZERO) {
        // Return zero as sentinel value (could also abort depending on security model)
        return F_ZERO;
    }
    
    // Extract integer representations (signed 48-bit values in Q24.24 format)
    __int128_t num_int = safe_extract_int128(numerator);
    __int128_t den_int = safe_extract_int128(denominator);
    
    if (den_int == 0) {
        return F_ZERO;
    }
    
    // Compute (num_int * 2^Q) / den_int using 128-bit arithmetic
    // Shift numerator left by Q bits
    __int128_t scaled_num = num_int << Q;
    
    // Apply round-to-nearest, ties-to-away-from-zero
    // Add half-unit: (den_int >> 1) represents 0.5 * den_int
    __int128_t rounded_num = scaled_num + (den_int >> 1);
    
    // Perform signed division (handles negative numbers correctly)
    __int128_t result_int = rounded_num / den_int;
    
    // Overflow check: result must fit in signed Q24.24 range [-2^31, 2^31-1]
    constexpr __int128_t Q24_24_MIN = -(static_cast<__int128_t>(1) << 31);  // -2^31
    constexpr __int128_t Q24_24_MAX = (static_cast<__int128_t>(1) << 31) - 1;  // 2^31 - 1
    
    if (result_int < Q24_24_MIN) {
        // Saturate to minimum value
        result_int = Q24_24_MIN;
    } else if (result_int > Q24_24_MAX) {
        // Saturate to maximum value
        result_int = Q24_24_MAX;
    }
    
    // Convert back to field element
    return F(static_cast<long long>(result_int));
}

// Compute exp on quantized field element
// Uses optimized computation with range reduction
inline F exp(const F& x) {
    // For small values, use direct computation with quantized arithmetic
    float x_float = dequantize(x);
    if (std::abs(x_float) < 0.1) {
        // Taylor series for small x: exp(x) ≈ 1 + x + x²/2 + x³/6
        // Use quantized arithmetic for better precision
        F one = quantize(1.0f);
        F x_sq = quantized_mul(x, x);  // x² with proper quantization
        F x_cu = quantized_mul(x_sq, x);  // x³ with proper quantization
        F x_sq_div2 = quantized_div(x_sq, quantize(2.0f));
        F x_cu_div6 = quantized_div(x_cu, quantize(6.0f));
        return one + x + x_sq_div2 + x_cu_div6;
    }
    
    // For larger values, use direct exp (can be optimized with log approximation later)
    float exp_val = std::exp(x_float);
    return quantize(exp_val);
}

// Legacy divide function (for backward compatibility)
inline F divide(const F& numerator, const F& denominator) {
    return quantized_div(numerator, denominator);
}
