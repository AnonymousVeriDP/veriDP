// Implementation file - include this after quantization.h
// This separates declarations from implementations to avoid circular dependencies

#pragma once

#include "field_arithmetic.h"
#include "quantization.h"

namespace field_arithmetic {

// Implementations of functions declared in field_arithmetic.h

// Natural logarithm with proper Chebyshev coefficients
inline F log_chebyshev(const F& x) {
    float x_float = dequantize(x, Q);
    if (x_float <= 0.0) return F_ZERO;
    
    int shift = 0;
    float x_norm = x_float;
    while (x_norm > 2.0) { x_norm *= 0.5; shift++; }
    while (x_norm < 0.5) { x_norm *= 2.0; shift--; }
    
    float y = 4.0 * x_norm - 3.0;
    F y_field = quantize(y);
    
    // Chebyshev coefficients for log(x) (degree 5)
    ChebyshevApprox<5> log_approx;
    log_approx.coefficients[0] = quantize(-0.693147f);
    log_approx.coefficients[1] = quantize(0.693147f);
    log_approx.coefficients[2] = quantize(0.0f);
    log_approx.coefficients[3] = quantize(-0.115524f);
    log_approx.coefficients[4] = quantize(0.0f);
    log_approx.coefficients[5] = quantize(0.023105f);
    
    F result = log_approx.evaluate(y_field);
    F log2 = quantize(0.69314718056f);
    result = result + quantize(static_cast<float>(shift)) * log2;
    
    return result;
}

// Square root with proper Chebyshev coefficients  
inline F sqrt_chebyshev(const F& x) {
    float x_float = dequantize(x, Q);
    if (x_float < 0.0) return F_ZERO;
    
    int shift = 0;
    float x_norm = x_float;
    while (x_norm > 4.0) { x_norm *= 0.25; shift += 2; }
    while (x_norm < 0.25) { x_norm *= 4.0; shift -= 2; }
    
    float y = (16.0f/15.0f) * x_norm - (17.0f/15.0f);
    F y_field = quantize(y);
    
    // Chebyshev coefficients for sqrt(x) (degree 4)
    ChebyshevApprox<4> sqrt_approx;
    sqrt_approx.coefficients[0] = quantize(1.0f);
    sqrt_approx.coefficients[1] = quantize(0.5f);
    sqrt_approx.coefficients[2] = quantize(-0.125f);
    sqrt_approx.coefficients[3] = quantize(0.0625f);
    sqrt_approx.coefficients[4] = quantize(-0.0390625f);
    
    F result = sqrt_approx.evaluate(y_field);
    
    if (shift > 0) {
        for (int i = 0; i < shift / 2; ++i) {
            result = result * quantize(2.0f);
        }
    } else if (shift < 0) {
        for (int i = 0; i < (-shift) / 2; ++i) {
            result = result / quantize(2.0f);
        }
    }
    
    return result;
}

// Cos/Sin LUT implementations
inline F CosSinLUT::cos(const F& x) {
    if (!initialized) init();
    
    float x_float = dequantize(x, Q);
    x_float = std::fmod(x_float, TWO_PI);
    if (x_float < 0.0) x_float += TWO_PI;
    
    int index = static_cast<int>((x_float / TWO_PI) * COS_SIN_LUT_SIZE);
    if (index >= COS_SIN_LUT_SIZE) index = COS_SIN_LUT_SIZE - 1;
    
    return cos_table[index];
}

inline F CosSinLUT::sin(const F& x) {
    if (!initialized) init();
    
    float x_float = dequantize(x, Q);
    x_float = std::fmod(x_float, TWO_PI);
    if (x_float < 0.0) x_float += TWO_PI;
    
    int index = static_cast<int>((x_float / TWO_PI) * COS_SIN_LUT_SIZE);
    if (index >= COS_SIN_LUT_SIZE) index = COS_SIN_LUT_SIZE - 1;
    
    return sin_table[index];
}

inline F CosSinLUT::cos_interp(const F& x) {
    if (!initialized) init();
    
    float x_float = dequantize(x, Q);
    x_float = std::fmod(x_float, TWO_PI);
    if (x_float < 0.0) x_float += TWO_PI;
    
    float idx_float = (x_float / TWO_PI) * COS_SIN_LUT_SIZE;
    int idx0 = static_cast<int>(idx_float) % COS_SIN_LUT_SIZE;
    int idx1 = (idx0 + 1) % COS_SIN_LUT_SIZE;
    float t = idx_float - std::floor(idx_float);
    
    F cos0 = cos_table[idx0];
    F cos1 = cos_table[idx1];
    float cos0_f = dequantize(cos0, Q);
    float cos1_f = dequantize(cos1, Q);
    float result = cos0_f + t * (cos1_f - cos0_f);
    
    return quantize(result);
}

inline F CosSinLUT::sin_interp(const F& x) {
    if (!initialized) init();
    
    float x_float = dequantize(x, Q);
    x_float = std::fmod(x_float, TWO_PI);
    if (x_float < 0.0) x_float += TWO_PI;
    
    float idx_float = (x_float / TWO_PI) * COS_SIN_LUT_SIZE;
    int idx0 = static_cast<int>(idx_float) % COS_SIN_LUT_SIZE;
    int idx1 = (idx0 + 1) % COS_SIN_LUT_SIZE;
    float t = idx_float - std::floor(idx_float);
    
    F sin0 = sin_table[idx0];
    F sin1 = sin_table[idx1];
    float sin0_f = dequantize(sin0, Q);
    float sin1_f = dequantize(sin1, Q);
    float result = sin0_f + t * (sin1_f - sin0_f);
    
    return quantize(result);
}

inline void CosSinLUT::init() {
    if (initialized) return;
    
    for (int i = 0; i < COS_SIN_LUT_SIZE; ++i) {
        float angle = (TWO_PI * i) / COS_SIN_LUT_SIZE;
        cos_table[i] = quantize(std::cos(angle));
        sin_table[i] = quantize(std::sin(angle));
    }
    
    initialized = true;
}

} // namespace field_arithmetic

