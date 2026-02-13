#pragma once

// 256-bit BLS12-381 scalar field using mcl library
// This provides correct, high-performance field arithmetic

#ifdef USE_FIELD_256

#include <mcl/bls12_381.hpp>
#include <cstdint>
#include <string>
#include <iostream>
#include <random>
#include <type_traits>

namespace virgo256 {

// Use mcl's Fr (scalar field of BLS12-381)
using mcl::bls12::Fr;

class fieldElement256 {
public:
    Fr value;  // The actual mcl field element
    
    // Constructors
    fieldElement256() : value() {}
    
    fieldElement256(const fieldElement256& other) : value(other.value) {}
    
    // Single constructor for all integer types to avoid ambiguity
    template<typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    fieldElement256(T x) {
        if (x < 0) {
            value = static_cast<int64_t>(x);
        } else {
            value = static_cast<uint64_t>(x);
        }
    }
    
    // Constructor from 4 limbs (for compatibility)
    fieldElement256(uint64_t l0, uint64_t l1, uint64_t l2, uint64_t l3) {
        // Convert from limbs to Fr
        std::string hex;
        char buf[17];
        snprintf(buf, 17, "%016llx", (unsigned long long)l3);
        hex += buf;
        snprintf(buf, 17, "%016llx", (unsigned long long)l2);
        hex += buf;
        snprintf(buf, 17, "%016llx", (unsigned long long)l1);
        hex += buf;
        snprintf(buf, 17, "%016llx", (unsigned long long)l0);
        hex += buf;
        value.setStr(hex, 16);
    }
    
    // Arithmetic operators
    fieldElement256 operator+(const fieldElement256& other) const {
        fieldElement256 ret;
        Fr::add(ret.value, value, other.value);
        return ret;
    }
    
    fieldElement256 operator-(const fieldElement256& other) const {
        fieldElement256 ret;
        Fr::sub(ret.value, value, other.value);
        return ret;
    }
    
    fieldElement256 operator-() const {
        fieldElement256 ret;
        Fr::neg(ret.value, value);
        return ret;
    }
    
    fieldElement256 operator*(const fieldElement256& other) const {
        fieldElement256 ret;
        Fr::mul(ret.value, value, other.value);
        return ret;
    }
    
    fieldElement256 operator/(const fieldElement256& other) const {
        fieldElement256 ret;
        Fr::div(ret.value, value, other.value);
        return ret;
    }
    
    // Comparison operators
    bool operator==(const fieldElement256& other) const {
        return value == other.value;
    }
    
    bool operator!=(const fieldElement256& other) const {
        return value != other.value;
    }
    
    bool operator<(const fieldElement256& other) const {
        // Compare as integers
        return value < other.value;
    }
    
    bool operator>(const fieldElement256& other) const {
        return other < *this;
    }
    
    // Assignment operators
    fieldElement256& operator=(const fieldElement256& other) {
        value = other.value;
        return *this;
    }
    
    fieldElement256& operator+=(const fieldElement256& other) {
        Fr::add(value, value, other.value);
        return *this;
    }
    
    fieldElement256& operator-=(const fieldElement256& other) {
        Fr::sub(value, value, other.value);
        return *this;
    }
    
    fieldElement256& operator*=(const fieldElement256& other) {
        Fr::mul(value, value, other.value);
        return *this;
    }
    
    fieldElement256& operator/=(const fieldElement256& other) {
        Fr::div(value, value, other.value);
        return *this;
    }
    
    // Conversion to bool
    explicit operator bool() const {
        return !value.isZero();
    }
    
    // Field operations
    [[nodiscard]] bool isZero() const {
        return value.isZero();
    }
    
    [[nodiscard]] bool isNegative() const {
        // Check if value > (p-1)/2
        Fr half;
        half.setStr("39f6d3a994cebea4199cec0404d0ec02a9ded2017fff2dff7fffffff80000000", 16);
        return value > half;
    }
    
    [[nodiscard]] fieldElement256 inv() const {
        fieldElement256 ret;
        Fr::inv(ret.value, value);
        return ret;
    }
    
    [[nodiscard]] fieldElement256 sqr() const {
        fieldElement256 ret;
        Fr::sqr(ret.value, value);
        return ret;
    }
    
    // Power function
    [[nodiscard]] static fieldElement256 fastPow(const fieldElement256& base, uint64_t exp) {
        fieldElement256 result = one();
        fieldElement256 b = base;
        while (exp > 0) {
            if (exp & 1) result = result * b;
            b = b * b;
            exp >>= 1;
        }
        return result;
    }
    
    // Convert to int128 (for compatibility with quantization)
    [[nodiscard]] __int128_t toint128() const {
        // Get the value as a string and parse
        std::string str = value.getStr(10);
        
        // Check if negative (value > p/2)
        if (isNegative()) {
            // Compute p - value
            Fr neg;
            Fr::neg(neg, value);
            std::string neg_str = neg.getStr(10);
            // Parse and negate
            __int128_t result = 0;
            for (char c : neg_str) {
                result = result * 10 + (c - '0');
            }
            return -result;
        }
        
        // Parse positive value
        __int128_t result = 0;
        for (char c : str) {
            result = result * 10 + (c - '0');
        }
        return result;
    }
    
    // Get limbs (for compatibility) - convert through string representation
    uint64_t getLimb(int i) const {
        // Get hex string and parse limbs
        std::string hex = value.getStr(16);
        // Pad to 64 chars (256 bits)
        while (hex.length() < 64) hex = "0" + hex;
        // Extract limb i (from right, little-endian)
        int start = 64 - (i + 1) * 16;
        if (start < 0) return 0;
        std::string limb_hex = hex.substr(start, 16);
        return std::stoull(limb_hex, nullptr, 16);
    }
    
    // Get bit at position (for compatibility)
    int getBit(int pos) const {
        int limb_idx = pos / 64;
        int bit_idx = pos % 64;
        uint64_t limb = getLimb(limb_idx);
        return (limb >> bit_idx) & 1;
    }
    
    // Get bit width
    unsigned char getBitWidth() const {
        for (int i = 255; i >= 0; i--) {
            if (getBit(i)) return i + 1;
        }
        return 0;
    }
    
    // Get raw bytes (for hashing) - returns 32 bytes in little-endian limb order
    void getBytes(unsigned char* buf) const {
        for (int i = 0; i < 4; i++) {
            uint64_t limb = getLimb(i);
            for (int j = 0; j < 8; j++) {
                buf[i * 8 + j] = (limb >> (j * 8)) & 0xFF;
            }
        }
    }
    
    // Compatibility: limbs array accessor (read-only via getLimb)
    // Note: This is a workaround for code that accesses .limbs directly
    struct LimbsProxy {
        const fieldElement256* parent;
        uint64_t operator[](int i) const { return parent->getLimb(i); }
    };
    LimbsProxy limbs_view() const { return {this}; }
    
    // Print function
    void print(FILE* f) const {
        fprintf(f, "0x%s\n", value.getStr(16).c_str());
    }
    
    std::string toString() const {
        return value.getStr(16);
    }
    
    // Static methods
    static void init() {
        // Initialize mcl for BLS12-381
        mcl::bn::CurveParam cp = mcl::BLS12_381;
        mcl::bn::initPairing(cp);
    }
    
    static fieldElement256 random() {
        fieldElement256 ret;
        ret.value.setByCSPRNG();
        return ret;
    }
    
    static fieldElement256 zero() {
        fieldElement256 ret;
        ret.value = 0;
        return ret;
    }
    
    static fieldElement256 one() {
        fieldElement256 ret;
        ret.value = 1;
        return ret;
    }
    
    // Root of unity for FFT
    static fieldElement256 getRootOfUnity(int log_order) {
        // BLS12-381 scalar field has 2-adicity of 32
        // Primitive 2^32-th root of unity
        if (log_order > 32) {
            fprintf(stderr, "Warning: Requested root of unity order 2^%d exceeds 2^32\n", log_order);
            log_order = 32;
        }
        
        // Correct primitive 2^32-th root of unity for BLS12-381 Fr
        // Computed as 5^((r-1)/2^32) mod r where r is the scalar field modulus
        // Verified: rou^(2^32) = 1 and rou^(2^31) != 1 (primitive)
        fieldElement256 primitive;
        primitive.value.setStr("212d79e5b416b6f0fd56dc8d168d6c0c4024ff270b3e0941b788f500b912f1f", 16);
        
        // Square (32 - log_order) times to get 2^log_order root
        fieldElement256 rou = primitive;
        for (int i = 0; i < 32 - log_order; i++) {
            Fr::sqr(rou.value, rou.value);
        }
        return rou;
    }
    
    // For negative value checks in quantization
    static fieldElement256 get_mod_minus_one_half() {
        fieldElement256 ret;
        ret.value.setStr("39f6d3a994cebea4199cec0404d0ec02a9ded2017fff2dff7fffffff80000000", 16);
        return ret;
    }
    
    // Static counters (for compatibility)
    static bool initialized;
    static int multCounter, addCounter;
    static bool isCounting;
    static bool isSumchecking;
    static const int __max_order = 32;
};

// Stream output
inline std::ostream& operator<<(std::ostream& out, const fieldElement256& c) {
    out << "0x" << c.value.getStr(16);
    return out;
}

} // namespace virgo256

#else
// If not using 256-bit field, provide empty declarations
namespace virgo256 {
class fieldElement256 {};
}
#endif
