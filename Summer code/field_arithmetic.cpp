#include "field_arithmetic.h"
#include "quantization.h"

// Static member definitions
namespace field_arithmetic {
    std::array<F, COS_SIN_LUT_SIZE> CosSinLUT::cos_table;
    std::array<F, COS_SIN_LUT_SIZE> CosSinLUT::sin_table;
    bool CosSinLUT::initialized = false;
}

