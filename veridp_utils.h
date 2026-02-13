#pragma once

#include <torch/torch.h>
#include "Summer code/config_pc.hpp"
#include "Summer code/utils.hpp"
#include "Summer code/fieldElement.hpp"
#include "Summer code/quantization.h"
#include <vector>
#include <cmath>

inline __int128_t safe_get_comparison_value(const F& val) {
    return safe_extract_int128(val);
}

// Convert PyTorch tensor to field element vector
inline std::vector<F> tensor_to_field(const torch::Tensor& tensor) {
    // Ensure tensor is on CPU and contiguous
    auto cpu_tensor = tensor.cpu().contiguous();
    auto flat = cpu_tensor.flatten();
    
    // Check tensor is float32
    if (flat.dtype() != torch::kFloat32) {
        throw std::runtime_error("tensor_to_field: expected float32 tensor");
    }
    
    auto accessor = flat.accessor<float, 1>();
    int64_t size = flat.size(0);
    
    std::vector<F> result;
    result.reserve(size);
    
    for (int64_t i = 0; i < size; i++) {
        result.push_back(quantize(accessor[i]));  // quantize from quantization.h
    }
    return result;
}

// Convert field element vector to PyTorch tensor
inline torch::Tensor field_to_tensor(const std::vector<F>& field_vec, 
                                     const std::vector<int64_t>& shape) {
    int64_t total_size = 1;
    for (auto s : shape) total_size *= s;
    
    if (static_cast<size_t>(total_size) != field_vec.size()) {
        throw std::runtime_error("Shape size mismatch in field_to_tensor");
    }
    
    auto tensor = torch::zeros(shape, torch::kFloat32);
    auto flat = tensor.flatten();
    auto accessor = flat.accessor<float, 1>();
    
    for (size_t i = 0; i < field_vec.size(); i++) {
        accessor[i] = dequantize(field_vec[i]);
    }
    
    return tensor;
}

// Convert 2D field matrix to PyTorch tensor
inline torch::Tensor field_matrix_to_tensor(const std::vector<std::vector<F>>& matrix) {
    if (matrix.empty()) return torch::zeros({0, 0}, torch::kFloat32);
    
    int64_t rows = static_cast<int64_t>(matrix.size());
    int64_t cols = static_cast<int64_t>(matrix[0].size());
    
    auto tensor = torch::zeros({rows, cols}, torch::kFloat32);
    auto accessor = tensor.accessor<float, 2>();
    
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            accessor[i][j] = dequantize(matrix[i][j]);
        }
    }
    
    return tensor;
}

// Convert PyTorch tensor to 2D field matrix
inline std::vector<std::vector<F>> tensor_to_field_matrix(const torch::Tensor& tensor) {
    if (tensor.dim() != 2) {
        throw std::runtime_error("tensor_to_field_matrix: expected 2D tensor");
    }
    
    int64_t rows = tensor.size(0);
    int64_t cols = tensor.size(1);
    
    std::vector<std::vector<F>> result(rows);
    auto accessor = tensor.accessor<float, 2>();
    
    for (int64_t i = 0; i < rows; i++) {
        result[i].reserve(cols);
        for (int64_t j = 0; j < cols; j++) {
            result[i].push_back(quantize(accessor[i][j]));
        }
    }
    
    return result;
}

// Extract model parameters as field elements
inline std::vector<std::vector<F>> extract_model_params_field(
    const std::vector<torch::Tensor>& params) {
    std::vector<std::vector<F>> result;
    result.reserve(params.size());
    
    for (const auto& param : params) {
        result.push_back(tensor_to_field(param));
    }
    
    return result;
}

// Apply field element parameters back to model
inline void apply_field_params_to_model(
    torch::nn::Module& model,
    const std::vector<std::vector<F>>& field_params) {
    auto params = model.parameters();
    if (params.size() != field_params.size()) {
        throw std::runtime_error("Parameter count mismatch");
    }
    
    size_t idx = 0;
    for (auto& param : params) {
        auto shape = param.sizes().vec();
        auto tensor = field_to_tensor(field_params[idx], shape);
        param.data().copy_(tensor);
        idx++;
    }
}

