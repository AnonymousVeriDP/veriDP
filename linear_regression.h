#pragma once
#include <torch/torch.h>

// Simple linear regression model: y = Wx + b
struct LinearRegressionImpl : torch::nn::Module {
    torch::nn::Linear fc{nullptr};

    explicit LinearRegressionImpl(int64_t input_dim) {
        fc = register_module("fc", torch::nn::Linear(input_dim, 1));
    }

    torch::Tensor forward(torch::Tensor x) {
        return fc->forward(x);
    }
};

TORCH_MODULE(LinearRegression);
