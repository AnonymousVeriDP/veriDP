#pragma once
#include <torch/torch.h>

// Simple logistic regression model: y = sigmoid(Wx + b)
// Forward returns logits for use with BCEWithLogits loss.
struct LogisticRegressionImpl : torch::nn::Module {
    torch::nn::Linear fc{nullptr};

    explicit LogisticRegressionImpl(int64_t input_dim) {
        fc = register_module("fc", torch::nn::Linear(input_dim, 1));
    }

    torch::Tensor forward(torch::Tensor x) {
        return fc->forward(x);
    }
};

TORCH_MODULE(LogisticRegression);
