#pragma once
#include <torch/torch.h>

struct CIFARCNNImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};

    CIFARCNNImpl() {
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(3, 32, 3).stride(1).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)));
        fc1 = register_module("fc1", torch::nn::Linear(64 * 8 * 8, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = torch::max_pool2d(x, 2);
        x = torch::relu(conv2->forward(x));
        x = torch::max_pool2d(x, 2);
        x = x.view({x.size(0), -1});
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }
};

TORCH_MODULE(CIFARCNN);
