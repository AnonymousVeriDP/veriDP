#pragma once
#include <torch/torch.h>

struct SimpleMLPImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};

    SimpleMLPImpl() {
        fc1 = register_module("fc1", torch::nn::Linear(784, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 10));
    }

    struct ForwardResult {
        torch::Tensor input;
        torch::Tensor z1;
        torch::Tensor a1;
        torch::Tensor z2;
        torch::Tensor output;
    };

    ForwardResult forward_with_intermediates(torch::Tensor x) {
        ForwardResult result;
        result.input = x.view({x.size(0), -1});
        result.z1 = fc1->forward(result.input);
        result.a1 = torch::relu(result.z1);
        result.z2 = fc2->forward(result.a1);
        result.output = result.z2;
        return result;
    }

    torch::Tensor forward(torch::Tensor x) {
        x = x.view({x.size(0), -1});
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    struct BackwardResult {
        torch::Tensor dL_dz2;
        torch::Tensor dL_dW2;
        torch::Tensor dL_db2;
        torch::Tensor dL_da1;
        torch::Tensor dL_dz1;
        torch::Tensor dL_dW1;
        torch::Tensor dL_db1;
    };

    BackwardResult backward_with_intermediates(
        const ForwardResult& fwd,
        const torch::Tensor& target,
        const torch::Tensor& loss) {
        
        BackwardResult result;
        int batch_size = fwd.input.size(0);
        
        torch::Tensor softmax_out = torch::softmax(fwd.z2, 1);
        torch::Tensor one_hot = torch::zeros_like(softmax_out);
        one_hot.scatter_(1, target.unsqueeze(1), 1.0);
        result.dL_dz2 = (softmax_out - one_hot) / batch_size;
        
        result.dL_dW2 = fwd.a1.t().mm(result.dL_dz2);
        result.dL_db2 = result.dL_dz2.sum(0);
        result.dL_da1 = result.dL_dz2.mm(fc2->weight);
        
        torch::Tensor relu_mask = (fwd.z1 > 0).to(torch::kFloat);
        result.dL_dz1 = result.dL_da1 * relu_mask;
        
        result.dL_dW1 = fwd.input.t().mm(result.dL_dz1);
        result.dL_db1 = result.dL_dz1.sum(0);
        
        return result;
    }
};

TORCH_MODULE(SimpleMLP);
