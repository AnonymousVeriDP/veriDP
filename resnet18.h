#pragma once
#include <torch/torch.h>
#include <algorithm>

struct BasicBlockImpl : torch::nn::Module {
    static constexpr int expansion = 1;

    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::GroupNorm gn1{nullptr};
    torch::nn::GroupNorm gn2{nullptr};
    torch::nn::Sequential downsample{nullptr};
    int stride;

    BasicBlockImpl(int64_t in_planes, int64_t planes, int stride_ = 1)
        : stride(stride_) {
        conv1 = register_module(
            "conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, planes, 3)
                                  .stride(stride)
                                  .padding(1)
                                  .bias(false)));
        conv2 = register_module(
            "conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(planes, planes, 3)
                                  .stride(1)
                                  .padding(1)
                                  .bias(false)));

        int64_t g1 = std::min<int64_t>(32, planes);
        int64_t g2 = std::min<int64_t>(32, planes);
        gn1 = register_module("gn1", torch::nn::GroupNorm(torch::nn::GroupNormOptions(g1, planes)));
        gn2 = register_module("gn2", torch::nn::GroupNorm(torch::nn::GroupNormOptions(g2, planes)));

        if (stride != 1 || in_planes != planes) {
            downsample = register_module(
                "downsample",
                torch::nn::Sequential(
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, planes, 1)
                                          .stride(stride)
                                          .bias(false)),
                    torch::nn::GroupNorm(torch::nn::GroupNormOptions(std::min<int64_t>(32, planes), planes))));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        auto identity = x;
        auto out = conv1->forward(x);
        out = gn1->forward(out);
        out = torch::relu(out);
        out = conv2->forward(out);
        out = gn2->forward(out);

        if (downsample) {
            identity = downsample->forward(x);
        }

        out += identity;
        out = torch::relu(out);
        return out;
    }
};

TORCH_MODULE(BasicBlock);

struct ResNet18Impl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::GroupNorm gn1{nullptr};
    torch::nn::Sequential layer1;
    torch::nn::Sequential layer2;
    torch::nn::Sequential layer3;
    torch::nn::Sequential layer4;
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Linear fc{nullptr};

    int64_t in_planes = 64;

    ResNet18Impl(int64_t num_classes = 10, int64_t in_channels = 3) {
        conv1 = register_module(
            "conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 3)
                                  .stride(1)
                                  .padding(1)
                                  .bias(false)));
        gn1 = register_module("gn1", torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, 64)));

        layer1 = register_module("layer1", make_layer(64, 2, 1));
        layer2 = register_module("layer2", make_layer(128, 2, 2));
        layer3 = register_module("layer3", make_layer(256, 2, 2));
        layer4 = register_module("layer4", make_layer(512, 2, 2));

        avgpool = register_module("avgpool", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
        fc = register_module("fc", torch::nn::Linear(512 * BasicBlockImpl::expansion, num_classes));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = conv1->forward(x);
        x = gn1->forward(x);
        x = torch::relu(x);

        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);

        x = avgpool->forward(x);
        x = x.view({x.size(0), -1});
        x = fc->forward(x);
        return x;
    }

    torch::nn::Sequential make_layer(int64_t planes, int blocks, int stride) {
        torch::nn::Sequential layers = torch::nn::Sequential();
        layers->push_back(BasicBlock(in_planes, planes, stride));
        in_planes = planes * BasicBlockImpl::expansion;
        for (int i = 1; i < blocks; i++) {
            layers->push_back(BasicBlock(in_planes, planes, 1));
        }
        return layers;
    }
};

TORCH_MODULE(ResNet18);
