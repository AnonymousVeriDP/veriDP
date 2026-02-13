#pragma once
#include <torch/torch.h>
#include <random>

// Synthetic logistic regression dataset:
// features ~ (i + 1 + noise), labels derived from a centered linear score.
struct SyntheticLogisticDataset : torch::data::datasets::Dataset<SyntheticLogisticDataset> {
    torch::Tensor data_;
    torch::Tensor targets_;

    SyntheticLogisticDataset(size_t size,
                             size_t dim,
                             int noise_range = 10,
                             float noise_increment = 0.001f,
                             uint64_t seed = 0) {
        data_ = torch::empty({static_cast<int64_t>(size), static_cast<int64_t>(dim)}, torch::kFloat32);
        targets_ = torch::empty({static_cast<int64_t>(size), 1}, torch::kFloat32);

        std::mt19937 gen(static_cast<uint32_t>(seed));
        std::uniform_int_distribution<uint32_t> dist(0u, UINT32_MAX);

        auto data_acc = data_.accessor<float, 2>();
        auto target_acc = targets_.accessor<float, 2>();

        auto sample_noise = [&]() -> float {
            if (noise_range <= 0 || noise_increment == 0.0f) return 0.0f;
            uint32_t tmp = dist(gen);
            int span = 2 * noise_range;
            int bucket = static_cast<int>(tmp % static_cast<uint32_t>(span));
            return -noise_increment * static_cast<float>(noise_range)
                   + static_cast<float>(bucket) * noise_increment;
        };

        float center = static_cast<float>(size) / 2.0f;

        for (size_t i = 0; i < size; i++) {
            float base = static_cast<float>(i + 1);
            float sum = 0.0f;
            for (size_t j = 0; j < dim; j++) {
                float val = base + sample_noise();
                data_acc[i][j] = val;
                sum += val;
            }
            float avg = sum / static_cast<float>(dim);
            float logit = avg - center;  // centered decision boundary
            target_acc[i][0] = (logit > 0.0f) ? 1.0f : 0.0f;
        }
    }

    torch::data::Example<> get(size_t index) override {
        return {data_[index], targets_[index]};
    }

    torch::optional<size_t> size() const override {
        return static_cast<size_t>(data_.size(0));
    }
};
