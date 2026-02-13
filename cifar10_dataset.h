#pragma once
#include <torch/torch.h>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace cifar10_dataset {

class CIFAR10 : public torch::data::datasets::Dataset<CIFAR10> {
public:
    enum class Mode { kTrain, kTest };

    explicit CIFAR10(const std::string& root, Mode mode = Mode::kTrain)
        : root_(root), mode_(mode) {
        if (mode_ == Mode::kTrain) {
            for (int i = 1; i <= 5; i++) {
                data_files_.push_back(root_ + "/cifar-10-batches-bin/data_batch_" + std::to_string(i) + ".bin");
            }
            size_ = 50000;
        } else {
            data_files_.push_back(root_ + "/cifar-10-batches-bin/test_batch.bin");
            size_ = 10000;
        }
    }

    torch::data::Example<> get(size_t index) override {
        constexpr size_t kRecordBytes = 3073;  // 1 label + 3072 image bytes
        constexpr size_t kImageBytes = 3072;
        if (index >= size_) {
            throw std::out_of_range("CIFAR10 index out of range");
        }

        size_t file_idx = (mode_ == Mode::kTrain) ? (index / 10000) : 0;
        size_t offset_in_file = (mode_ == Mode::kTrain) ? (index % 10000) : index;
        const std::string& file_path = data_files_.at(file_idx);

        std::ifstream file(file_path, std::ios::binary);
        if (!file.good()) {
            throw std::runtime_error("Error opening CIFAR-10 file at " + file_path);
        }

        file.seekg(static_cast<std::streamoff>(offset_in_file * kRecordBytes), std::ios::beg);
        uint8_t label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);
        if (!file.good()) {
            throw std::runtime_error("Error reading CIFAR-10 label from " + file_path);
        }

        std::vector<uint8_t> buffer(kImageBytes);
        file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(kImageBytes));
        if (!file.good()) {
            throw std::runtime_error("Error reading CIFAR-10 image from " + file_path);
        }

        auto img = torch::from_blob(buffer.data(), {3, 32, 32}, torch::kUInt8).clone();
        img = img.to(torch::kFloat32).div_(255.0);

        auto target = torch::tensor(static_cast<int64_t>(label), torch::kLong);
        return {img, target};
    }

    torch::optional<size_t> size() const override {
        return size_;
    }

private:
    std::string root_;
    Mode mode_;
    std::vector<std::string> data_files_;
    size_t size_ = 0;
};

}  // namespace cifar10_dataset
