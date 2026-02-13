#include <torch/torch.h>
#include "dp_sgd_libtorch.h"
#include "mnist_cnn.h"
#include "simple_mlp.h"
#include "cifar_cnn.h"
#include "resnet18.h"
#include "cifar10_dataset.h"
#include <string>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <vector>
#include <cctype>
#include <memory>

enum class DatasetKind { MNIST, CIFAR10 };
enum class ModelKind { MNIST_CNN, CIFAR_CNN, SIMPLE_MLP, RESNET18 };

static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static bool ends_with_path(const std::string& path, const std::string& suffix) {
    if (path.size() < suffix.size()) return false;
    return path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::string normalize_cifar_root(const std::string& path) {
    const std::string bin_dir = "cifar-10-batches-bin";
    if (ends_with_path(path, bin_dir)) {
        if (path.size() == bin_dir.size()) return ".";
        size_t pos = path.find_last_of("/\\");
        if (pos == std::string::npos) return ".";
        return path.substr(0, pos);
    }
    return path;
}

static bool cifar_exists_at_root(const std::string& root) {
    std::ifstream test_file(root + "/cifar-10-batches-bin/data_batch_1.bin");
    return test_file.good();
}

int main(int argc, char* argv[]) {
    torch::manual_seed(0);

    DatasetKind dataset_kind = DatasetKind::MNIST;
    ModelKind model_kind = ModelKind::MNIST_CNN;
    bool model_explicit = false;

    // Get dataset path from command-line argument, environment variable, or use default
    std::string data_path;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--") != 0 && data_path.empty()) {
            data_path = arg;
        }
    }
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--dataset=") == 0) {
            std::string ds = to_lower(arg.substr(10));
            if (ds == "mnist") {
                dataset_kind = DatasetKind::MNIST;
            } else if (ds == "cifar10") {
                dataset_kind = DatasetKind::CIFAR10;
            } else {
                std::cerr << "Unknown dataset: " << ds << " (defaulting to MNIST)\n";
            }
        } else if (arg.find("--model=") == 0) {
            std::string m = to_lower(arg.substr(8));
            model_explicit = true;
            if (m == "mnist_cnn") {
                model_kind = ModelKind::MNIST_CNN;
            } else if (m == "cifar_cnn") {
                model_kind = ModelKind::CIFAR_CNN;
            } else if (m == "simple_mlp") {
                model_kind = ModelKind::SIMPLE_MLP;
            } else if (m == "resnet18") {
                model_kind = ModelKind::RESNET18;
            } else {
                std::cerr << "Unknown model: " << m << " (defaulting to dataset default)\n";
            }
        } else if (arg == "--simple-mlp") {
            model_explicit = true;
            model_kind = ModelKind::SIMPLE_MLP;
        }
    }

    if (!model_explicit) {
        model_kind = (dataset_kind == DatasetKind::CIFAR10) ? ModelKind::CIFAR_CNN : ModelKind::MNIST_CNN;
    }

    if (dataset_kind == DatasetKind::CIFAR10 && model_kind == ModelKind::SIMPLE_MLP) {
        std::cerr << "Simple MLP expects 784-dim MNIST inputs. Falling back to CIFAR CNN.\n";
        model_kind = ModelKind::CIFAR_CNN;
    }
    if (dataset_kind == DatasetKind::MNIST && model_kind == ModelKind::RESNET18) {
        std::cerr << "ResNet18 is configured for 3x32x32 inputs. Falling back to MNIST CNN.\n";
        model_kind = ModelKind::MNIST_CNN;
    }

    if (data_path.empty()) {
        if (const char* env_data_path = std::getenv("DATASET_PATH")) {
            data_path = env_data_path;
        } else if (dataset_kind == DatasetKind::MNIST && std::getenv("MNIST_DATA_PATH")) {
            data_path = std::getenv("MNIST_DATA_PATH");
        } else if (dataset_kind == DatasetKind::CIFAR10 && std::getenv("CIFAR10_DATA_PATH")) {
            data_path = std::getenv("CIFAR10_DATA_PATH");
        } else {
            // Default to relative path: "data" in current directory
            data_path = "data";
        }
    }
    if (dataset_kind == DatasetKind::CIFAR10) {
        data_path = normalize_cifar_root(data_path);
        if (!cifar_exists_at_root(data_path)) {
            if (!data_path.empty() && data_path[0] != '/' && data_path.find(":\\") == std::string::npos) {
                std::string parent_candidate = "../" + data_path;
                if (cifar_exists_at_root(parent_candidate)) {
                    data_path = parent_candidate;
                }
            }
        }
    }

    // Dataset
    std::unique_ptr<torch::data::StatelessDataLoader<
        torch::data::datasets::MapDataset<
            torch::data::datasets::MapDataset<
                torch::data::datasets::MNIST,
                torch::data::transforms::Normalize<>>,
            torch::data::transforms::Stack<>>,
        torch::data::samplers::SequentialSampler>> data_loader_mnist;

    std::unique_ptr<torch::data::StatelessDataLoader<
        torch::data::datasets::MapDataset<
            torch::data::datasets::MapDataset<
                cifar10_dataset::CIFAR10,
                torch::data::transforms::Normalize<>>,
            torch::data::transforms::Stack<>>,
        torch::data::samplers::SequentialSampler>> data_loader_cifar;

    if (dataset_kind == DatasetKind::MNIST) {
        auto dataset = torch::data::datasets::MNIST(data_path)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
        size_t ds_size = dataset.size().value();
        data_loader_mnist = torch::data::make_data_loader(
            std::move(dataset),
            torch::data::samplers::SequentialSampler(ds_size),
            torch::data::DataLoaderOptions().batch_size(32)
        );
    } else {
        auto dataset = cifar10_dataset::CIFAR10(data_path)
                           .map(torch::data::transforms::Normalize<>(
                               std::vector<double>{0.4914, 0.4822, 0.4465},
                               std::vector<double>{0.2023, 0.1994, 0.2010}))
                           .map(torch::data::transforms::Stack<>());
        size_t ds_size = dataset.size().value();
        data_loader_cifar = torch::data::make_data_loader(
            std::move(dataset),
            torch::data::samplers::SequentialSampler(ds_size),
            torch::data::DataLoaderOptions().batch_size(32)
        );
    }

    std::cout << "Using dataset path: " << data_path << std::endl;
    

    double C = 1.0;
    double eta = 0.01;
    double sigma = 1.1;

    MNISTCNN mnist_cnn;
    SimpleMLP simple_mlp;
    CIFARCNN cifar_cnn;
    ResNet18 resnet18;

    if (model_kind == ModelKind::MNIST_CNN) {
        mnist_cnn->to(torch::kCPU);
    } else if (model_kind == ModelKind::SIMPLE_MLP) {
        simple_mlp->to(torch::kCPU);
    } else if (model_kind == ModelKind::CIFAR_CNN) {
        cifar_cnn->to(torch::kCPU);
    } else if (model_kind == ModelKind::RESNET18) {
        resnet18->to(torch::kCPU);
    }

    // Training loop
    for (int epoch = 0; epoch < 1; epoch++) {
        int batch_idx = 0;
        if (dataset_kind == DatasetKind::MNIST) {
            for (auto& batch : *data_loader_mnist) {
                if (model_kind == ModelKind::SIMPLE_MLP) {
                    dp_sgd_step(simple_mlp, batch.data, batch.target, C, eta, sigma);
                } else {
                    dp_sgd_step(mnist_cnn, batch.data, batch.target, C, eta, sigma);
                }
                if (batch_idx % 100 == 0)
                    std::cout << "Batch " << batch_idx << " done\n";
                batch_idx++;
            }
        } else {
            for (auto& batch : *data_loader_cifar) {
                if (model_kind == ModelKind::CIFAR_CNN) {
                    dp_sgd_step(cifar_cnn, batch.data, batch.target, C, eta, sigma);
                } else if (model_kind == ModelKind::RESNET18) {
                    dp_sgd_step(resnet18, batch.data, batch.target, C, eta, sigma);
                } else {
                    dp_sgd_step(cifar_cnn, batch.data, batch.target, C, eta, sigma);
                }
                if (batch_idx % 100 == 0)
                    std::cout << "Batch " << batch_idx << " done\n";
                batch_idx++;
            }
        }
    }

    std::cout << "Training complete.\n";
    return 0;
}
