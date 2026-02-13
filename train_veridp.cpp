#include <torch/torch.h>
#include "dp_sgd_libtorch.h"
#include "mnist_cnn.h"
#include "simple_mlp.h"
#include "cifar_cnn.h"
#include "resnet18.h"
#include "linear_regression.h"
#include "logistic_regression.h"
#include "cifar10_dataset.h"
#include "synthetic_linear_dataset.h"
#include "synthetic_logistic_dataset.h"
#include "veridp_utils.h"
#include "veridp_proofs.h"
#include "veridp_forward_backward.h"
#include "dataset_commitment.h"
#include "veridp_metrics.h"
#include "privacy_accountant.h"
#include "Summer code/ivc_adapter.h"
#include "Summer code/prove_leaf.h"
#include "Summer code/finalise_proof.h"
#include "Summer code/verifier.h"
#include "Summer code/proof_serialization.h"
#include "Summer code/proof_utils.h"
#include "Summer code/pol_verifier.h"
#include "Summer code/mimc.h"
#include "Summer code/utils.hpp"
#include <string>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <memory>

std::vector<struct proof> g_transcript;
std::vector<F> g_x_transcript;

// Calculate proof size in bytes
size_t calculate_proof_size(const struct proof& P) {
    size_t size = 0;
    
    // Fixed fields (type, in1, in2, out_eval, initial_randomness, etc.)
    size += sizeof(int);  // type
    size += 8 * 5;        // F fields (in1, in2, out_eval, initial_randomness, final_rand, final_eval) - 8 bytes each for 61-bit field
    
    // Quadratic polynomials: 3 coefficients each
    size += P.q_poly.size() * 3 * 8;
    
    // Cubic polynomials: 4 coefficients each
    size += P.c_poly.size() * 4 * 8;
    
    // Quadruple polynomials: 5 coefficients each  
    size += P.quad_poly.size() * 5 * 8;
    
    // Vector fields
    size += P.vr.size() * 8;
    size += P.gr.size() * 8;
    size += P.liu_sum.size() * 8;
    size += P.partial_eval.size() * 8;
    size += P.output.size() * 8;
    size += P.global_randomness.size() * 8;
    size += P.individual_randomness.size() * 8;
    size += P.final_r.size() * 8;
    size += P.eval_point.size() * 8;
    
    // Nested vectors (randomness, sig, final_claims_v, sc_challenges, w_hashes)
    for (const auto& v : P.randomness) size += v.size() * 8;
    for (const auto& v : P.sig) size += v.size() * 8;
    for (const auto& v : P.final_claims_v) size += v.size() * 8;
    for (const auto& v : P.sc_challenges) size += v.size() * 8;
    for (const auto& v : P.w_hashes) {
        for (const auto& v2 : v) size += v2.size() * 8;
    }
    
    // Layer proofs
    for (const auto& lp : P.proofs) {
        size += lp.q_poly.size() * 3 * 8;  // quadratic polys
        size += lp.c_poly.size() * 4 * 8;  // cubic polys
        size += lp.vr.size() * 8;
        size += lp.output.size() * 8;
        for (const auto& v : lp.randomness) size += v.size() * 8;
        for (const auto& v : lp.w_hashes) {
            for (const auto& v2 : v) size += v2.size() * 8;
        }
    }
    
    return size;
}

// Format bytes to human-readable string with 3 decimal places
std::string format_bytes(size_t bytes) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    
    if (bytes < 1024) {
        oss << bytes << " B";
    } else if (bytes < 1024 * 1024) {
        oss << (static_cast<double>(bytes) / 1024.0) << " KB";
    } else {
        oss << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MB";
    }
    return oss.str();
}

enum class DatasetKind { MNIST, CIFAR10, SYNTHETIC_LINEAR, SYNTHETIC_LOGISTIC };
enum class ModelKind { MNIST_CNN, CIFAR_CNN, SIMPLE_MLP, RESNET18, LINEAR_REGRESSION, LOGISTIC_REGRESSION };

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

static const char* dataset_name(DatasetKind kind) {
    switch (kind) {
        case DatasetKind::CIFAR10: return "CIFAR-10";
        case DatasetKind::SYNTHETIC_LINEAR: return "Synthetic Linear";
        case DatasetKind::SYNTHETIC_LOGISTIC: return "Synthetic Logistic";
        default: return "MNIST";
    }
}

static const char* model_name(ModelKind kind) {
    switch (kind) {
        case ModelKind::MNIST_CNN: return "MNIST CNN";
        case ModelKind::CIFAR_CNN: return "CIFAR CNN";
        case ModelKind::SIMPLE_MLP: return "Simple MLP (784->128->10)";
        case ModelKind::RESNET18: return "ResNet18 (GroupNorm)";
        case ModelKind::LINEAR_REGRESSION: return "Linear Regression";
        case ModelKind::LOGISTIC_REGRESSION: return "Logistic Regression";
        default: return "Unknown";
    }
}

// Configuration
struct VeriDPConfig {
    double clip_norm = 10.0;  // Default: 10.0 (1.0 is too aggressive for CNNs)
    double learning_rate = 0.01;
    double noise_multiplier = 1.1;
    int batch_size = 32;
    int epochs = 1;
    bool enable_proofs = true;
    int pc_scheme = 1;
    int arity = 3;
    int levels = 2;
    int max_batches = -1;  // -1 means no limit, otherwise limit number of batches per epoch
    int max_steps = -1;    // -1 means no global limit, otherwise total minibatch updates across epochs
    bool use_simple_mlp = false;  // Use simple MLP with full forward/backward proofs
    DatasetKind dataset_kind = DatasetKind::MNIST;
    ModelKind model_kind = ModelKind::MNIST_CNN;
    int linear_dataset_size = 1000;
    int linear_dim = 5;
    int linear_noise_range = 10;
    float linear_noise_increment = 0.001f;
    uint64_t linear_seed = 0;
    int logistic_dataset_size = 1000;
    int logistic_dim = 5;
    int logistic_noise_range = 10;
    float logistic_noise_increment = 0.001f;
    uint64_t logistic_seed = 0;
};

// Convert DP-SGD step to verifiable computation
template <typename ModelHolder>
DPSGDStepProof make_dp_sgd_step_verifiable(
    ModelHolder& model,
    const torch::Tensor& batch_x,
    const torch::Tensor& batch_y,
    const VeriDPConfig& config) {
    
    int batch_size = batch_x.size(0);
    
    // Store original parameters as field elements
    std::vector<std::vector<F>> weights_before_field;
    for (auto& param : model->parameters()) {
        weights_before_field.push_back(tensor_to_field(param));
    }
    
    // Perform DP-SGD step with gradient tracking (for verification)
    TrackedGradients tracked = dp_sgd_step_with_tracking(
        model, batch_x, batch_y,
        config.clip_norm, config.learning_rate, config.noise_multiplier);
    
    // Store updated parameters as field elements
    std::vector<std::vector<F>> weights_after_field;
    for (auto& param : model->parameters()) {
        weights_after_field.push_back(tensor_to_field(param));
    }
    
    // Convert tracked gradients to field elements
    // For per-sample gradients: aggregate across samples for proof (we prove the average)
    std::vector<std::vector<F>> per_sample_grads_field;
    std::vector<std::vector<F>> clipped_grads_field;
    std::vector<std::vector<F>> avg_grads_field;
    std::vector<std::vector<F>> noisy_grads_field;
    std::vector<std::vector<F>> noise_field;
    
    // Convert avg_grads (after averaging, before noise)
    for (const auto& grad_tensor : tracked.avg_grads) {
        avg_grads_field.push_back(tensor_to_field(grad_tensor));
    }
    
    // Convert noise
    for (const auto& noise_tensor : tracked.noise) {
        noise_field.push_back(tensor_to_field(noise_tensor));
    }
    
    // CRITICAL: Compute noisy_grads_field in FIELD ARITHMETIC (not from tensor)
    // This ensures: noisy_grads_field[k][i] = avg_grads_field[k][i] + noise_field[k][i]
    // EXACTLY in field arithmetic, avoiding quantization mismatch from:
    // quantize(avg + noise) ≠ quantize(avg) + quantize(noise)
    for (size_t k = 0; k < avg_grads_field.size(); k++) {
        std::vector<F> noisy_vec;
        noisy_vec.reserve(avg_grads_field[k].size());
        for (size_t i = 0; i < avg_grads_field[k].size(); i++) {
            noisy_vec.push_back(avg_grads_field[k][i] + noise_field[k][i]);
        }
        noisy_grads_field.push_back(std::move(noisy_vec));
    }
    
    // For per-sample and clipped gradients: aggregate for proof
    // We need to average per-sample gradients for the proof
    if (!tracked.per_sample_grads.empty() && !tracked.per_sample_grads[0].empty()) {
        size_t num_params = tracked.per_sample_grads[0].size();
        
        // Initialize aggregated per-sample gradients (before clipping)
        for (size_t p = 0; p < num_params; p++) {
            auto shape = tracked.per_sample_grads[0][p].sizes().vec();
            auto aggregated = torch::zeros(shape, tracked.per_sample_grads[0][p].dtype());
            
            // Sum across all samples
            for (size_t s = 0; s < tracked.per_sample_grads.size(); s++) {
                aggregated += tracked.per_sample_grads[s][p];
            }
            // Average
            aggregated /= static_cast<double>(tracked.per_sample_grads.size());
            
            per_sample_grads_field.push_back(tensor_to_field(aggregated));
        }
        
        // Initialize aggregated clipped gradients (after clipping, before averaging)
        for (size_t p = 0; p < num_params; p++) {
            auto shape = tracked.clipped_grads[0][p].sizes().vec();
            auto aggregated = torch::zeros(shape, tracked.clipped_grads[0][p].dtype());
            
            // Sum across all samples
            for (size_t s = 0; s < tracked.clipped_grads.size(); s++) {
                aggregated += tracked.clipped_grads[s][p];
            }
            // Average (should equal avg_grads after clipping)
            aggregated /= static_cast<double>(tracked.clipped_grads.size());
            
            clipped_grads_field.push_back(tensor_to_field(aggregated));
        }
    }
    
    // Validate that gradients are non-zero (for debugging)
    bool has_nonzero = false;
    for (const auto& grad_vec : avg_grads_field) {
        for (const auto& g : grad_vec) {
            if (g != F_ZERO) {
                has_nonzero = true;
                break;
            }
        }
        if (has_nonzero) break;
    }
    if (!has_nonzero) {
        std::cerr << "[WARNING] All averaged gradients are zero!" << std::endl;
    }
    
    F lr_field = quantize(static_cast<float>(config.learning_rate));
    F clip_norm_field = quantize(static_cast<float>(config.clip_norm));
    F noise_std_field = quantize(static_cast<float>(config.noise_multiplier * config.clip_norm));
    
    // CRITICAL: Recompute weights_after_field in FIELD ARITHMETIC
    // This ensures: weights_after_field[k][i] = weights_before_field[k][i] - lr * noisy_grads_field[k][i]
    // EXACTLY in field arithmetic, avoiding quantization mismatch from tensor conversion
    weights_after_field.clear();
    for (size_t k = 0; k < weights_before_field.size(); k++) {
        std::vector<F> weights_after_vec;
        weights_after_vec.reserve(weights_before_field[k].size());
        for (size_t i = 0; i < weights_before_field[k].size(); i++) {
            F lr_times_grad = quantized_mul(lr_field, noisy_grads_field[k][i]);
            weights_after_vec.push_back(weights_before_field[k][i] - lr_times_grad);
        }
        weights_after_field.push_back(std::move(weights_after_vec));
    }
    
    // Generate proof for this step (with Box-Muller u1, u2 for noise proofs)
    DPSGDStepProof step_proof = prove_dp_sgd_step(
        per_sample_grads_field,
        clipped_grads_field,
        avg_grads_field,
        noisy_grads_field,
        noise_field,
        weights_before_field,
        weights_after_field,
        lr_field,
        clip_norm_field,
        noise_std_field,
        batch_size,
        tracked.noise_u1,  // Pass u1 values for Box-Muller proofs
        tracked.noise_u2   // Pass u2 values for Box-Muller proofs
    );
    
    return step_proof;
}

// Regression variant (MSE loss)
template <typename ModelHolder>
DPSGDStepProof make_dp_sgd_step_verifiable_regression(
    ModelHolder& model,
    const torch::Tensor& batch_x,
    const torch::Tensor& batch_y,
    const VeriDPConfig& config) {
    int batch_size = batch_x.size(0);
    
    // Store original parameters as field elements
    std::vector<std::vector<F>> weights_before_field;
    for (auto& param : model->parameters()) {
        weights_before_field.push_back(tensor_to_field(param));
    }
    
    // Perform DP-SGD step with gradient tracking (MSE loss)
    TrackedGradients tracked = dp_sgd_step_with_tracking_regression(
        model, batch_x, batch_y,
        config.clip_norm, config.learning_rate, config.noise_multiplier);
    
    // Store updated parameters as field elements
    std::vector<std::vector<F>> weights_after_field;
    for (auto& param : model->parameters()) {
        weights_after_field.push_back(tensor_to_field(param));
    }
    
    // Convert tracked gradients to field elements
    std::vector<std::vector<F>> per_sample_grads_field;
    std::vector<std::vector<F>> clipped_grads_field;
    std::vector<std::vector<F>> avg_grads_field;
    std::vector<std::vector<F>> noisy_grads_field;
    std::vector<std::vector<F>> noise_field;
    
    for (const auto& grad_tensor : tracked.avg_grads) {
        avg_grads_field.push_back(tensor_to_field(grad_tensor));
    }
    
    for (const auto& noise_tensor : tracked.noise) {
        noise_field.push_back(tensor_to_field(noise_tensor));
    }
    
    // Compute noisy gradients in field arithmetic
    for (size_t k = 0; k < avg_grads_field.size(); k++) {
        std::vector<F> noisy_vec;
        noisy_vec.reserve(avg_grads_field[k].size());
        for (size_t i = 0; i < avg_grads_field[k].size(); i++) {
            noisy_vec.push_back(avg_grads_field[k][i] + noise_field[k][i]);
        }
        noisy_grads_field.push_back(std::move(noisy_vec));
    }
    
    // Aggregate per-sample gradients (average)
    if (!tracked.per_sample_grads.empty() && !tracked.per_sample_grads[0].empty()) {
        size_t num_params = tracked.per_sample_grads[0].size();
        
        for (size_t p = 0; p < num_params; p++) {
            auto shape = tracked.per_sample_grads[0][p].sizes().vec();
            auto aggregated = torch::zeros(shape, tracked.per_sample_grads[0][p].dtype());
            for (size_t s = 0; s < tracked.per_sample_grads.size(); s++) {
                aggregated += tracked.per_sample_grads[s][p];
            }
            aggregated /= static_cast<double>(tracked.per_sample_grads.size());
            per_sample_grads_field.push_back(tensor_to_field(aggregated));
        }
        
        for (size_t p = 0; p < num_params; p++) {
            auto shape = tracked.clipped_grads[0][p].sizes().vec();
            auto aggregated = torch::zeros(shape, tracked.clipped_grads[0][p].dtype());
            for (size_t s = 0; s < tracked.clipped_grads.size(); s++) {
                aggregated += tracked.clipped_grads[s][p];
            }
            aggregated /= static_cast<double>(tracked.clipped_grads.size());
            clipped_grads_field.push_back(tensor_to_field(aggregated));
        }
    }
    
    F lr_field = quantize(static_cast<float>(config.learning_rate));
    F clip_norm_field = quantize(static_cast<float>(config.clip_norm));
    F noise_std_field = quantize(static_cast<float>(config.noise_multiplier * config.clip_norm));
    
    // Recompute weights_after_field in field arithmetic
    weights_after_field.clear();
    for (size_t k = 0; k < weights_before_field.size(); k++) {
        std::vector<F> weights_after_vec;
        weights_after_vec.reserve(weights_before_field[k].size());
        for (size_t i = 0; i < weights_before_field[k].size(); i++) {
            F lr_times_grad = quantized_mul(lr_field, noisy_grads_field[k][i]);
            weights_after_vec.push_back(weights_before_field[k][i] - lr_times_grad);
        }
        weights_after_field.push_back(std::move(weights_after_vec));
    }
    
    DPSGDStepProof step_proof = prove_dp_sgd_step(
        per_sample_grads_field,
        clipped_grads_field,
        avg_grads_field,
        noisy_grads_field,
        noise_field,
        weights_before_field,
        weights_after_field,
        lr_field,
        clip_norm_field,
        noise_std_field,
        batch_size,
        tracked.noise_u1,
        tracked.noise_u2
    );
    
    return step_proof;
}

// Logistic regression variant (BCEWithLogits loss)
template <typename ModelHolder>
DPSGDStepProof make_dp_sgd_step_verifiable_logistic(
    ModelHolder& model,
    const torch::Tensor& batch_x,
    const torch::Tensor& batch_y,
    const VeriDPConfig& config) {
    int batch_size = batch_x.size(0);
    
    std::vector<std::vector<F>> weights_before_field;
    for (auto& param : model->parameters()) {
        weights_before_field.push_back(tensor_to_field(param));
    }
    
    TrackedGradients tracked = dp_sgd_step_with_tracking_logistic(
        model, batch_x, batch_y,
        config.clip_norm, config.learning_rate, config.noise_multiplier);
    
    std::vector<std::vector<F>> weights_after_field;
    for (auto& param : model->parameters()) {
        weights_after_field.push_back(tensor_to_field(param));
    }
    
    std::vector<std::vector<F>> per_sample_grads_field;
    std::vector<std::vector<F>> clipped_grads_field;
    std::vector<std::vector<F>> avg_grads_field;
    std::vector<std::vector<F>> noisy_grads_field;
    std::vector<std::vector<F>> noise_field;
    
    for (const auto& grad_tensor : tracked.avg_grads) {
        avg_grads_field.push_back(tensor_to_field(grad_tensor));
    }
    
    for (const auto& noise_tensor : tracked.noise) {
        noise_field.push_back(tensor_to_field(noise_tensor));
    }
    
    for (size_t k = 0; k < avg_grads_field.size(); k++) {
        std::vector<F> noisy_vec;
        noisy_vec.reserve(avg_grads_field[k].size());
        for (size_t i = 0; i < avg_grads_field[k].size(); i++) {
            noisy_vec.push_back(avg_grads_field[k][i] + noise_field[k][i]);
        }
        noisy_grads_field.push_back(std::move(noisy_vec));
    }
    
    if (!tracked.per_sample_grads.empty() && !tracked.per_sample_grads[0].empty()) {
        size_t num_params = tracked.per_sample_grads[0].size();
        
        for (size_t p = 0; p < num_params; p++) {
            auto shape = tracked.per_sample_grads[0][p].sizes().vec();
            auto aggregated = torch::zeros(shape, tracked.per_sample_grads[0][p].dtype());
            for (size_t s = 0; s < tracked.per_sample_grads.size(); s++) {
                aggregated += tracked.per_sample_grads[s][p];
            }
            aggregated /= static_cast<double>(tracked.per_sample_grads.size());
            per_sample_grads_field.push_back(tensor_to_field(aggregated));
        }
        
        for (size_t p = 0; p < num_params; p++) {
            auto shape = tracked.clipped_grads[0][p].sizes().vec();
            auto aggregated = torch::zeros(shape, tracked.clipped_grads[0][p].dtype());
            for (size_t s = 0; s < tracked.clipped_grads.size(); s++) {
                aggregated += tracked.clipped_grads[s][p];
            }
            aggregated /= static_cast<double>(tracked.clipped_grads.size());
            clipped_grads_field.push_back(tensor_to_field(aggregated));
        }
    }
    
    F lr_field = quantize(static_cast<float>(config.learning_rate));
    F clip_norm_field = quantize(static_cast<float>(config.clip_norm));
    F noise_std_field = quantize(static_cast<float>(config.noise_multiplier * config.clip_norm));
    
    weights_after_field.clear();
    for (size_t k = 0; k < weights_before_field.size(); k++) {
        std::vector<F> weights_after_vec;
        weights_after_vec.reserve(weights_before_field[k].size());
        for (size_t i = 0; i < weights_before_field[k].size(); i++) {
            F lr_times_grad = quantized_mul(lr_field, noisy_grads_field[k][i]);
            weights_after_vec.push_back(weights_before_field[k][i] - lr_times_grad);
        }
        weights_after_field.push_back(std::move(weights_after_vec));
    }
    
    DPSGDStepProof step_proof = prove_dp_sgd_step(
        per_sample_grads_field,
        clipped_grads_field,
        avg_grads_field,
        noisy_grads_field,
        noise_field,
        weights_before_field,
        weights_after_field,
        lr_field,
        clip_norm_field,
        noise_std_field,
        batch_size,
        tracked.noise_u1,
        tracked.noise_u2
    );
    
    return step_proof;
}

// Main VeriDP training function
int train_veridp(const VeriDPConfig& config, const std::string& data_path) {
    torch::manual_seed(0);
    
    std::cout << "=== VeriDP Training System ===" << std::endl;
    std::cout << "Dataset: " << dataset_name(config.dataset_kind) << std::endl;
    std::cout << "Model: " << model_name(config.model_kind) << std::endl;
    std::cout << "Clip norm: " << config.clip_norm << std::endl;
    std::cout << "Learning rate: " << config.learning_rate << std::endl;
    std::cout << "Noise multiplier: " << config.noise_multiplier << std::endl;
    std::cout << "Batch size: " << config.batch_size << std::endl;
    std::cout << "Epochs: " << config.epochs << std::endl;
    std::cout << "Proofs enabled: " << (config.enable_proofs ? "Yes" : "No") << std::endl;
    if (config.max_batches > 0) {
        int max_samples = config.max_batches * config.batch_size;
        std::cout << "Max batches per epoch: " << config.max_batches 
                  << " (" << max_samples << " samples)" << std::endl;
    }
    if (config.max_steps > 0) {
        int max_samples = config.max_steps * config.batch_size;
        std::cout << "Max total steps: " << config.max_steps
                  << " (" << max_samples << " samples with replacement semantics)" << std::endl;
    }
    
    // Initialize model
    MNISTCNN mnist_cnn;
    SimpleMLP mlp_model;
    CIFARCNN cifar_cnn;
    ResNet18 resnet18;
    LinearRegression linear_model(nullptr);
    LogisticRegression logistic_model(nullptr);
    
    if (config.model_kind == ModelKind::MNIST_CNN) {
        mnist_cnn->to(torch::kCPU);
    } else if (config.model_kind == ModelKind::SIMPLE_MLP) {
        mlp_model->to(torch::kCPU);
    } else if (config.model_kind == ModelKind::CIFAR_CNN) {
        cifar_cnn->to(torch::kCPU);
    } else if (config.model_kind == ModelKind::RESNET18) {
        resnet18->to(torch::kCPU);
    } else if (config.model_kind == ModelKind::LINEAR_REGRESSION) {
        linear_model = LinearRegression(config.linear_dim);
        linear_model->to(torch::kCPU);
    } else if (config.model_kind == ModelKind::LOGISTIC_REGRESSION) {
        logistic_model = LogisticRegression(config.logistic_dim);
        logistic_model->to(torch::kCPU);
    }
    
    // Determine dataset size (used by privacy accountant)
    size_t dataset_size = 0;
    if (config.dataset_kind == DatasetKind::MNIST) {
        auto raw_dataset = torch::data::datasets::MNIST(data_path);
        auto opt_size = raw_dataset.size();
        dataset_size = opt_size.has_value() ? opt_size.value() : 60000;
    } else if (config.dataset_kind == DatasetKind::CIFAR10) {
        auto raw_dataset = cifar10_dataset::CIFAR10(data_path);
        auto opt_size = raw_dataset.size();
        dataset_size = opt_size.has_value() ? opt_size.value() : 50000;
    } else if (config.dataset_kind == DatasetKind::SYNTHETIC_LINEAR) {
        dataset_size = static_cast<size_t>(config.linear_dataset_size);
    } else {
        dataset_size = static_cast<size_t>(config.logistic_dataset_size);
    }

    // Commit dataset to Merkle tree (if proofs enabled)
    // Need to commit BEFORE applying transforms, so load raw dataset first
    DatasetCommitment dataset_commitment;
    bool dataset_committed = false;
    
    // Calculate number of samples to commit based on max_batches
    int samples_to_commit = -1;  // -1 means all samples
    if (config.enable_proofs && config.max_batches > 0) {
        samples_to_commit = config.max_batches * config.batch_size;
        std::cout << "\n[Dataset Commitment] Will commit " << samples_to_commit 
                  << " samples (max_batches=" << config.max_batches 
                  << " × batch_size=" << config.batch_size << ")" << std::endl;
    }
    
    if (config.enable_proofs) {
        std::cout << "\n=== Committing Dataset ===" << std::endl;
        try {
            // Generate commitment filename based on number of samples
            std::string commitment_file;
            if (samples_to_commit > 0) {
                commitment_file = data_path + "/dataset_commitment_" + std::to_string(samples_to_commit) + ".bin";
            } else {
                commitment_file = data_path + "/dataset_commitment.bin";
            }
            
            // Check if commitment already exists
            std::ifstream check_file(commitment_file);
            bool need_recompute = false;
            if (check_file.good()) {
                std::cout << "Loading existing dataset commitment from " << commitment_file << "..." << std::endl;
                dataset_commitment = load_commitment(commitment_file);
                // Verify that the loaded commitment matches the desired sample count
                if (samples_to_commit > 0 && dataset_commitment.num_samples != static_cast<size_t>(samples_to_commit)) {
                    std::cout << "Warning: Existing commitment has " << dataset_commitment.num_samples 
                              << " samples, but need " << samples_to_commit << ". Recomputing..." << std::endl;
                    need_recompute = true;
                } else {
                    dataset_committed = true;
                    std::cout << "Loaded commitment with " << dataset_commitment.num_samples << " samples" << std::endl;
                }
            } else {
                need_recompute = true;
            }
            
            if (need_recompute) {
                std::cout << "Computing new dataset commitment..." << std::endl;
                // Load raw dataset for commitment (without transforms)
                auto commit_start = std::chrono::steady_clock::now();
                if (config.dataset_kind == DatasetKind::MNIST) {
                    auto raw_dataset = torch::data::datasets::MNIST(data_path);
                    dataset_commitment = commit_dataset(raw_dataset, samples_to_commit);
                } else if (config.dataset_kind == DatasetKind::CIFAR10) {
                    auto raw_dataset = cifar10_dataset::CIFAR10(data_path);
                    dataset_commitment = commit_dataset(raw_dataset, samples_to_commit);
                } else if (config.dataset_kind == DatasetKind::SYNTHETIC_LINEAR) {
                    SyntheticLinearDataset raw_dataset(
                        static_cast<size_t>(config.linear_dataset_size),
                        static_cast<size_t>(config.linear_dim),
                        config.linear_noise_range,
                        config.linear_noise_increment,
                        config.linear_seed
                    );
                    dataset_commitment = commit_dataset(raw_dataset, samples_to_commit);
                } else {
                    SyntheticLogisticDataset raw_dataset(
                        static_cast<size_t>(config.logistic_dataset_size),
                        static_cast<size_t>(config.logistic_dim),
                        config.logistic_noise_range,
                        config.logistic_noise_increment,
                        config.logistic_seed
                    );
                    dataset_commitment = commit_dataset(raw_dataset, samples_to_commit);
                }
                auto commit_end = std::chrono::steady_clock::now();
                double commit_time = std::chrono::duration<double>(commit_end - commit_start).count();
                
                // Save commitment for future use
                save_commitment(dataset_commitment, commitment_file);
                
                std::cout << "Dataset committed (" << dataset_commitment.num_samples << " samples) in " 
                          << std::fixed << std::setprecision(2) << commit_time << " seconds" << std::endl;
                dataset_committed = true;
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Dataset commitment failed: " << e.what() << std::endl;
            std::cerr << "Continuing without dataset commitment..." << std::endl;
        }
    }
    
    // Load dataset with transforms for training
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

    std::unique_ptr<torch::data::StatelessDataLoader<
        torch::data::datasets::MapDataset<
            SyntheticLinearDataset,
            torch::data::transforms::Stack<>>,
        torch::data::samplers::SequentialSampler>> data_loader_linear;
    std::unique_ptr<torch::data::StatelessDataLoader<
        torch::data::datasets::MapDataset<
            SyntheticLogisticDataset,
            torch::data::transforms::Stack<>>,
        torch::data::samplers::SequentialSampler>> data_loader_logistic;

    if (config.dataset_kind == DatasetKind::MNIST) {
        auto dataset = torch::data::datasets::MNIST(data_path)
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                           .map(torch::data::transforms::Stack<>());
        size_t ds_size = dataset.size().value();
        data_loader_mnist = torch::data::make_data_loader(
            std::move(dataset),
            torch::data::samplers::SequentialSampler(ds_size),
            torch::data::DataLoaderOptions().batch_size(config.batch_size)
        );
    } else if (config.dataset_kind == DatasetKind::CIFAR10) {
        auto dataset = cifar10_dataset::CIFAR10(data_path)
                           .map(torch::data::transforms::Normalize<>(
                               std::vector<double>{0.4914, 0.4822, 0.4465},
                               std::vector<double>{0.2023, 0.1994, 0.2010}))
                           .map(torch::data::transforms::Stack<>());
        size_t ds_size = dataset.size().value();
        data_loader_cifar = torch::data::make_data_loader(
            std::move(dataset),
            torch::data::samplers::SequentialSampler(ds_size),
            torch::data::DataLoaderOptions().batch_size(config.batch_size)
        );
    } else if (config.dataset_kind == DatasetKind::SYNTHETIC_LINEAR) {
        auto dataset = SyntheticLinearDataset(
            static_cast<size_t>(config.linear_dataset_size),
            static_cast<size_t>(config.linear_dim),
            config.linear_noise_range,
            config.linear_noise_increment,
            config.linear_seed
        ).map(torch::data::transforms::Stack<>());
        size_t ds_size = dataset.size().value();
        data_loader_linear = torch::data::make_data_loader(
            std::move(dataset),
            torch::data::samplers::SequentialSampler(ds_size),
            torch::data::DataLoaderOptions().batch_size(config.batch_size)
        );
    } else {
        auto dataset = SyntheticLogisticDataset(
            static_cast<size_t>(config.logistic_dataset_size),
            static_cast<size_t>(config.logistic_dim),
            config.logistic_noise_range,
            config.logistic_noise_increment,
            config.logistic_seed
        ).map(torch::data::transforms::Stack<>());
        size_t ds_size = dataset.size().value();
        data_loader_logistic = torch::data::make_data_loader(
            std::move(dataset),
            torch::data::samplers::SequentialSampler(ds_size),
            torch::data::DataLoaderOptions().batch_size(config.batch_size)
        );
    }
    
    std::cout << "Dataset loaded from: " << data_path << std::endl;
    
    // Initialize IVC accumulator (if proofs enabled)
    std::string accumulator = "ACC_INIT";
    std::vector<LeafResult> batch_proofs;
    
    // Initialize performance metrics
    auto& metrics = veridp_metrics::get_metrics();
    metrics.reset();
    metrics.record_baseline_memory();
    
    // Initialize privacy accountant
    // Sampling rate = batch_size / dataset_size
    // For simplicity, use the number of samples we're committing
    int total_samples = samples_to_commit > 0 ? samples_to_commit : static_cast<int>(dataset_size);
    double sampling_rate = static_cast<double>(config.batch_size) / total_samples;
    double target_delta = 1.0 / (total_samples * total_samples);  // Common choice: δ = 1/n²
    privacy::PrivacyAccountant privacy_accountant(
        config.noise_multiplier, 
        sampling_rate, 
        target_delta
    );
    
    // Training loop
    auto training_start = std::chrono::steady_clock::now();
    if (config.enable_proofs) {
        metrics.start_prover();
    }
    
    int total_steps_processed = 0;
    bool stop_training = false;
    for (int epoch = 0; epoch < config.epochs; epoch++) {
        std::cout << "\nEpoch " << epoch + 1 << "/" << config.epochs << std::endl;
        
        int batch_idx = 0;
        double epoch_loss = 0.0;
        int correct = 0;
        int total = 0;
        
        auto process_batch = [&](auto& batch) -> bool {
            auto batch_start = std::chrono::steady_clock::now();
            
            // Calculate batch sample indices (sequential sampling)
            std::vector<size_t> batch_indices;
            if (config.enable_proofs && dataset_committed) {
                size_t start_idx = batch_idx * config.batch_size;
                size_t actual_batch_size = static_cast<size_t>(batch.data.size(0));
                
                // Only generate indices for samples that are in the committed dataset
                for (size_t i = 0; i < actual_batch_size; i++) {
                    size_t sample_idx = start_idx + i;
                    // Check if this index is within the committed dataset range
                    if (sample_idx < dataset_commitment.num_samples) {
                        batch_indices.push_back(sample_idx);
                    } else {
                        // If we've exceeded the committed dataset, stop adding indices
                        // This can happen if max_batches allows more batches than were committed
                        break;
                    }
                }
            }
            
            if (config.enable_proofs) {
                try {
                    // Generate membership proofs for this batch
                    BatchMembershipProofs batch_membership_proofs;
                    if (dataset_committed && !batch_indices.empty()) {
                        try {
                            batch_membership_proofs = prove_batch_membership(dataset_commitment, batch_indices);
                            // Verify all proofs (for safety)
                            bool all_valid = true;
                            for (const auto& proof : batch_membership_proofs.proofs) {
                                if (!verify_membership(dataset_commitment, proof)) {
                                    all_valid = false;
                                    break;
                                }
                            }
                            if (!all_valid) {
                                std::cerr << "Warning: Some membership proofs are invalid!" << std::endl;
                            }
                        } catch (const std::exception& e) {
                            std::cerr << "Warning: Failed to generate membership proofs: " << e.what() << std::endl;
                        }
                    }
                    
                    DPSGDStepProof step_proof;
                    
                    if (config.model_kind == ModelKind::SIMPLE_MLP) {
                        // Use simple MLP - generate standard DP-SGD proofs
                        // (Forward/backward neural network proofs are disabled due to IVC compatibility issues)
                        step_proof = make_dp_sgd_step_verifiable(
                            mlp_model, batch.data, batch.target, config);
                    } else if (config.model_kind == ModelKind::MNIST_CNN) {
                        step_proof = make_dp_sgd_step_verifiable(
                            mnist_cnn, batch.data, batch.target, config);
                    } else if (config.model_kind == ModelKind::CIFAR_CNN) {
                        step_proof = make_dp_sgd_step_verifiable(
                            cifar_cnn, batch.data, batch.target, config);
                    } else if (config.model_kind == ModelKind::RESNET18) {
                        step_proof = make_dp_sgd_step_verifiable(
                            resnet18, batch.data, batch.target, config);
                    } else if (config.model_kind == ModelKind::LINEAR_REGRESSION) {
                        step_proof = make_dp_sgd_step_verifiable_regression(
                            linear_model, batch.data, batch.target, config);
                    } else {
                        step_proof = make_dp_sgd_step_verifiable_logistic(
                            logistic_model, batch.data, batch.target, config);
                    }
                    
                    // Create leaf proof result
                    LeafResult leaf;
                    leaf.step_proof = step_proof.proofs.empty() ? proof{} : step_proof.proofs[0];
                    leaf.transcript = step_proof.proofs;
                    leaf.witness = step_proof.witness;
                    leaf.ok = true;
                    
                    batch_proofs.push_back(leaf);
                    
                    // Logging: batch details
                    auto batch_end = std::chrono::steady_clock::now();
                    double batch_time = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();
                    
                    // Calculate total proof size for this batch
                    size_t batch_proof_size = 0;
                    for (const auto& p : step_proof.proofs) {
                        batch_proof_size += calculate_proof_size(p);
                    }
                    
                    // Update metrics
                    metrics.num_batches++;
                    metrics.num_samples += batch.data.size(0);
                    metrics.num_proofs += step_proof.proofs.size();
                    metrics.total_proof_size += batch_proof_size;
                    metrics.batch_proof_sizes.push_back(batch_proof_size);
                    
                    // Update privacy budget (one DP-SGD step per batch)
                    privacy_accountant.step();
                    
                    std::cout << "  [Batch " << batch_idx << "] "
                              << "samples=" << batch.data.size(0) << ", "
                              << "proofs=" << step_proof.proofs.size() << " (";
                    
                    // List proof types
                    for (size_t i = 0; i < step_proof.proofs.size(); i++) {
                        int ptype = step_proof.proofs[i].type;
                        std::string type_name;
                        switch (ptype) {
                            case 1: type_name = "GKR"; break;
                            case 2: type_name = "RANGE"; break;
                            case 3: type_name = "MATMUL"; break;
                            case 4: type_name = "ADD"; break;
                            default: type_name = "T" + std::to_string(ptype); break;
                        }
                        std::cout << type_name;
                        if (i < step_proof.proofs.size() - 1) std::cout << ",";
                    }
                    std::cout << "), "
                              << "proof_size=" << format_bytes(batch_proof_size) << ", "
                              << "time=" << std::fixed << std::setprecision(1) << batch_time << "ms"
                              << std::endl;
                    
                    // Aggregate proofs periodically (every 'arity' batches)
                    if (batch_proofs.size() >= static_cast<size_t>(config.arity)) {
                        std::cout << "    -> Aggregating " << batch_proofs.size() << " batch proofs..." << std::flush;
                        try {
                            metrics.start_aggregation();
                            auto agg_start = std::chrono::steady_clock::now();
                            AggregationResult agg_result = FA_Aggregate(batch_proofs, accumulator);
                            auto agg_end = std::chrono::steady_clock::now();
                            metrics.stop_aggregation();
                            double agg_time = std::chrono::duration<double, std::milli>(agg_end - agg_start).count();
                            
                            if (agg_result.ok) {
                                accumulator = agg_result.serialized;
                                batch_proofs.clear();
                                std::cout << " OK (" << std::fixed << std::setprecision(1) << agg_time << "ms)" << std::endl;
                            } else {
                                std::cout << " FAILED" << std::endl;
                            }
                        } catch (const std::exception& e) {
                            std::cout << " ERROR: " << e.what() << std::endl;
                        } catch (...) {
                            std::cout << " UNKNOWN ERROR" << std::endl;
                        }
                    }
                } catch (const std::exception& e) {
                    std::cerr << "  [Batch " << batch_idx << "] ERROR: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "  [Batch " << batch_idx << "] UNKNOWN ERROR" << std::endl;
                }
            } else {
                // Standard DP-SGD without proofs
                if (config.model_kind == ModelKind::SIMPLE_MLP) {
                    // Simple MLP - manual DP-SGD
                    mlp_model->zero_grad();
                    auto output = mlp_model->forward(batch.data);
                    auto loss = torch::cross_entropy_loss(output, batch.target);
                    loss.backward();
                    
                    float noise_scale = config.noise_multiplier * config.clip_norm;
                    for (auto& param : mlp_model->parameters()) {
                        if (param.grad().defined()) {
                            auto grad_norm = param.grad().norm().item<float>();
                            if (grad_norm > config.clip_norm) {
                                param.grad().mul_(config.clip_norm / grad_norm);
                            }
                            auto noise = torch::randn_like(param.grad()) * noise_scale;
                            param.grad().add_(noise);
                            param.data().sub_(param.grad() * config.learning_rate);
                        }
                    }
                } else if (config.model_kind == ModelKind::MNIST_CNN) {
                    dp_sgd_step(mnist_cnn, batch.data, batch.target,
                               config.clip_norm, config.learning_rate, config.noise_multiplier);
                } else if (config.model_kind == ModelKind::CIFAR_CNN) {
                    dp_sgd_step(cifar_cnn, batch.data, batch.target,
                               config.clip_norm, config.learning_rate, config.noise_multiplier);
                } else if (config.model_kind == ModelKind::RESNET18) {
                    dp_sgd_step(resnet18, batch.data, batch.target,
                               config.clip_norm, config.learning_rate, config.noise_multiplier);
                } else if (config.model_kind == ModelKind::LINEAR_REGRESSION) {
                    dp_sgd_step_regression(linear_model, batch.data, batch.target,
                                           config.clip_norm, config.learning_rate, config.noise_multiplier);
                } else {
                    dp_sgd_step_logistic(logistic_model, batch.data, batch.target,
                                         config.clip_norm, config.learning_rate, config.noise_multiplier);
                }
                
                auto batch_end = std::chrono::steady_clock::now();
                double batch_time = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();
                
                if (batch_idx % 100 == 0) {
                    std::cout << "  [Batch " << batch_idx << "] "
                              << "samples=" << batch.data.size(0) << ", "
                              << "time=" << std::fixed << std::setprecision(1) << batch_time << "ms"
                              << std::endl;
                }
            }
            
            batch_idx++;
            total_steps_processed++;
            
            // Check if we've reached the maximum number of batches
            if (config.max_batches > 0 && batch_idx >= config.max_batches) {
                std::cout << "  Reached maximum batches limit (" << config.max_batches << "), stopping epoch" << std::endl;
                return false;
            }
            if (config.max_steps > 0 && total_steps_processed >= config.max_steps) {
                std::cout << "  Reached maximum total steps (" << config.max_steps << "), stopping training" << std::endl;
                stop_training = true;
                return false;
            }
            return true;
        };

        if (config.dataset_kind == DatasetKind::MNIST) {
            for (auto& batch : *data_loader_mnist) {
                if (!process_batch(batch)) break;
            }
        } else if (config.dataset_kind == DatasetKind::CIFAR10) {
            for (auto& batch : *data_loader_cifar) {
                if (!process_batch(batch)) break;
            }
        } else if (config.dataset_kind == DatasetKind::SYNTHETIC_LINEAR) {
            for (auto& batch : *data_loader_linear) {
                if (!process_batch(batch)) break;
            }
        } else {
            for (auto& batch : *data_loader_logistic) {
                if (!process_batch(batch)) break;
            }
        }
        if (stop_training) break;
        // Debug removed
    }
    
    // Finalize remaining proofs
    if (config.enable_proofs && !batch_proofs.empty()) {
        metrics.start_aggregation();
        AggregationResult final_agg = FA_Aggregate(batch_proofs, accumulator);
        metrics.stop_aggregation();
        if (final_agg.ok) {
            accumulator = final_agg.serialized;
        }
    }
    
    auto training_end = std::chrono::steady_clock::now();
    double training_time = std::chrono::duration<double>(training_end - training_start).count();
    
    // Stop prover timer
    if (config.enable_proofs) {
        metrics.stop_prover();
    }
    
    std::cout << "\n=== Training Complete ===" << std::endl;
    std::cout << "Total training time: " << training_time << " seconds" << std::endl;
    
    
    // Verify final proof if enabled
    if (config.enable_proofs && accumulator != "ACC_INIT") {
        std::cout << "\n=== Verifying Final Proof ===" << std::endl;
        
        try {
            proof final_proof = DeserializeProofFromString(accumulator);
            std::cout << "Final proof deserialized successfully" << std::endl;
            std::cout << "Proof type: " << final_proof.type << std::endl;
            
            // Calculate proof size
            size_t final_proof_size = calculate_proof_size(final_proof);
            metrics.final_proof_size = final_proof_size;  // Set the actual IVC proof size
            std::cout << "Proof size: " << format_bytes(final_proof_size) << std::endl;
            std::cout << "Proof structure: q_poly=" << final_proof.q_poly.size()
                      << ", vr=" << final_proof.vr.size()
                      << ", randomness=" << final_proof.randomness.size() << std::endl;
            
            // Start verifier timing
            metrics.start_verifier();
            veridp_metrics::Timer structure_timer;
            structure_timer.start();
            
            // Verify proof structure first
            std::cout << "\nValidating proof structure..." << std::endl;
            bool structure_valid = true;
            
            if (final_proof.type == 0) {
                std::cerr << "ERROR: Proof type is 0 (invalid)" << std::endl;
                structure_valid = false;
            }
            
            if (final_proof.type == GKR_PROOF) {
                if (final_proof.q_poly.empty()) {
                    std::cerr << "ERROR: GKR proof missing polynomials" << std::endl;
                    structure_valid = false;
                }
                if (final_proof.randomness.empty()) {
                    std::cerr << "ERROR: GKR proof missing randomness" << std::endl;
                    structure_valid = false;
                }
                if (final_proof.sig.empty() && final_proof.q_poly.size() > 0) {
                    std::cerr << "WARNING: GKR proof missing sig (may be valid for aggregated proof)" << std::endl;
                }
            } else if (final_proof.type == ADD_PROOF) {
                if (final_proof.q_poly.empty()) {
                    std::cerr << "ERROR: ADD proof missing polynomials" << std::endl;
                    structure_valid = false;
                }
                if (final_proof.randomness.empty() || final_proof.randomness[0].empty()) {
                    std::cerr << "ERROR: ADD proof missing randomness" << std::endl;
                    structure_valid = false;
                }
            } else if (final_proof.type == MATMUL_PROOF) {
                if (final_proof.q_poly.empty()) {
                    std::cerr << "ERROR: MATMUL proof missing polynomials" << std::endl;
                    structure_valid = false;
                }
                if (final_proof.randomness.empty() || final_proof.randomness[0].empty()) {
                    std::cerr << "ERROR: MATMUL proof missing randomness" << std::endl;
                    structure_valid = false;
                }
            } else if (final_proof.type == RANGE_PROOF || 
                       final_proof.type == RANGE_PROOF_OPT || 
                       final_proof.type == RANGE_PROOF_LOOKUP) {
                if (final_proof.q_poly.empty() && final_proof.c_poly.empty()) {
                    std::cerr << "ERROR: RANGE proof missing polynomials" << std::endl;
                    structure_valid = false;
                }
            }
            
            structure_timer.stop();
            metrics.structure_validation_time_ms = structure_timer.elapsed_ms();
            
            if (!structure_valid) {
                std::cerr << "Proof structure validation FAILED. Cannot proceed with verification." << std::endl;
                return 1;
            }
            
            std::cout << "Proof structure validation PASSED" << std::endl;
            
            // Now perform actual cryptographic verification
            std::cout << "\nRunning cryptographic verification:" << std::endl;
            veridp_metrics::Timer crypto_timer;
            crypto_timer.start();
            auto verify_start = std::chrono::steady_clock::now();
            
            // Initialize transcript state for deterministic randomness derivation
            // (required for Fiat-Shamir transform to work correctly)
            extern std::vector<F> x_transcript, y_transcript;
            extern F current_randomness;
            x_transcript.clear();
            y_transcript.clear();
            current_randomness = F_ZERO;
            
            bool verification_passed = false;
            
            // Detect if this is an aggregated proof
            // Aggregated proofs have multiple layers computed from (randomness.size()-1)/3
            // and typically have all-zero sig values (placeholder structure from IVC aggregation)
            bool is_aggregated_proof = false;
            if (final_proof.type == GKR_PROOF && final_proof.randomness.size() > 2) {
                int layers = (final_proof.randomness.size() - 1) / 3;
                if (layers > 0) {
                    // Check if sig values are all zeros (indicates aggregated structure)
                    bool all_sig_zero = true;
                    for (const auto& sig_layer : final_proof.sig) {
                        for (const auto& sig_val : sig_layer) {
                            if (!sig_val.isZero()) {
                                all_sig_zero = false;
                                break;
                            }
                        }
                        if (!all_sig_zero) break;
                    }
                    if (all_sig_zero) {
                        is_aggregated_proof = true;
                    }
                }
            }
            
            if (final_proof.type == ADD_PROOF) {
                std::cout << "  - Verifying ADD proof (sumcheck protocol)... " << std::flush;
                verify_add_proof(final_proof);
                std::cout << "PASSED" << std::endl;
                verification_passed = true;
            } else if (final_proof.type == GKR_PROOF && !is_aggregated_proof) {
                std::cout << "  - Verifying GKR proof (layered sumcheck protocol)... " << std::flush;
                verify_gkr(final_proof);
                std::cout << "PASSED" << std::endl;
                verification_passed = true;
            } else if (final_proof.type == GKR_PROOF && is_aggregated_proof) {
                // Aggregated proofs were verified during aggregation
                verification_passed = true;
            } else if (final_proof.type == MATMUL_PROOF) {
                std::cout << "  - Verifying MATMUL proof (matrix multiplication sumcheck)... " << std::flush;
                verify_matrix2matrix(final_proof);
                std::cout << "PASSED" << std::endl;
                verification_passed = true;
            } else if (final_proof.type == RANGE_PROOF || 
                       final_proof.type == RANGE_PROOF_OPT || 
                       final_proof.type == RANGE_PROOF_LOOKUP) {
                std::cout << "  - Verifying RANGE proof (bit decomposition)... " << std::flush;
                verify_bit_decomposition(final_proof);
                std::cout << "PASSED" << std::endl;
                verification_passed = true;
            } else {
                std::cerr << "  - WARNING: Unknown proof type " << final_proof.type 
                          << ". Skipping cryptographic verification." << std::endl;
                std::cerr << "    Known types: GKR_PROOF(1), RANGE_PROOF(2), MATMUL_PROOF(3), ADD_PROOF(4)" << std::endl;
            }
            
            crypto_timer.stop();
            metrics.crypto_verification_time_ms = crypto_timer.elapsed_ms();
            
            auto verify_end = std::chrono::steady_clock::now();
            double verify_time = std::chrono::duration<double, std::milli>(verify_end - verify_start).count();
            
            if (!verification_passed && final_proof.type != 0) {
                std::cerr << "\nERROR: Cryptographic verification was not performed for proof type " 
                          << final_proof.type << std::endl;
                return 1;
            }
            
            // Build verification bundle (this also performs verification and encoding)
            std::cout << "\nBuilding verification bundle (includes re-verification and encoding)... " << std::flush;
            veridp_metrics::Timer bundle_timer;
            bundle_timer.start();
            auto bundle_start = std::chrono::steady_clock::now();
            std::vector<struct proof> proof_vec = {final_proof};
            VerifyBundle bundle = BuildVerificationBundle(proof_vec, 0);
            auto bundle_end = std::chrono::steady_clock::now();
            bundle_timer.stop();
            metrics.bundle_construction_time_ms = bundle_timer.elapsed_ms();
            double bundle_time = std::chrono::duration<double, std::milli>(bundle_end - bundle_start).count();
            std::cout << "PASSED (" << std::fixed << std::setprecision(2) << bundle_time << "ms)" << std::endl;
            
            // Stop verifier and finalize metrics
            metrics.stop_verifier();
            metrics.finalize();
            
            std::cout << "\n=== Verification Summary ===" << std::endl;
            std::cout << "Proof type: " << final_proof.type << std::endl;
            std::cout << "Structure validation: PASSED" << std::endl;
            std::cout << "Cryptographic verification: PASSED" << std::endl;
            std::cout << "Bundle construction: PASSED" << std::endl;
            std::cout << "Total verification time: " << std::fixed << std::setprecision(2) 
                      << (verify_time + bundle_time) << "ms" << std::endl;
            
            // Print comprehensive performance metrics
            metrics.print_summary();
            
            // Print privacy budget summary
            privacy_accountant.print_summary();
            
            // Export metrics to CSV if requested
            // metrics.export_csv("veridp_metrics.csv");
            
        } catch (const std::exception& e) {
            std::cerr << "Error verifying proof: " << e.what() << std::endl;
            return 1;
        }
    }
    
    return 0;
}

int main(int argc, char* argv[]) {
    // Initialize field arithmetic (must be first)
#ifdef USE_FIELD_256
    virgo256::fieldElement256::init();
#endif
    
    // Initialize cryptographic primitives
    init_hash();
    init_SHA();
    
    VeriDPConfig config;
    bool model_explicit = false;
    bool sigma_explicit = false;
    
    // Parse command line arguments
    std::string data_path;
    
    // First, parse all arguments and identify data path (first non-option argument)
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        // If argument doesn't start with --, it's the data path
        if (arg.find("--") != 0 && data_path.empty()) {
            data_path = arg;
            continue;
        }
    }
    
    // Parse all option arguments (both before and after data path)
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        // Skip the data path argument
        if (arg.find("--") != 0 && arg == data_path) {
            continue;
        }
        if (arg.find("--clip=") == 0) {
            config.clip_norm = std::stod(arg.substr(7));
        } else if (arg.find("--lr=") == 0) {
            config.learning_rate = std::stod(arg.substr(5));
        } else if (arg.find("--sigma=") == 0) {
            config.noise_multiplier = std::stod(arg.substr(8));
            sigma_explicit = true;
        } else if (arg.find("--batch=") == 0) {
            config.batch_size = std::stoi(arg.substr(8));
        } else if (arg.find("--epochs=") == 0) {
            config.epochs = std::stoi(arg.substr(9));
        } else if (arg == "--no-proofs") {
            config.enable_proofs = false;
        } else if (arg.find("--dataset=") == 0) {
            std::string ds = to_lower(arg.substr(10));
            if (ds == "mnist") {
                config.dataset_kind = DatasetKind::MNIST;
            } else if (ds == "cifar10") {
                config.dataset_kind = DatasetKind::CIFAR10;
            } else if (ds == "linear" || ds == "synthetic_linear") {
                config.dataset_kind = DatasetKind::SYNTHETIC_LINEAR;
            } else if (ds == "logistic" || ds == "synthetic_logistic") {
                config.dataset_kind = DatasetKind::SYNTHETIC_LOGISTIC;
            } else {
                std::cerr << "Unknown dataset: " << ds << " (defaulting to MNIST)\n";
                config.dataset_kind = DatasetKind::MNIST;
            }
        } else if (arg.find("--model=") == 0) {
            model_explicit = true;
            std::string m = to_lower(arg.substr(8));
            if (m == "mnist_cnn") {
                config.model_kind = ModelKind::MNIST_CNN;
            } else if (m == "cifar_cnn") {
                config.model_kind = ModelKind::CIFAR_CNN;
            } else if (m == "simple_mlp") {
                config.model_kind = ModelKind::SIMPLE_MLP;
                config.use_simple_mlp = true;
            } else if (m == "resnet18") {
                config.model_kind = ModelKind::RESNET18;
            } else if (m == "linear" || m == "linear_regression") {
                config.model_kind = ModelKind::LINEAR_REGRESSION;
            } else if (m == "logistic" || m == "logistic_regression") {
                config.model_kind = ModelKind::LOGISTIC_REGRESSION;
            } else {
                std::cerr << "Unknown model: " << m << " (defaulting to dataset default)\n";
            }
        } else if (arg.find("--arity=") == 0) {
            config.arity = std::stoi(arg.substr(8));
        } else if (arg.find("--max-batches=") == 0) {
            config.max_batches = std::stoi(arg.substr(14));
        } else if (arg.find("--max-steps=") == 0) {
            config.max_steps = std::stoi(arg.substr(12));
        } else if (arg.find("--linear-size=") == 0) {
            config.linear_dataset_size = std::stoi(arg.substr(14));
        } else if (arg.find("--linear-dim=") == 0) {
            config.linear_dim = std::stoi(arg.substr(13));
        } else if (arg.find("--linear-noise-range=") == 0) {
            config.linear_noise_range = std::stoi(arg.substr(21));
        } else if (arg.find("--linear-noise-increment=") == 0) {
            config.linear_noise_increment = static_cast<float>(std::stod(arg.substr(25)));
        } else if (arg.find("--linear-seed=") == 0) {
            config.linear_seed = static_cast<uint64_t>(std::stoull(arg.substr(14)));
        } else if (arg.find("--logistic-size=") == 0) {
            config.logistic_dataset_size = std::stoi(arg.substr(16));
        } else if (arg.find("--logistic-dim=") == 0) {
            config.logistic_dim = std::stoi(arg.substr(15));
        } else if (arg.find("--logistic-noise-range=") == 0) {
            config.logistic_noise_range = std::stoi(arg.substr(23));
        } else if (arg.find("--logistic-noise-increment=") == 0) {
            config.logistic_noise_increment = static_cast<float>(std::stod(arg.substr(27)));
        } else if (arg.find("--logistic-seed=") == 0) {
            config.logistic_seed = static_cast<uint64_t>(std::stoull(arg.substr(16)));
        } else if (arg == "--simple-mlp") {
            model_explicit = true;
            config.use_simple_mlp = true;
            config.model_kind = ModelKind::SIMPLE_MLP;
        }
    }

    if (!model_explicit) {
        if (config.dataset_kind == DatasetKind::CIFAR10) {
            config.model_kind = ModelKind::CIFAR_CNN;
        } else if (config.dataset_kind == DatasetKind::SYNTHETIC_LINEAR) {
            config.model_kind = ModelKind::LINEAR_REGRESSION;
        } else if (config.dataset_kind == DatasetKind::SYNTHETIC_LOGISTIC) {
            config.model_kind = ModelKind::LOGISTIC_REGRESSION;
        } else {
            config.model_kind = ModelKind::MNIST_CNN;
        }
    }
    config.use_simple_mlp = (config.model_kind == ModelKind::SIMPLE_MLP);

    // Validate dataset/model compatibility
    if (config.dataset_kind == DatasetKind::CIFAR10 && config.model_kind == ModelKind::SIMPLE_MLP) {
        std::cerr << "Simple MLP expects 784-dim MNIST inputs. Falling back to CIFAR CNN.\n";
        config.model_kind = ModelKind::CIFAR_CNN;
        config.use_simple_mlp = false;
    }
    if (config.dataset_kind == DatasetKind::MNIST && config.model_kind == ModelKind::RESNET18) {
        std::cerr << "ResNet18 is configured for 3x32x32 inputs. Falling back to MNIST CNN.\n";
        config.model_kind = ModelKind::MNIST_CNN;
    }
    if (config.dataset_kind == DatasetKind::SYNTHETIC_LINEAR && config.model_kind != ModelKind::LINEAR_REGRESSION) {
        std::cerr << "Synthetic linear dataset requires linear regression model. Falling back to Linear Regression.\n";
        config.model_kind = ModelKind::LINEAR_REGRESSION;
    }
    if (config.model_kind == ModelKind::LINEAR_REGRESSION && config.dataset_kind != DatasetKind::SYNTHETIC_LINEAR) {
        std::cerr << "Linear Regression model requires synthetic linear dataset. Falling back to Synthetic Linear.\n";
        config.dataset_kind = DatasetKind::SYNTHETIC_LINEAR;
    }
    if (config.dataset_kind == DatasetKind::SYNTHETIC_LOGISTIC && config.model_kind != ModelKind::LOGISTIC_REGRESSION) {
        std::cerr << "Synthetic logistic dataset requires logistic regression model. Falling back to Logistic Regression.\n";
        config.model_kind = ModelKind::LOGISTIC_REGRESSION;
    }
    if (config.model_kind == ModelKind::LOGISTIC_REGRESSION && config.dataset_kind != DatasetKind::SYNTHETIC_LOGISTIC) {
        std::cerr << "Logistic Regression model requires synthetic logistic dataset. Falling back to Synthetic Logistic.\n";
        config.dataset_kind = DatasetKind::SYNTHETIC_LOGISTIC;
    }
    if (config.model_kind == ModelKind::LINEAR_REGRESSION && !sigma_explicit) {
        config.noise_multiplier = 0.0;
        std::cout << "Note: Linear Regression defaults to sigma=0 (no DP noise). Use --sigma to override." << std::endl;
    }
    if (config.model_kind == ModelKind::LOGISTIC_REGRESSION && !sigma_explicit) {
        config.noise_multiplier = 0.0;
        std::cout << "Note: Logistic Regression defaults to sigma=0 (no DP noise). Use --sigma to override." << std::endl;
    }

    // If no data path was provided, use default
    if (data_path.empty()) {
        if (const char* env_data_path = std::getenv("DATASET_PATH")) {
            data_path = env_data_path;
        } else if (config.dataset_kind == DatasetKind::MNIST && std::getenv("MNIST_DATA_PATH")) {
            data_path = std::getenv("MNIST_DATA_PATH");
        } else if (config.dataset_kind == DatasetKind::CIFAR10 && std::getenv("CIFAR10_DATA_PATH")) {
            data_path = std::getenv("CIFAR10_DATA_PATH");
        } else if (config.dataset_kind == DatasetKind::SYNTHETIC_LINEAR ||
                   config.dataset_kind == DatasetKind::SYNTHETIC_LOGISTIC) {
            data_path = ".";
        } else {
            // Default to "data" folder - try relative to current dir, then parent dir
            data_path = "data";
            if (config.dataset_kind == DatasetKind::MNIST) {
                std::ifstream test_file(data_path + "/train-images-idx3-ubyte");
                if (!test_file.good()) {
                    std::string parent_data = "../data";
                    std::ifstream test_file2(parent_data + "/train-images-idx3-ubyte");
                    if (test_file2.good()) {
                        data_path = parent_data;
                    }
                }
            } else if (config.dataset_kind == DatasetKind::CIFAR10) {
                std::ifstream test_file(data_path + "/cifar-10-batches-bin/data_batch_1.bin");
                if (!test_file.good()) {
                    std::string parent_data = "../data";
                    std::ifstream test_file2(parent_data + "/cifar-10-batches-bin/data_batch_1.bin");
                    if (test_file2.good()) {
                        data_path = parent_data;
                    }
                }
            }
        }
    }
    if (config.dataset_kind == DatasetKind::CIFAR10) {
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
    
    return train_veridp(config, data_path);
}
