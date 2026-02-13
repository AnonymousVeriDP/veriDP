#pragma once
#include <torch/torch.h>
#include <vector>
#include <cmath>
#include <random>

inline double box_muller(double mean, double stddev) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double u1 = dis(gen);
    double u2 = dis(gen);
    double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2 * M_PI * u2);
    return mean + stddev * z;
}

struct VerifiableBoxMuller {
    double z;
    double u1;
    double u2;
    double noise;
};

inline VerifiableBoxMuller box_muller_verifiable(double mean, double stddev) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    VerifiableBoxMuller result;
    result.u1 = dis(gen);
    result.u2 = dis(gen);
    result.z = std::sqrt(-2.0 * std::log(result.u1)) * std::cos(2 * M_PI * result.u2);
    result.noise = mean + stddev * result.z;
    return result;
}

inline torch::Tensor gaussian_noise_like(const torch::Tensor& ref, double stddev) {
    auto noise = torch::zeros_like(ref);
    auto acc = noise.flatten();
    for (int64_t i = 0; i < acc.size(0); i++)
        acc[i] = box_muller(0.0, stddev);
    return noise;
}

inline std::pair<torch::Tensor, std::pair<std::vector<double>, std::vector<double>>> 
gaussian_noise_verifiable(const torch::Tensor& ref, double stddev) {
    auto noise = torch::zeros_like(ref);
    auto acc = noise.flatten();
    std::vector<double> u1_values, u2_values;
    u1_values.reserve(acc.size(0));
    u2_values.reserve(acc.size(0));
    for (int64_t i = 0; i < acc.size(0); i++) {
        VerifiableBoxMuller bm = box_muller_verifiable(0.0, stddev);
        acc[i] = bm.noise;
        u1_values.push_back(bm.u1);
        u2_values.push_back(bm.u2);
    }
    return std::make_pair(noise, std::make_pair(u1_values, u2_values));
}

template <typename Model>
inline std::vector<torch::Tensor> compute_per_sample_gradients(
        Model& model, const torch::Tensor& x, const torch::Tensor& y) {
    model->zero_grad();
    auto output = model->forward(x);
    auto loss = torch::nn::functional::cross_entropy(output, y);
    loss.backward();
    std::vector<torch::Tensor> grads;
    grads.reserve(model->parameters().size());
    for (auto& p : model->parameters())
        grads.push_back(p.grad().detach().clone());
    return grads;
}

template <typename Model>
inline std::vector<torch::Tensor> compute_per_sample_gradients_regression(
        Model& model, const torch::Tensor& x, const torch::Tensor& y) {
    model->zero_grad();
    auto output = model->forward(x);
    auto loss = torch::nn::functional::mse_loss(output, y);
    loss.backward();
    std::vector<torch::Tensor> grads;
    grads.reserve(model->parameters().size());
    for (auto& p : model->parameters())
        grads.push_back(p.grad().detach().clone());
    return grads;
}

template <typename Model>
inline void dp_sgd_step(Model& model, const torch::Tensor& batch_x, const torch::Tensor& batch_y,
                        double C, double eta, double sigma) {
    int batch_size = batch_x.size(0);
    std::vector<std::vector<torch::Tensor>> per_sample_grads;
    per_sample_grads.reserve(batch_size);

    for (int i = 0; i < batch_size; i++) {
        auto x_i = batch_x[i].unsqueeze(0);
        auto y_i = batch_y[i].unsqueeze(0);
        per_sample_grads.push_back(compute_per_sample_gradients(model, x_i, y_i));
    }

    for (int i = 0; i < batch_size; i++) {
        double norm_sq = 0.0;
        for (auto& g : per_sample_grads[i])
            norm_sq += g.pow(2).sum().item<double>();
        double norm = std::sqrt(norm_sq);
        double clip_factor = (norm > C ? C / norm : 1.0);
        for (auto& g : per_sample_grads[i])
            g.mul_(clip_factor);
    }

    std::vector<torch::Tensor> avg_grads;
    avg_grads.reserve(model->parameters().size());
    for (auto& p : model->parameters())
        avg_grads.push_back(torch::zeros_like(p));
    for (int i = 0; i < batch_size; i++)
        for (size_t k = 0; k < avg_grads.size(); k++)
            avg_grads[k] += per_sample_grads[i][k];
    for (auto& g : avg_grads)
        g /= batch_size;

    // DP-SGD: noise is N(0, σ²C²) added to SUM, then divided by batch_size
    // Equivalently: N(0, (σC/batch_size)²) added to the AVERAGE
    double noise_std = sigma * C / batch_size;
    for (size_t k = 0; k < avg_grads.size(); k++)
        avg_grads[k] += gaussian_noise_like(avg_grads[k], noise_std);

    size_t k = 0;
    for (auto& p : model->parameters()) {
        p.data().sub_(eta * avg_grads[k]);
        k++;
    }
}

template <typename Model>
inline void dp_sgd_step_regression(Model& model, const torch::Tensor& batch_x, const torch::Tensor& batch_y,
                                   double C, double eta, double sigma) {
    int batch_size = batch_x.size(0);
    std::vector<std::vector<torch::Tensor>> per_sample_grads;
    per_sample_grads.reserve(batch_size);

    for (int i = 0; i < batch_size; i++) {
        auto x_i = batch_x[i].unsqueeze(0);
        auto y_i = batch_y[i].unsqueeze(0);
        per_sample_grads.push_back(compute_per_sample_gradients_regression(model, x_i, y_i));
    }

    for (int i = 0; i < batch_size; i++) {
        double norm_sq = 0.0;
        for (auto& g : per_sample_grads[i])
            norm_sq += g.pow(2).sum().item<double>();
        double norm = std::sqrt(norm_sq);
        double clip_factor = (norm > C ? C / norm : 1.0);
        for (auto& g : per_sample_grads[i])
            g.mul_(clip_factor);
    }

    std::vector<torch::Tensor> avg_grads;
    avg_grads.reserve(model->parameters().size());
    for (auto& p : model->parameters())
        avg_grads.push_back(torch::zeros_like(p));
    for (int i = 0; i < batch_size; i++)
        for (size_t k = 0; k < avg_grads.size(); k++)
            avg_grads[k] += per_sample_grads[i][k];
    for (auto& g : avg_grads)
        g /= batch_size;

    double noise_std = sigma * C / batch_size;
    for (size_t k = 0; k < avg_grads.size(); k++)
        avg_grads[k] += gaussian_noise_like(avg_grads[k], noise_std);

    size_t k = 0;
    for (auto& p : model->parameters()) {
        p.data().sub_(eta * avg_grads[k]);
        k++;
    }
}

struct TrackedGradients {
    std::vector<std::vector<torch::Tensor>> per_sample_grads;
    std::vector<std::vector<torch::Tensor>> clipped_grads;
    std::vector<torch::Tensor> avg_grads;
    std::vector<torch::Tensor> noise;
    std::vector<torch::Tensor> noisy_grads;
    std::vector<std::vector<double>> noise_u1;
    std::vector<std::vector<double>> noise_u2;
};

template <typename Model>
inline TrackedGradients dp_sgd_step_with_tracking(Model& model, const torch::Tensor& batch_x,
                                                   const torch::Tensor& batch_y,
                                                   double C, double eta, double sigma) {
    int batch_size = batch_x.size(0);
    TrackedGradients tracked;
    std::vector<std::vector<torch::Tensor>> per_sample_grads;
    per_sample_grads.reserve(batch_size);

    for (int i = 0; i < batch_size; i++) {
        auto x_i = batch_x[i].unsqueeze(0);
        auto y_i = batch_y[i].unsqueeze(0);
        per_sample_grads.push_back(compute_per_sample_gradients(model, x_i, y_i));
    }
    tracked.per_sample_grads = per_sample_grads;

    for (int i = 0; i < batch_size; i++) {
        double norm_sq = 0.0;
        for (auto& g : per_sample_grads[i])
            norm_sq += g.pow(2).sum().item<double>();
        double norm = std::sqrt(norm_sq);
        double clip_factor = (norm > C ? C / norm : 1.0);
        for (auto& g : per_sample_grads[i])
            g.mul_(clip_factor);
    }
    tracked.clipped_grads = per_sample_grads;

    std::vector<torch::Tensor> avg_grads;
    avg_grads.reserve(model->parameters().size());
    for (auto& p : model->parameters())
        avg_grads.push_back(torch::zeros_like(p));
    for (int i = 0; i < batch_size; i++)
        for (size_t k = 0; k < avg_grads.size(); k++)
            avg_grads[k] += per_sample_grads[i][k];
    for (auto& g : avg_grads)
        g /= batch_size;
    tracked.avg_grads = avg_grads;

    // DP-SGD: noise is N(0, σ²C²) added to SUM, then divided by batch_size
    // Equivalently: N(0, (σC/batch_size)²) added to the AVERAGE
    double noise_std = sigma * C / batch_size;
    std::vector<torch::Tensor> noise, noisy_grads;
    std::vector<std::vector<double>> noise_u1, noise_u2;
    noise.reserve(avg_grads.size());
    noisy_grads.reserve(avg_grads.size());
    noise_u1.reserve(avg_grads.size());
    noise_u2.reserve(avg_grads.size());

    if (noise_std == 0.0) {
        for (size_t k = 0; k < avg_grads.size(); k++) {
            noise.push_back(torch::zeros_like(avg_grads[k]));
            noisy_grads.push_back(avg_grads[k].clone());
            noise_u1.push_back({});
            noise_u2.push_back({});
        }
    } else {
        for (size_t k = 0; k < avg_grads.size(); k++) {
            auto noise_result = gaussian_noise_verifiable(avg_grads[k], noise_std);
            noise.push_back(noise_result.first.clone());
            noise_u1.push_back(noise_result.second.first);
            noise_u2.push_back(noise_result.second.second);
            noisy_grads.push_back((avg_grads[k] + noise_result.first).clone());
        }
    }
    tracked.noise = noise;
    tracked.noisy_grads = noisy_grads;
    tracked.noise_u1 = noise_u1;
    tracked.noise_u2 = noise_u2;

    size_t k = 0;
    for (auto& p : model->parameters()) {
        p.data().sub_(eta * noisy_grads[k]);
        k++;
    }
    return tracked;
}

template <typename Model>
inline TrackedGradients dp_sgd_step_with_tracking_regression(Model& model, const torch::Tensor& batch_x,
                                                             const torch::Tensor& batch_y,
                                                             double C, double eta, double sigma) {
    int batch_size = batch_x.size(0);
    TrackedGradients tracked;
    std::vector<std::vector<torch::Tensor>> per_sample_grads;
    per_sample_grads.reserve(batch_size);

    for (int i = 0; i < batch_size; i++) {
        auto x_i = batch_x[i].unsqueeze(0);
        auto y_i = batch_y[i].unsqueeze(0);
        per_sample_grads.push_back(compute_per_sample_gradients_regression(model, x_i, y_i));
    }
    tracked.per_sample_grads = per_sample_grads;

    for (int i = 0; i < batch_size; i++) {
        double norm_sq = 0.0;
        for (auto& g : per_sample_grads[i])
            norm_sq += g.pow(2).sum().item<double>();
        double norm = std::sqrt(norm_sq);
        double clip_factor = (norm > C ? C / norm : 1.0);
        for (auto& g : per_sample_grads[i])
            g.mul_(clip_factor);
    }
    tracked.clipped_grads = per_sample_grads;

    std::vector<torch::Tensor> avg_grads;
    avg_grads.reserve(model->parameters().size());
    for (auto& p : model->parameters())
        avg_grads.push_back(torch::zeros_like(p));
    for (int i = 0; i < batch_size; i++)
        for (size_t k = 0; k < avg_grads.size(); k++)
            avg_grads[k] += per_sample_grads[i][k];
    for (auto& g : avg_grads)
        g /= batch_size;
    tracked.avg_grads = avg_grads;

    double noise_std = sigma * C / batch_size;
    std::vector<torch::Tensor> noise, noisy_grads;
    std::vector<std::vector<double>> noise_u1, noise_u2;
    noise.reserve(avg_grads.size());
    noisy_grads.reserve(avg_grads.size());
    noise_u1.reserve(avg_grads.size());
    noise_u2.reserve(avg_grads.size());

    if (noise_std == 0.0) {
        for (size_t k = 0; k < avg_grads.size(); k++) {
            noise.push_back(torch::zeros_like(avg_grads[k]));
            noisy_grads.push_back(avg_grads[k].clone());
            noise_u1.push_back({});
            noise_u2.push_back({});
        }
    } else {
        for (size_t k = 0; k < avg_grads.size(); k++) {
            auto noise_result = gaussian_noise_verifiable(avg_grads[k], noise_std);
            noise.push_back(noise_result.first.clone());
            noise_u1.push_back(noise_result.second.first);
            noise_u2.push_back(noise_result.second.second);
            noisy_grads.push_back((avg_grads[k] + noise_result.first).clone());
        }
    }
    tracked.noise = noise;
    tracked.noisy_grads = noisy_grads;
    tracked.noise_u1 = noise_u1;
    tracked.noise_u2 = noise_u2;

    size_t k = 0;
    for (auto& p : model->parameters()) {
        p.data().sub_(eta * noisy_grads[k]);
        k++;
    }
    return tracked;
}

template <typename Model>
inline std::vector<torch::Tensor> compute_per_sample_gradients_logistic(
        Model& model, const torch::Tensor& x, const torch::Tensor& y) {
    model->zero_grad();
    auto logits = model->forward(x);
    auto loss = torch::nn::functional::binary_cross_entropy_with_logits(
        logits, y);
    loss.backward();
    std::vector<torch::Tensor> grads;
    grads.reserve(model->parameters().size());
    for (auto& p : model->parameters())
        grads.push_back(p.grad().detach().clone());
    return grads;
}

template <typename Model>
inline void dp_sgd_step_logistic(Model& model, const torch::Tensor& batch_x, const torch::Tensor& batch_y,
                                 double C, double eta, double sigma) {
    int batch_size = batch_x.size(0);
    std::vector<std::vector<torch::Tensor>> per_sample_grads;
    per_sample_grads.reserve(batch_size);

    for (int i = 0; i < batch_size; i++) {
        auto x_i = batch_x[i].unsqueeze(0);
        auto y_i = batch_y[i].unsqueeze(0);
        per_sample_grads.push_back(compute_per_sample_gradients_logistic(model, x_i, y_i));
    }

    for (int i = 0; i < batch_size; i++) {
        double norm_sq = 0.0;
        for (auto& g : per_sample_grads[i])
            norm_sq += g.pow(2).sum().item<double>();
        double norm = std::sqrt(norm_sq);
        double clip_factor = (norm > C ? C / norm : 1.0);
        for (auto& g : per_sample_grads[i])
            g.mul_(clip_factor);
    }

    std::vector<torch::Tensor> avg_grads;
    avg_grads.reserve(model->parameters().size());
    for (auto& p : model->parameters())
        avg_grads.push_back(torch::zeros_like(p));
    for (int i = 0; i < batch_size; i++)
        for (size_t k = 0; k < avg_grads.size(); k++)
            avg_grads[k] += per_sample_grads[i][k];
    for (auto& g : avg_grads)
        g /= batch_size;

    double noise_std = sigma * C / batch_size;
    for (size_t k = 0; k < avg_grads.size(); k++)
        avg_grads[k] += gaussian_noise_like(avg_grads[k], noise_std);

    size_t k = 0;
    for (auto& p : model->parameters()) {
        p.data().sub_(eta * avg_grads[k]);
        k++;
    }
}

template <typename Model>
inline TrackedGradients dp_sgd_step_with_tracking_logistic(Model& model, const torch::Tensor& batch_x,
                                                           const torch::Tensor& batch_y,
                                                           double C, double eta, double sigma) {
    int batch_size = batch_x.size(0);
    TrackedGradients tracked;
    std::vector<std::vector<torch::Tensor>> per_sample_grads;
    per_sample_grads.reserve(batch_size);

    for (int i = 0; i < batch_size; i++) {
        auto x_i = batch_x[i].unsqueeze(0);
        auto y_i = batch_y[i].unsqueeze(0);
        per_sample_grads.push_back(compute_per_sample_gradients_logistic(model, x_i, y_i));
    }
    tracked.per_sample_grads = per_sample_grads;

    for (int i = 0; i < batch_size; i++) {
        double norm_sq = 0.0;
        for (auto& g : per_sample_grads[i])
            norm_sq += g.pow(2).sum().item<double>();
        double norm = std::sqrt(norm_sq);
        double clip_factor = (norm > C ? C / norm : 1.0);
        for (auto& g : per_sample_grads[i])
            g.mul_(clip_factor);
    }
    tracked.clipped_grads = per_sample_grads;

    std::vector<torch::Tensor> avg_grads;
    avg_grads.reserve(model->parameters().size());
    for (auto& p : model->parameters())
        avg_grads.push_back(torch::zeros_like(p));
    for (int i = 0; i < batch_size; i++)
        for (size_t k = 0; k < avg_grads.size(); k++)
            avg_grads[k] += per_sample_grads[i][k];
    for (auto& g : avg_grads)
        g /= batch_size;
    tracked.avg_grads = avg_grads;

    double noise_std = sigma * C / batch_size;
    std::vector<torch::Tensor> noise, noisy_grads;
    std::vector<std::vector<double>> noise_u1, noise_u2;
    noise.reserve(avg_grads.size());
    noisy_grads.reserve(avg_grads.size());
    noise_u1.reserve(avg_grads.size());
    noise_u2.reserve(avg_grads.size());

    if (noise_std == 0.0) {
        for (size_t k = 0; k < avg_grads.size(); k++) {
            noise.push_back(torch::zeros_like(avg_grads[k]));
            noisy_grads.push_back(avg_grads[k].clone());
            noise_u1.push_back({});
            noise_u2.push_back({});
        }
    } else {
        for (size_t k = 0; k < avg_grads.size(); k++) {
            auto noise_result = gaussian_noise_verifiable(avg_grads[k], noise_std);
            noise.push_back(noise_result.first.clone());
            noise_u1.push_back(noise_result.second.first);
            noise_u2.push_back(noise_result.second.second);
            noisy_grads.push_back((avg_grads[k] + noise_result.first).clone());
        }
    }

    tracked.noise = noise;
    tracked.noisy_grads = noisy_grads;
    tracked.noise_u1 = noise_u1;
    tracked.noise_u2 = noise_u2;

    size_t k = 0;
    for (auto& p : model->parameters()) {
        p.data().sub_(eta * noisy_grads[k]);
        k++;
    }
    return tracked;
}
