#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace privacy {

inline double compute_rdp_gaussian(double q, double sigma, double alpha) {
    if (alpha <= 1.0) return 0.0;
    if (sigma <= 0.0) return std::numeric_limits<double>::infinity();
    
    // For subsampled Gaussian mechanism
    // Using the formula from "Rényi Differential Privacy of the Sampled Gaussian Mechanism"
    
    if (q == 0.0) return 0.0;
    if (q == 1.0) {
        // Full batch: standard Gaussian mechanism
        return alpha / (2.0 * sigma * sigma);
    }
    
    // Subsampled case: use tight RDP bound
    // RDP(α) ≤ (1/(α-1)) * log(1 + q² * (α choose 2) * min(4*(e^(1/σ²)-1), e^(2/σ²)*2))
    // Simplified approximation for practical use:
    double rho = 1.0 / (sigma * sigma);
    
    // Poisson subsampling bound (tight for small q)
    double log_term = 0.0;
    for (int k = 2; k <= static_cast<int>(alpha); k++) {
        // Binomial coefficient (alpha choose k)
        double binom = 1.0;
        for (int j = 0; j < k; j++) {
            binom *= (alpha - j) / (j + 1);
        }
        log_term += binom * std::pow(q, k) * std::exp((k - 1) * k * rho / 2.0);
    }
    
    if (log_term <= 0.0) return 0.0;
    
    return std::log1p(log_term) / (alpha - 1.0);
}

inline double rdp_to_eps(double rdp_alpha, double alpha, double delta) {
    if (delta <= 0.0) return std::numeric_limits<double>::infinity();
    return rdp_alpha + std::log(1.0 / delta) / (alpha - 1.0);
}

class PrivacyAccountant {
public:
    PrivacyAccountant(double noise_multiplier, double sampling_rate, double target_delta = 1e-5)
        : sigma(noise_multiplier), q(sampling_rate), delta(target_delta), 
          total_iterations(0), rdp_sum(0.0) {
        
        for (double alpha = 2.0; alpha <= 256.0; alpha *= 1.1) {
            rdp_orders.push_back(alpha);
        }
        rdp_orders.push_back(256.0);
        rdp_accumulated.resize(rdp_orders.size(), 0.0);
    }
    
    void step() {
        total_iterations++;
        for (size_t i = 0; i < rdp_orders.size(); i++) {
            double rdp = compute_rdp_gaussian(q, sigma, rdp_orders[i]);
            rdp_accumulated[i] += rdp;
        }
    }
    
    void step(int num_steps) {
        for (int i = 0; i < num_steps; i++) {
            step();
        }
    }
    
    std::pair<double, double> get_privacy_spent() const {
        double best_eps = std::numeric_limits<double>::infinity();
        
        for (size_t i = 0; i < rdp_orders.size(); i++) {
            double eps = rdp_to_eps(rdp_accumulated[i], rdp_orders[i], delta);
            best_eps = std::min(best_eps, eps);
        }
        
        return {best_eps, delta};
    }
    
    double get_epsilon(double target_delta) const {
        double best_eps = std::numeric_limits<double>::infinity();
        
        for (size_t i = 0; i < rdp_orders.size(); i++) {
            double eps = rdp_to_eps(rdp_accumulated[i], rdp_orders[i], target_delta);
            best_eps = std::min(best_eps, eps);
        }
        
        return best_eps;
    }
    
    double get_optimal_order() const {
        double best_eps = std::numeric_limits<double>::infinity();
        double best_order = 2.0;
        
        for (size_t i = 0; i < rdp_orders.size(); i++) {
            double eps = rdp_to_eps(rdp_accumulated[i], rdp_orders[i], delta);
            if (eps < best_eps) {
                best_eps = eps;
                best_order = rdp_orders[i];
            }
        }
        
        return best_order;
    }
    
    void reset() {
        total_iterations = 0;
        std::fill(rdp_accumulated.begin(), rdp_accumulated.end(), 0.0);
    }
    
    void print_summary() const {
        auto [eps, del] = get_privacy_spent();
        (void)del;  // Suppress unused variable warning
        
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              PRIVACY BUDGET SUMMARY                          ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "║ Noise multiplier (σ):     " << std::setw(15) << sigma << "                    ║\n";
        std::cout << "║ Sampling rate (q):        " << std::setw(15) << q << "                    ║\n";
        std::cout << "║ Total iterations:         " << std::setw(15) << total_iterations << "                    ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "║ Final ε:                  " << std::setw(15) << eps << "                    ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    }
    
    int get_iterations() const { return total_iterations; }
    double get_noise_multiplier() const { return sigma; }
    double get_sampling_rate() const { return q; }
    double get_delta() const { return delta; }
    
private:
    double sigma;
    double q;
    double delta;
    int total_iterations;
    double rdp_sum;
    std::vector<double> rdp_orders;
    std::vector<double> rdp_accumulated;
};

inline double compute_epsilon(
    double noise_multiplier,
    double sampling_rate,
    int num_iterations,
    double delta = 1e-5
) {
    PrivacyAccountant accountant(noise_multiplier, sampling_rate, delta);
    accountant.step(num_iterations);
    return accountant.get_epsilon(delta);
}

inline double compute_noise_multiplier(
    double target_epsilon,
    double sampling_rate,
    int num_iterations,
    double delta = 1e-5,
    double tol = 0.01
) {
    double low = 0.1;
    double high = 100.0;
    
    while (high - low > tol) {
        double mid = (low + high) / 2.0;
        double eps = compute_epsilon(mid, sampling_rate, num_iterations, delta);
        
        if (eps > target_epsilon) {
            low = mid;  // Need more noise
        } else {
            high = mid; // Can use less noise
        }
    }
    
    return (low + high) / 2.0;
}

} // namespace privacy

