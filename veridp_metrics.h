#pragma once

#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace veridp_metrics {

class Timer {
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        running = true;
    }
    
    void stop() {
        if (running) {
            end_time = std::chrono::high_resolution_clock::now();
            running = false;
        }
    }
    
    double elapsed_ms() const {
        auto end = running ? std::chrono::high_resolution_clock::now() : end_time;
        return std::chrono::duration<double, std::milli>(end - start_time).count();
    }
    
    double elapsed_sec() const {
        return elapsed_ms() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool running = false;
};

inline size_t get_current_memory_bytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
    return 0;
#else
    // Linux: read from /proc/self/status
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line.substr(6));
            size_t kb;
            iss >> kb;
            return kb * 1024;  // Convert KB to bytes
        }
    }
    return 0;
#endif
}

inline size_t get_peak_memory_bytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.PeakWorkingSetSize;
    }
    return 0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss * 1024;  // Convert KB to bytes
    }
    return 0;
#endif
}

struct PerformanceMetrics {
    size_t total_proof_size = 0;  // Cumulative batch proofs (for internal tracking)
    size_t final_proof_size = 0;  // Actual final IVC proof size (what verifier receives)
    std::vector<size_t> batch_proof_sizes;
    double total_prover_time_ms = 0;
    double clipping_proof_time_ms = 0;
    double noise_generation_proof_time_ms = 0;
    double noise_addition_proof_time_ms = 0;
    double weight_update_proof_time_ms = 0;
    double aggregation_time_ms = 0;
    double other_proof_time_ms = 0;
    double total_verifier_time_ms = 0;
    double structure_validation_time_ms = 0;
    double crypto_verification_time_ms = 0;
    double bundle_construction_time_ms = 0;
    size_t baseline_memory = 0;
    size_t peak_memory = 0;
    size_t final_memory = 0;
    size_t peak_memory_tracked = 0;
    int num_batches = 0;
    int num_samples = 0;
    int num_proofs = 0;
    double test_accuracy = -1.0;  // -1 means not computed
    Timer prover_timer;
    Timer clipping_timer;
    Timer noise_gen_timer;
    Timer noise_add_timer;
    Timer weight_update_timer;
    Timer aggregation_timer;
    Timer verifier_timer;
    
    void reset() {
        total_proof_size = 0;
        final_proof_size = 0;
        batch_proof_sizes.clear();
        total_prover_time_ms = 0;
        clipping_proof_time_ms = 0;
        noise_generation_proof_time_ms = 0;
        noise_addition_proof_time_ms = 0;
        weight_update_proof_time_ms = 0;
        aggregation_time_ms = 0;
        other_proof_time_ms = 0;
        total_verifier_time_ms = 0;
        structure_validation_time_ms = 0;
        crypto_verification_time_ms = 0;
        bundle_construction_time_ms = 0;
        baseline_memory = 0;
        peak_memory = 0;
        final_memory = 0;
        peak_memory_tracked = 0;
        num_batches = 0;
        num_samples = 0;
        num_proofs = 0;
        test_accuracy = -1.0;
    }
    
    void record_baseline_memory() {
        size_t cur = get_current_memory_bytes();
        baseline_memory = cur;
        if (cur > peak_memory_tracked) peak_memory_tracked = cur;
    }
    
    void record_peak_memory() {
        size_t cur = get_current_memory_bytes();
        if (cur > peak_memory_tracked) peak_memory_tracked = cur;
        peak_memory = peak_memory_tracked ? peak_memory_tracked : get_peak_memory_bytes();
    }
    
    void record_final_memory() {
        size_t cur = get_current_memory_bytes();
        final_memory = cur;
        if (cur > peak_memory_tracked) peak_memory_tracked = cur;
    }
    
    void start_clipping_proof() { clipping_timer.start(); }
    void stop_clipping_proof() { 
        clipping_timer.stop(); 
        clipping_proof_time_ms += clipping_timer.elapsed_ms();
    }
    
    void start_noise_generation_proof() { noise_gen_timer.start(); }
    void stop_noise_generation_proof() { 
        noise_gen_timer.stop(); 
        noise_generation_proof_time_ms += noise_gen_timer.elapsed_ms();
    }
    
    void start_noise_addition_proof() { noise_add_timer.start(); }
    void stop_noise_addition_proof() { 
        noise_add_timer.stop(); 
        noise_addition_proof_time_ms += noise_add_timer.elapsed_ms();
    }
    
    void start_weight_update_proof() { weight_update_timer.start(); }
    void stop_weight_update_proof() { 
        weight_update_timer.stop(); 
        weight_update_proof_time_ms += weight_update_timer.elapsed_ms();
    }
    
    void start_aggregation() { aggregation_timer.start(); }
    void stop_aggregation() { 
        aggregation_timer.stop(); 
        aggregation_time_ms += aggregation_timer.elapsed_ms();
    }
    
    void start_prover() { prover_timer.start(); }
    void stop_prover() { 
        prover_timer.stop(); 
        total_prover_time_ms += prover_timer.elapsed_ms();
    }
    
    void start_verifier() { verifier_timer.start(); }
    void stop_verifier() { 
        verifier_timer.stop(); 
        total_verifier_time_ms = verifier_timer.elapsed_ms();
    }
    
    void finalize() {
        double tracked_time = clipping_proof_time_ms + 
                              noise_generation_proof_time_ms + 
                              noise_addition_proof_time_ms + 
                              weight_update_proof_time_ms +
                              aggregation_time_ms;
        other_proof_time_ms = total_prover_time_ms - tracked_time;
        if (other_proof_time_ms < 0) other_proof_time_ms = 0;
        record_final_memory();
        record_peak_memory();
    }
    
    double clipping_percentage() const {
        if (total_prover_time_ms <= 0) return 0;
        return (clipping_proof_time_ms / total_prover_time_ms) * 100.0;
    }
    
    double noise_generation_percentage() const {
        if (total_prover_time_ms <= 0) return 0;
        return (noise_generation_proof_time_ms / total_prover_time_ms) * 100.0;
    }
    
    double noise_addition_percentage() const {
        if (total_prover_time_ms <= 0) return 0;
        return (noise_addition_proof_time_ms / total_prover_time_ms) * 100.0;
    }
    
    double weight_update_percentage() const {
        if (total_prover_time_ms <= 0) return 0;
        return (weight_update_proof_time_ms / total_prover_time_ms) * 100.0;
    }
    
    double aggregation_percentage() const {
        if (total_prover_time_ms <= 0) return 0;
        return (aggregation_time_ms / total_prover_time_ms) * 100.0;
    }
    
    double other_percentage() const {
        if (total_prover_time_ms <= 0) return 0;
        return (other_proof_time_ms / total_prover_time_ms) * 100.0;
    }
    
    static std::string format_bytes(size_t bytes) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3);
        if (bytes < 1024) {
            ss << bytes << " B";
        } else if (bytes < 1024 * 1024) {
            ss << static_cast<double>(bytes) / 1024.0 << " KB";
        } else if (bytes < 1024ULL * 1024 * 1024) {
            ss << static_cast<double>(bytes) / (1024.0 * 1024.0) << " MB";
        } else {
            ss << static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) << " GB";
        }
        return ss.str();
    }
    
    void print_summary() const {
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              VeriDP PERFORMANCE METRICS                      ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        
        // General stats
        std::cout << "║ GENERAL                                                      ║\n";
        std::cout << "║   Batches processed:      " << std::setw(10) << num_batches << "                         ║\n";
        std::cout << "║   Samples processed:      " << std::setw(10) << num_samples << "                         ║\n";
        std::cout << "║   Total proofs generated: " << std::setw(10) << num_proofs << "                         ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        
        // Proof size
        std::cout << "║ PROOF SIZE                                                   ║\n";
        std::cout << "║   Final IVC proof size:   " << std::setw(15) << format_bytes(final_proof_size) << "                    ║\n";
        if (num_batches > 0) {
            std::cout << "║   Avg batch proof:        " << std::setw(15) << format_bytes(total_proof_size / num_batches) << "                    ║\n";
        }
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        
        // Prover time
        std::cout << "║ PROVER TIME                                                  ║\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "║   Total prover time:      " << std::setw(10) << total_prover_time_ms << " ms                      ║\n";
        std::cout << "║   ├─ Clipping proof:      " << std::setw(10) << clipping_proof_time_ms 
                  << " ms (" << std::setw(5) << clipping_percentage() << "%)           ║\n";
        std::cout << "║   ├─ Noise generation:    " << std::setw(10) << noise_generation_proof_time_ms 
                  << " ms (" << std::setw(5) << noise_generation_percentage() << "%)           ║\n";
        std::cout << "║   ├─ Noise addition:      " << std::setw(10) << noise_addition_proof_time_ms 
                  << " ms (" << std::setw(5) << noise_addition_percentage() << "%)           ║\n";
        std::cout << "║   ├─ Weight update:       " << std::setw(10) << weight_update_proof_time_ms 
                  << " ms (" << std::setw(5) << weight_update_percentage() << "%)           ║\n";
        std::cout << "║   ├─ Aggregation:         " << std::setw(10) << aggregation_time_ms 
                  << " ms (" << std::setw(5) << aggregation_percentage() << "%)           ║\n";
        std::cout << "║   └─ Other:               " << std::setw(10) << other_proof_time_ms 
                  << " ms (" << std::setw(5) << other_percentage() << "%)           ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        
        // Verifier time
        std::cout << "║ VERIFIER TIME                                                ║\n";
        std::cout << "║   Total verifier time:    " << std::setw(10) << total_verifier_time_ms << " ms                      ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        
        // Memory
        std::cout << "║ MEMORY USAGE                                                 ║\n";
        std::cout << "║   Baseline memory:        " << std::setw(15) << format_bytes(baseline_memory) << "                    ║\n";
        std::cout << "║   Peak memory:            " << std::setw(15) << format_bytes(peak_memory) << "                    ║\n";
        std::cout << "║   Final memory:           " << std::setw(15) << format_bytes(final_memory) << "                    ║\n";
        std::cout << "║   Memory overhead:        " << std::setw(15) << format_bytes(peak_memory - baseline_memory) << "                    ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
        
        // Key ratios
        std::cout << "║ KEY RATIOS                                                   ║\n";
        if (total_verifier_time_ms > 0) {
            double ratio = total_prover_time_ms / total_verifier_time_ms;
            std::cout << "║   Prover/Verifier ratio:  " << std::setw(10) << ratio << "x                       ║\n";
        }
        if (num_samples > 0) {
            double ms_per_sample = total_prover_time_ms / num_samples;
            std::cout << "║   Prover time/sample:     " << std::setw(10) << ms_per_sample << " ms                      ║\n";
        }
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    }
    
    void export_csv(const std::string& filepath) const {
        std::ofstream out(filepath);
        if (!out) {
            std::cerr << "Failed to open " << filepath << " for writing\n";
            return;
        }
        
        out << "metric,value,unit\n";
        out << "num_batches," << num_batches << ",count\n";
        out << "num_samples," << num_samples << ",count\n";
        out << "num_proofs," << num_proofs << ",count\n";
        out << "final_proof_size," << final_proof_size << ",bytes\n";
        out << "cumulative_batch_proof_size," << total_proof_size << ",bytes\n";
        out << "total_prover_time," << total_prover_time_ms << ",ms\n";
        out << "clipping_proof_time," << clipping_proof_time_ms << ",ms\n";
        out << "noise_generation_proof_time," << noise_generation_proof_time_ms << ",ms\n";
        out << "noise_addition_proof_time," << noise_addition_proof_time_ms << ",ms\n";
        out << "weight_update_proof_time," << weight_update_proof_time_ms << ",ms\n";
        out << "aggregation_time," << aggregation_time_ms << ",ms\n";
        out << "other_proof_time," << other_proof_time_ms << ",ms\n";
        out << "clipping_percentage," << clipping_percentage() << ",percent\n";
        out << "noise_generation_percentage," << noise_generation_percentage() << ",percent\n";
        out << "total_verifier_time," << total_verifier_time_ms << ",ms\n";
        out << "baseline_memory," << baseline_memory << ",bytes\n";
        out << "peak_memory," << peak_memory << ",bytes\n";
        out << "final_memory," << final_memory << ",bytes\n";
        out << "memory_overhead," << (peak_memory - baseline_memory) << ",bytes\n";
        if (test_accuracy >= 0) {
            out << "test_accuracy," << test_accuracy << ",percent\n";
        }
        
        std::cout << "[Metrics] Exported to " << filepath << "\n";
    }
};

inline PerformanceMetrics& get_metrics() {
    static PerformanceMetrics metrics;
    return metrics;
}

class ScopedTimer {
public:
    enum Component {
        CLIPPING,
        NOISE_GENERATION,
        NOISE_ADDITION,
        WEIGHT_UPDATE,
        AGGREGATION,
        PROVER,
        VERIFIER
    };
    
    ScopedTimer(Component comp) : component(comp) {
        auto& m = get_metrics();
        switch (component) {
            case CLIPPING: m.start_clipping_proof(); break;
            case NOISE_GENERATION: m.start_noise_generation_proof(); break;
            case NOISE_ADDITION: m.start_noise_addition_proof(); break;
            case WEIGHT_UPDATE: m.start_weight_update_proof(); break;
            case AGGREGATION: m.start_aggregation(); break;
            case PROVER: m.start_prover(); break;
            case VERIFIER: m.start_verifier(); break;
        }
    }
    
    ~ScopedTimer() {
        auto& m = get_metrics();
        switch (component) {
            case CLIPPING: m.stop_clipping_proof(); break;
            case NOISE_GENERATION: m.stop_noise_generation_proof(); break;
            case NOISE_ADDITION: m.stop_noise_addition_proof(); break;
            case WEIGHT_UPDATE: m.stop_weight_update_proof(); break;
            case AGGREGATION: m.stop_aggregation(); break;
            case PROVER: m.stop_prover(); break;
            case VERIFIER: m.stop_verifier(); break;
        }
    }
    
private:
    Component component;
};

#define VERIDP_TIME_CLIPPING() veridp_metrics::ScopedTimer _timer_clipping(veridp_metrics::ScopedTimer::CLIPPING)
#define VERIDP_TIME_NOISE_GEN() veridp_metrics::ScopedTimer _timer_noise_gen(veridp_metrics::ScopedTimer::NOISE_GENERATION)
#define VERIDP_TIME_NOISE_ADD() veridp_metrics::ScopedTimer _timer_noise_add(veridp_metrics::ScopedTimer::NOISE_ADDITION)
#define VERIDP_TIME_WEIGHT_UPDATE() veridp_metrics::ScopedTimer _timer_weight(veridp_metrics::ScopedTimer::WEIGHT_UPDATE)
#define VERIDP_TIME_AGGREGATION() veridp_metrics::ScopedTimer _timer_agg(veridp_metrics::ScopedTimer::AGGREGATION)
#define VERIDP_TIME_PROVER() veridp_metrics::ScopedTimer _timer_prover(veridp_metrics::ScopedTimer::PROVER)
#define VERIDP_TIME_VERIFIER() veridp_metrics::ScopedTimer _timer_verifier(veridp_metrics::ScopedTimer::VERIFIER)

} // namespace veridp_metrics
