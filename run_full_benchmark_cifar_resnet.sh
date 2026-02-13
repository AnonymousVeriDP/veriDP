#!/bin/bash

# VeriDP Full Benchmark Suite (CIFAR-10 + ResNet18)
# Mirrors run_full_benchmark.sh but uses CIFAR-10 + ResNet18 only

OUTPUT_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Auto-detect executable location
if [ -f "./train_veridp" ]; then
    EXECUTABLE="./train_veridp"
    DATA_PATH="../data"
elif [ -f "./build/train_veridp" ]; then
    EXECUTABLE="./build/train_veridp"
    DATA_PATH="./data"
else
    echo "ERROR: Cannot find train_veridp executable"
    echo "Run this script from either:"
    echo "  - The project root (veriDP-test/)"
    echo "  - The build directory (veriDP-test/build/)"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "==============================================="
echo "        VeriDP FULL BENCHMARK (CIFAR-10)       "
echo "==============================================="
echo "Results will be saved to: $OUTPUT_DIR/"
echo ""

# Function to extract metrics and write to CSV
run_and_record() {
    local csv_file=$1
    local batches=$2
    local clip=$3
    local sigma=$4
    local lr=$5
    
    # Build command
    local cmd="$EXECUTABLE $DATA_PATH --dataset=cifar10 --model=resnet18 --max-batches=$batches --clip=$clip --sigma=$sigma --lr=$lr"
    
    echo -n "  Testing: batches=$batches, clip=$clip, sigma=$sigma, lr=$lr ... "
    
    # Run and capture output
    local start_time=$(date +%s)
    output=$($cmd 2>&1)
    local end_time=$(date +%s)
    local wall_time=$((end_time - start_time))
    
    # Extract all metrics
    local samples=$(echo "$output" | grep "Samples processed:" | grep -oE '[0-9]+' | tail -1)
    local proofs=$(echo "$output" | grep "Total proofs generated:" | grep -oE '[0-9]+' | tail -1)
    local proof_size=$(echo "$output" | grep "Final IVC proof size:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    local avg_proof_size=$(echo "$output" | grep "Avg batch proof:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    
    local prover_time=$(echo "$output" | grep "Total prover time:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    local clipping_time=$(echo "$output" | grep "Clipping proof:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    local clipping_pct=$(echo "$output" | grep "Clipping proof:" | grep -oE '[0-9]+\.[0-9]+' | sed -n '2p')
    local noise_gen_time=$(echo "$output" | grep "Noise generation:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    local noise_gen_pct=$(echo "$output" | grep "Noise generation:" | grep -oE '[0-9]+\.[0-9]+' | sed -n '2p')
    local noise_add_time=$(echo "$output" | grep "Noise addition:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    local noise_add_pct=$(echo "$output" | grep "Noise addition:" | grep -oE '[0-9]+\.[0-9]+' | sed -n '2p')
    local weight_time=$(echo "$output" | grep "Weight update:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    local weight_pct=$(echo "$output" | grep "Weight update:" | grep -oE '[0-9]+\.[0-9]+' | sed -n '2p')
    
    local aggregation_time=$(echo "$output" | grep "Aggregation:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    local aggregation_pct=$(echo "$output" | grep "Aggregation:" | grep -oE '[0-9]+\.[0-9]+' | sed -n '2p')
    
    local other_time=$(echo "$output" | grep "Other:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    local other_pct=$(echo "$output" | grep "Other:" | grep -oE '[0-9]+\.[0-9]+' | sed -n '2p')
    
    local verifier_time=$(echo "$output" | grep "Total verifier time:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    local pv_ratio=$(echo "$output" | grep "Prover/Verifier ratio:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    local time_per_sample=$(echo "$output" | grep "Prover time/sample:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    
    local baseline_mem=$(echo "$output" | grep "Baseline memory:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    local peak_mem=$(echo "$output" | grep "Peak memory:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    local final_mem=$(echo "$output" | grep "Final memory:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    local mem_overhead=$(echo "$output" | grep "Memory overhead:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    
    local epsilon=$(echo "$output" | grep -E "Final [Îε]:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    local sampling_rate=$(echo "$output" | grep "Sampling rate" | grep -oE '[0-9]+\.[0-9]+' | head -1)

    # Fallbacks for missing metrics
    for v in samples proofs proof_size avg_proof_size prover_time clipping_time clipping_pct \
             noise_gen_time noise_gen_pct noise_add_time noise_add_pct weight_time weight_pct \
             aggregation_time aggregation_pct other_time other_pct verifier_time pv_ratio \
             time_per_sample baseline_mem peak_mem final_mem mem_overhead epsilon sampling_rate; do
        if [ -z "${!v}" ]; then
            printf -v "$v" "NA"
        fi
    done
    
    # Write row to CSV
    echo "resnet18,$batches,$clip,$sigma,$lr,$samples,$proofs,$proof_size,$avg_proof_size,$prover_time,$clipping_time,$clipping_pct,$noise_gen_time,$noise_gen_pct,$noise_add_time,$noise_add_pct,$weight_time,$weight_pct,$aggregation_time,$aggregation_pct,$other_time,$other_pct,$verifier_time,$pv_ratio,$time_per_sample,$baseline_mem,$peak_mem,$final_mem,$mem_overhead,$epsilon,$sampling_rate,$wall_time" >> "$csv_file"
    
    echo "done (${wall_time}s, Îµ=$epsilon)"
}

# CSV header
CSV_HEADER="model,batches,clip_norm,noise_mult,learning_rate,samples,proofs,proof_size_kb,avg_proof_size_kb,prover_time_ms,clipping_time_ms,clipping_pct,noise_gen_time_ms,noise_gen_pct,noise_add_time_ms,noise_add_pct,weight_update_time_ms,weight_update_pct,aggregation_time_ms,aggregation_pct,other_time_ms,other_pct,verifier_time_ms,prover_verifier_ratio,prover_time_per_sample_ms,baseline_memory_mb,peak_memory_mb,final_memory_mb,memory_overhead_mb,epsilon,sampling_rate,wall_time_sec"

# ============================================
# BENCHMARK 1: Batch Scaling (CIFAR-10 + ResNet)
# ============================================
echo ""
echo "==============================================="
echo "  BENCHMARK 1: Batch Scaling (ResNet18)"
echo "==============================================="
CSV1="$OUTPUT_DIR/cifar10_resnet_batch_scaling_$TIMESTAMP.csv"
echo "$CSV_HEADER" > "$CSV1"

# for batches in 1 2 5 10 20 50 100; do
#     run_and_record "$CSV1" "$batches" 10 1.1 0.01
# done
echo "Saved to: $CSV1"

# ============================================
# BENCHMARK 2: Clip Norm Sweep
# ============================================
echo ""
echo "==============================================="
echo "  BENCHMARK 2: Clip Norm Sweep (ResNet18)"
echo "==============================================="
CSV2="$OUTPUT_DIR/cifar10_resnet_clip_norm_sweep_$TIMESTAMP.csv"
echo "$CSV_HEADER" > "$CSV2"

for clip in 1 2 5 10 20 50; do
    run_and_record "$CSV2" 20 "$clip" 1.1 0.01
done
echo "Saved to: $CSV2"

# ============================================
# BENCHMARK 3: Noise Multiplier Sweep
# ============================================
echo ""
echo "==============================================="
echo "  BENCHMARK 3: Noise Multiplier Sweep (ResNet18)"
echo "==============================================="
CSV3="$OUTPUT_DIR/cifar10_resnet_noise_mult_sweep_$TIMESTAMP.csv"
echo "$CSV_HEADER" > "$CSV3"

for sigma in 0.1 0.5 1.0 1.1 2.0 5.0; do
    run_and_record "$CSV3" 20 10 "$sigma" 0.01
done
echo "Saved to: $CSV3"

# ============================================
# BENCHMARK 4: Learning Rate Sweep
# ============================================
echo ""
echo "==============================================="
echo "  BENCHMARK 4: Learning Rate Sweep (ResNet18)"
echo "==============================================="
CSV4="$OUTPUT_DIR/cifar10_resnet_lr_sweep_$TIMESTAMP.csv"
echo "$CSV_HEADER" > "$CSV4"

for lr in 0.001 0.01 0.1; do
    run_and_record "$CSV4" 20 10 1.1 "$lr"
done
echo "Saved to: $CSV4"

# ============================================
# Summary
# ============================================
echo ""
echo "==============================================="
echo "             BENCHMARK COMPLETE                "
echo "==============================================="
echo "Results saved to: $OUTPUT_DIR/"
echo "Files generated:"
echo "  - cifar10_resnet_batch_scaling_$TIMESTAMP.csv"
echo "  - cifar10_resnet_clip_norm_sweep_$TIMESTAMP.csv"
echo "  - cifar10_resnet_noise_mult_sweep_$TIMESTAMP.csv"
echo "  - cifar10_resnet_lr_sweep_$TIMESTAMP.csv"
