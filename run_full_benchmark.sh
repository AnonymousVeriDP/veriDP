#!/bin/bash

# VeriDP Full Benchmark Suite
# Comprehensive benchmarking: batch scaling + parameter variations

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

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              VeriDP FULL BENCHMARK SUITE                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║ Results will be saved to: $OUTPUT_DIR/                       "
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Function to extract metrics and write to CSV
run_and_record() {
    local csv_file=$1
    local model=$2
    local batches=$3
    local clip=$4
    local sigma=$5
    local lr=$6
    
    # Build command
    local cmd="$EXECUTABLE $DATA_PATH --max-batches=$batches --clip=$clip --sigma=$sigma --lr=$lr"
    local model_flag=""
    if [ "$model" == "mlp" ]; then
        cmd="$cmd --simple-mlp"
        model_flag="--simple-mlp"
    fi
    
    echo -n "  Testing: model=$model, batches=$batches, clip=$clip, σ=$sigma, lr=$lr ... "
    
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
    
    local epsilon=$(echo "$output" | grep "Final ε:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    local sampling_rate=$(echo "$output" | grep "Sampling rate" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    
    # Write row to CSV
    echo "$model,$batches,$clip,$sigma,$lr,$samples,$proofs,$proof_size,$avg_proof_size,$prover_time,$clipping_time,$clipping_pct,$noise_gen_time,$noise_gen_pct,$noise_add_time,$noise_add_pct,$weight_time,$weight_pct,$aggregation_time,$aggregation_pct,$other_time,$other_pct,$verifier_time,$pv_ratio,$time_per_sample,$baseline_mem,$peak_mem,$final_mem,$mem_overhead,$epsilon,$sampling_rate,$wall_time" >> "$csv_file"
    
    echo "done (${wall_time}s, ε=$epsilon)"
}

# CSV header
CSV_HEADER="model,batches,clip_norm,noise_mult,learning_rate,samples,proofs,proof_size_kb,avg_proof_size_kb,prover_time_ms,clipping_time_ms,clipping_pct,noise_gen_time_ms,noise_gen_pct,noise_add_time_ms,noise_add_pct,weight_update_time_ms,weight_update_pct,aggregation_time_ms,aggregation_pct,other_time_ms,other_pct,verifier_time_ms,prover_verifier_ratio,prover_time_per_sample_ms,baseline_memory_mb,peak_memory_mb,final_memory_mb,memory_overhead_mb,epsilon,sampling_rate,wall_time_sec"

# ============================================
# BENCHMARK 1: Batch Scaling (CNN)
# ============================================
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  BENCHMARK 1: Batch Scaling (CNN)"
echo "════════════════════════════════════════════════════════════════"
CSV1="$OUTPUT_DIR/batch_scaling_cnn_$TIMESTAMP.csv"
echo "$CSV_HEADER" > "$CSV1"

for batches in 1 2 5 10 20 50 100; do
    run_and_record "$CSV1" "cnn" "$batches" 10 1.1 0.01
done
echo "Saved to: $CSV1"

# ============================================
# BENCHMARK 2: Batch Scaling (MLP)
# ============================================
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  BENCHMARK 2: Batch Scaling (MLP)"
echo "════════════════════════════════════════════════════════════════"
CSV2="$OUTPUT_DIR/batch_scaling_mlp_$TIMESTAMP.csv"
echo "$CSV_HEADER" > "$CSV2"

for batches in 1 2 5 10 20 50 100; do
    run_and_record "$CSV2" "mlp" "$batches" 10 1.1 0.01
done
echo "Saved to: $CSV2"

# ============================================
# BENCHMARK 3: CNN vs MLP Comparison
# ============================================
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  BENCHMARK 3: CNN vs MLP Comparison"
echo "════════════════════════════════════════════════════════════════"
CSV3="$OUTPUT_DIR/cnn_vs_mlp_$TIMESTAMP.csv"
echo "$CSV_HEADER" > "$CSV3"

for batches in 5 10 20 50; do
    run_and_record "$CSV3" "cnn" "$batches" 10 1.1 0.01
    run_and_record "$CSV3" "mlp" "$batches" 10 1.1 0.01
done
echo "Saved to: $CSV3"

# ============================================
# BENCHMARK 4: Clip Norm Sweep
# ============================================
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  BENCHMARK 4: Clip Norm Sweep"
echo "════════════════════════════════════════════════════════════════"
CSV4="$OUTPUT_DIR/clip_norm_sweep_$TIMESTAMP.csv"
echo "$CSV_HEADER" > "$CSV4"

for clip in 1 2 5 10 20 50; do
    run_and_record "$CSV4" "cnn" 20 "$clip" 1.1 0.01
done
echo "Saved to: $CSV4"

# ============================================
# BENCHMARK 5: Noise Multiplier Sweep
# ============================================
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  BENCHMARK 5: Noise Multiplier Sweep"
echo "════════════════════════════════════════════════════════════════"
CSV5="$OUTPUT_DIR/noise_mult_sweep_$TIMESTAMP.csv"
echo "$CSV_HEADER" > "$CSV5"

for sigma in 0.1 0.5 1.0 1.1 2.0 5.0; do
    run_and_record "$CSV5" "cnn" 20 10 "$sigma" 0.01
done
echo "Saved to: $CSV5"

# ============================================
# Summary
# ============================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              BENCHMARK COMPLETE                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║ Results saved to: $OUTPUT_DIR/"
echo "║"
echo "║ Files generated:"
echo "║   - batch_scaling_cnn_$TIMESTAMP.csv"
echo "║   - batch_scaling_mlp_$TIMESTAMP.csv"
echo "║   - cnn_vs_mlp_$TIMESTAMP.csv"
echo "║   - clip_norm_sweep_$TIMESTAMP.csv"
echo "║   - noise_mult_sweep_$TIMESTAMP.csv"
echo "╚══════════════════════════════════════════════════════════════╝"

