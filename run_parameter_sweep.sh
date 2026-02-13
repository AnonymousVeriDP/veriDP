#!/bin/bash

# VeriDP Parameter Sweep Script
# Tests different configurations: CNN vs MLP, clip norms, noise multipliers, etc.

OUTPUT_CSV="veridp_parameter_sweep.csv"

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

# Fixed batch count for parameter comparison
FIXED_BATCHES=20

# Parameters to sweep
MODELS=("cnn" "mlp")
CLIP_NORMS=(1 5 10 20 50)
NOISE_MULTIPLIERS=(0.5 1.0 1.1 2.0)
LEARNING_RATES=(0.001 0.01 0.1)

# Standardized CSV header (same as all other scripts)
CSV_HEADER="model,batches,clip_norm,noise_mult,learning_rate,samples,proofs,proof_size_kb,avg_proof_size_kb,prover_time_ms,clipping_time_ms,clipping_pct,noise_gen_time_ms,noise_gen_pct,noise_add_time_ms,noise_add_pct,weight_update_time_ms,weight_update_pct,aggregation_time_ms,aggregation_pct,other_time_ms,other_pct,verifier_time_ms,prover_verifier_ratio,prover_time_per_sample_ms,baseline_memory_mb,peak_memory_mb,final_memory_mb,memory_overhead_mb,epsilon,sampling_rate,wall_time_sec"

echo "$CSV_HEADER" > "$OUTPUT_CSV"

echo "=== VeriDP Parameter Sweep ==="
echo "Output will be saved to: $OUTPUT_CSV"
echo "Fixed batches: $FIXED_BATCHES"
echo ""

run_experiment() {
    local model=$1
    local clip=$2
    local sigma=$3
    local lr=$4
    
    # Build command
    local cmd="$EXECUTABLE $DATA_PATH --max-batches=$FIXED_BATCHES --clip=$clip --sigma=$sigma --lr=$lr"
    if [ "$model" == "mlp" ]; then
        cmd="$cmd --simple-mlp"
    fi
    
    echo -n "Running: model=$model, clip=$clip, sigma=$sigma, lr=$lr ... "
    
    # Run and capture output
    local start_time=$(date +%s)
    output=$($cmd 2>&1)
    local end_time=$(date +%s)
    local wall_time=$((end_time - start_time))
    
    # Extract metrics
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
    
    # Write to CSV (same order as header)
    echo "$model,$FIXED_BATCHES,$clip,$sigma,$lr,$samples,$proofs,$proof_size,$avg_proof_size,$prover_time,$clipping_time,$clipping_pct,$noise_gen_time,$noise_gen_pct,$noise_add_time,$noise_add_pct,$weight_time,$weight_pct,$aggregation_time,$aggregation_pct,$other_time,$other_pct,$verifier_time,$pv_ratio,$time_per_sample,$baseline_mem,$peak_mem,$final_mem,$mem_overhead,$epsilon,$sampling_rate,$wall_time" >> "$OUTPUT_CSV"
    
    echo "done (${wall_time}s, ε=$epsilon)"
}

# ============================================
# Experiment 1: CNN vs MLP comparison
# ============================================
echo ""
echo "=== Experiment 1: CNN vs MLP ==="
for model in "${MODELS[@]}"; do
    run_experiment "$model" 10 1.1 0.01
done

# ============================================
# Experiment 2: Clip norm sweep (CNN only)
# ============================================
echo ""
echo "=== Experiment 2: Clip Norm Sweep ==="
for clip in "${CLIP_NORMS[@]}"; do
    run_experiment "cnn" "$clip" 1.1 0.01
done

# ============================================
# Experiment 3: Noise multiplier sweep
# ============================================
echo ""
echo "=== Experiment 3: Noise Multiplier Sweep ==="
for sigma in "${NOISE_MULTIPLIERS[@]}"; do
    run_experiment "cnn" 10 "$sigma" 0.01
done

# ============================================
# Experiment 4: Learning rate sweep
# ============================================
echo ""
echo "=== Experiment 4: Learning Rate Sweep ==="
for lr in "${LEARNING_RATES[@]}"; do
    run_experiment "cnn" 10 1.1 "$lr"
done

echo ""
echo "=== Parameter Sweep Complete ==="
echo "Results saved to: $OUTPUT_CSV"
echo ""
echo "Total experiments run: $(( ${#MODELS[@]} + ${#CLIP_NORMS[@]} + ${#NOISE_MULTIPLIERS[@]} + ${#LEARNING_RATES[@]} ))"
