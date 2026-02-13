#!/bin/bash

# VeriDP CIFAR-10 + ResNet Benchmark Script
# Mirrors run_benchmark.sh but uses CIFAR-10 + ResNet18

OUTPUT_CSV="veridp_cifar10_resnet_benchmark_results.csv"

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

# Batch counts to test
BATCH_COUNTS=(1 2 5 10 20 50 100 200 500)

# Standardized CSV header (same as all other scripts)
CSV_HEADER="model,batches,clip_norm,noise_mult,learning_rate,samples,proofs,proof_size_kb,avg_proof_size_kb,prover_time_ms,clipping_time_ms,clipping_pct,noise_gen_time_ms,noise_gen_pct,noise_add_time_ms,noise_add_pct,weight_update_time_ms,weight_update_pct,aggregation_time_ms,aggregation_pct,other_time_ms,other_pct,verifier_time_ms,prover_verifier_ratio,prover_time_per_sample_ms,baseline_memory_mb,peak_memory_mb,final_memory_mb,memory_overhead_mb,epsilon,sampling_rate,wall_time_sec"

echo "$CSV_HEADER" > "$OUTPUT_CSV"

echo "=== VeriDP CIFAR-10 + ResNet Benchmark ==="
echo "Output will be saved to: $OUTPUT_CSV"
echo ""

# Default parameters
MODEL="resnet18"
CLIP=10
SIGMA=1.1
LR=0.01

for batches in "${BATCH_COUNTS[@]}"; do
    echo -n "Running with --max-batches=$batches ... "
    
    # Run train_veridp and capture output
    start_time=$(date +%s)
    output=$($EXECUTABLE $DATA_PATH --dataset=cifar10 --model=$MODEL --max-batches=$batches --clip=$CLIP --sigma=$SIGMA --lr=$LR 2>&1)
    end_time=$(date +%s)
    wall_time=$((end_time - start_time))
    
    # Extract metrics
    samples=$(echo "$output" | grep "Samples processed:" | grep -oE '[0-9]+' | tail -1)
    proofs=$(echo "$output" | grep "Total proofs generated:" | grep -oE '[0-9]+' | tail -1)
    proof_size=$(echo "$output" | grep "Final IVC proof size:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    avg_proof_size=$(echo "$output" | grep "Avg batch proof:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    
    prover_time=$(echo "$output" | grep "Total prover time:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    
    clipping_time=$(echo "$output" | grep "Clipping proof:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    clipping_pct=$(echo "$output" | grep "Clipping proof:" | grep -oE '[0-9]+\.[0-9]+' | sed -n '2p')
    
    noise_gen_time=$(echo "$output" | grep "Noise generation:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    noise_gen_pct=$(echo "$output" | grep "Noise generation:" | grep -oE '[0-9]+\.[0-9]+' | sed -n '2p')
    
    noise_add_time=$(echo "$output" | grep "Noise addition:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    noise_add_pct=$(echo "$output" | grep "Noise addition:" | grep -oE '[0-9]+\.[0-9]+' | sed -n '2p')
    
    weight_time=$(echo "$output" | grep "Weight update:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    weight_pct=$(echo "$output" | grep "Weight update:" | grep -oE '[0-9]+\.[0-9]+' | sed -n '2p')
    
    aggregation_time=$(echo "$output" | grep "Aggregation:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    aggregation_pct=$(echo "$output" | grep "Aggregation:" | grep -oE '[0-9]+\.[0-9]+' | sed -n '2p')
    
    other_time=$(echo "$output" | grep "Other:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    other_pct=$(echo "$output" | grep "Other:" | grep -oE '[0-9]+\.[0-9]+' | sed -n '2p')
    
    verifier_time=$(echo "$output" | grep "Total verifier time:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    pv_ratio=$(echo "$output" | grep "Prover/Verifier ratio:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    time_per_sample=$(echo "$output" | grep "Prover time/sample:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    
    baseline_mem=$(echo "$output" | grep "Baseline memory:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    peak_mem=$(echo "$output" | grep "Peak memory:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    final_mem=$(echo "$output" | grep "Final memory:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    mem_overhead=$(echo "$output" | grep "Memory overhead:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    
    epsilon=$(echo "$output" | grep -E "Final [Îε]:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
    sampling_rate=$(echo "$output" | grep "Sampling rate" | grep -oE '[0-9]+\.[0-9]+' | head -1)

    # Fallbacks for missing metrics
    for v in samples proofs proof_size avg_proof_size prover_time clipping_time clipping_pct \
             noise_gen_time noise_gen_pct noise_add_time noise_add_pct weight_time weight_pct \
             aggregation_time aggregation_pct other_time other_pct verifier_time pv_ratio \
             time_per_sample baseline_mem peak_mem final_mem mem_overhead epsilon sampling_rate; do
        if [ -z "${!v}" ]; then
            printf -v "$v" "NA"
        fi
    done
    
    # Write to CSV (same order as header)
    echo "$MODEL,$batches,$CLIP,$SIGMA,$LR,$samples,$proofs,$proof_size,$avg_proof_size,$prover_time,$clipping_time,$clipping_pct,$noise_gen_time,$noise_gen_pct,$noise_add_time,$noise_add_pct,$weight_time,$weight_pct,$aggregation_time,$aggregation_pct,$other_time,$other_pct,$verifier_time,$pv_ratio,$time_per_sample,$baseline_mem,$peak_mem,$final_mem,$mem_overhead,$epsilon,$sampling_rate,$wall_time" >> "$OUTPUT_CSV"
    
    echo "done (${wall_time}s, Îµ=$epsilon)"
done

echo ""
echo "=== Benchmark Complete ==="
echo "Results saved to: $OUTPUT_CSV"
