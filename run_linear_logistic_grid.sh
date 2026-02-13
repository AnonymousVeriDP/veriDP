#!/bin/bash

# VeriDP synthetic model grid benchmark:
# - Models: linear, logistic
# - Batch sizes: 2,5,10,20,30,50
# - Dataset sizes: 100,250,500,1000,2000
# - Fixed feature dimension: 16
#
# Records performance metrics to CSV.

OUTPUT_CSV="veridp_linear_logistic_grid_results.csv"

# Auto-detect executable location
if [ -f "./train_veridp" ]; then
    EXECUTABLE="./train_veridp"
    DATA_PATH="."
elif [ -f "./build/train_veridp" ]; then
    EXECUTABLE="./build/train_veridp"
    DATA_PATH="."
else
    echo "ERROR: Cannot find train_veridp executable"
    echo "Run this script from the project root (veriDP-test/)"
    exit 1
fi

MODELS=("linear" "logistic")
BATCHES=(2 5 10 20 30 50)
DATASET_SIZES=(100 250 500 1000 2000)

DIM=16
CLIP=10
LR=0.1

# Keep logistic with DP noise, and also enable DP noise for linear.
LINEAR_SIGMA=1.0
LOGISTIC_SIGMA=1.0

# Dataset-generation noise controls.
# Linear dataset generation noise is disabled.
LINEAR_NOISE_RANGE=0
LINEAR_NOISE_INC=0.001
# Logistic dataset generation keeps synthetic noise.
LOGISTIC_NOISE_RANGE=10
LOGISTIC_NOISE_INC=0.001

CSV_HEADER="model,dataset_size,feature_dim,batch_size,max_steps,clip_norm,noise_mult,learning_rate,samples,proofs,proof_size_kb,avg_proof_size_kb,prover_time_ms,clipping_time_ms,clipping_pct,noise_gen_time_ms,noise_gen_pct,noise_add_time_ms,noise_add_pct,weight_update_time_ms,weight_update_pct,aggregation_time_ms,aggregation_pct,other_time_ms,other_pct,verifier_time_ms,prover_verifier_ratio,prover_time_per_sample_ms,baseline_memory_mb,peak_memory_mb,final_memory_mb,memory_overhead_mb,epsilon,sampling_rate,wall_time_sec"
echo "$CSV_HEADER" > "$OUTPUT_CSV"

echo "=== VeriDP Linear/Logistic Grid Benchmark ==="
echo "Output: $OUTPUT_CSV"
echo ""

run_case() {
    local model=$1
    local dataset_size=$2
    local batch_size=$3
    local sigma=$4

    # Process approximately one full pass over dataset.
    local max_steps=$(( (dataset_size + batch_size - 1) / batch_size ))

    local cmd="$EXECUTABLE $DATA_PATH --dataset=$model --model=$model --batch=$batch_size --max-steps=$max_steps --clip=$CLIP --lr=$LR --sigma=$sigma"
    if [ "$model" = "linear" ]; then
        cmd="$cmd --linear-size=$dataset_size --linear-dim=$DIM --linear-noise-range=$LINEAR_NOISE_RANGE --linear-noise-increment=$LINEAR_NOISE_INC"
    else
        cmd="$cmd --logistic-size=$dataset_size --logistic-dim=$DIM --logistic-noise-range=$LOGISTIC_NOISE_RANGE --logistic-noise-increment=$LOGISTIC_NOISE_INC"
    fi

    echo -n "Running model=$model size=$dataset_size batch=$batch_size steps=$max_steps sigma=$sigma ... "

    local start_time=$(date +%s)
    local output
    output=$($cmd 2>&1)
    local end_time=$(date +%s)
    local wall_time=$((end_time - start_time))

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

    local epsilon=$(echo "$output" | sed -nE 's/.*Final [Îε]:[[:space:]]*([0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?|inf|nan|INF|NAN).*/\1/p' | head -1)
    local sampling_rate=$(echo "$output" | sed -nE 's/.*Sampling rate[^0-9eE.+-]*([0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?).*/\1/p' | head -1)

    # Fallbacks for parse misses so rows are always written.
    for v in samples proofs proof_size avg_proof_size prover_time clipping_time clipping_pct \
             noise_gen_time noise_gen_pct noise_add_time noise_add_pct weight_time weight_pct \
             aggregation_time aggregation_pct other_time other_pct verifier_time pv_ratio \
             time_per_sample baseline_mem peak_mem final_mem mem_overhead epsilon sampling_rate; do
        if [ -z "${!v}" ]; then
            printf -v "$v" "NA"
        fi
    done

    echo "$model,$dataset_size,$DIM,$batch_size,$max_steps,$CLIP,$sigma,$LR,$samples,$proofs,$proof_size,$avg_proof_size,$prover_time,$clipping_time,$clipping_pct,$noise_gen_time,$noise_gen_pct,$noise_add_time,$noise_add_pct,$weight_time,$weight_pct,$aggregation_time,$aggregation_pct,$other_time,$other_pct,$verifier_time,$pv_ratio,$time_per_sample,$baseline_mem,$peak_mem,$final_mem,$mem_overhead,$epsilon,$sampling_rate,$wall_time" >> "$OUTPUT_CSV"

    echo "done (${wall_time}s, eps=$epsilon)"
}

for model in "${MODELS[@]}"; do
    if [ "$model" = "linear" ]; then
        sigma="$LINEAR_SIGMA"
    else
        sigma="$LOGISTIC_SIGMA"
    fi

    for dataset_size in "${DATASET_SIZES[@]}"; do
        for batch_size in "${BATCHES[@]}"; do
            run_case "$model" "$dataset_size" "$batch_size" "$sigma"
        done
    done
done

echo ""
echo "=== Grid benchmark complete ==="
echo "Results saved to: $OUTPUT_CSV"
