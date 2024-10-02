#!/bin/bash

# Define the order and parameters for each dataset
datasets=(
    "Cora:0.01:0.41:32:0,10,50,100,500:10,50,100,500"
    "Citeseer:0.01:0.31:32:0,10,50,100,500:10,50,100,500"
    "Pubmed:0.01:0.31:32:0,10,50,100,500:10,50,100,500"
    "cornell.npz:0.001:0.4:128:0,5,10,20,50,100:5,10,20,50,100"
    "texas.npz:0.001:0.4:128:0,5,10,20,50,100:5,10,20,50,100"
    "wisconsin.npz:0.001:0.4:128:0,5,10,20,50,100:5,10,20,50,100"
    "chameleon_filtered.npz:0.001:0.21:128:0,20,50,100,500:20,50,100,500,1000"
    "squirrel_filtered.npz:0.001:0.51:128:0,20,50,100,500:20,50,100,500,1000"
    "actor.npz:0.001:0.51:128:0,20,50,100,500:20,50,100,500,1000"
)

methods=(
    "proxyaddmax"
    "proxyaddmin"
    "proxydelmax"
    "proxydelmin"
)

# Define a single output file for all results
output_file="allspectralresults.csv"

echo "Starting script execution"
echo "Datasets to process: ${datasets[@]%%:*}"

for dataset_info in "${datasets[@]}"; do
    IFS=':' read -r dataset lr dropout hidden_dim add_iters_str del_iters_str <<< "$dataset_info"
    echo "Processing dataset: $dataset"
    echo "  Parameters: LR=$lr, dropout=$dropout, hidden_dim=$hidden_dim"
    
    IFS=',' read -ra add_iters_values <<< "$add_iters_str"
    IFS=',' read -ra del_iters_values <<< "$del_iters_str"
    
    echo "  add_iters values: ${add_iters_values[@]}"
    echo "  del_iters values: ${del_iters_values[@]}"
    
    for method in "${methods[@]}"; do
        echo "  Running method: $method"
        if [[ $method == *"add"* ]]; then
            iters_values=("${add_iters_values[@]}")
        else
            iters_values=("${del_iters_values[@]}")
        fi
        
        for max_iters in "${iters_values[@]}"; do
            echo "    Running with max_iters=$max_iters"
            python main.py \
                --dataset "$dataset" \
                --out "$output_file" \
                --model SimpleGCN \
                --LR $lr \
                --hidden_dimension $hidden_dim \
                --dropout $dropout \
                --max_iters $max_iters \
                --method $method \
                2>&1 | tee -a debug_output.log
        done
    done
done

echo "Script execution completed"