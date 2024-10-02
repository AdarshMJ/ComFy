#!/bin/bash

# Define the order and parameters for each dataset
datasets=(
    "Cora:0.01:0.41:32:10,50,100,500,1000:10,50,100,500,1000"
    "Citeseer:0.01:0.31:32:10,50,100,500,1000:10,50,100,500,1000"
    "Pubmed:0.01:0.31:32:10,50,100,500,1000:10,50,100,500,1000"
    "cornell.npz:0.001:0.4:128:5,10,20,50,100:5,10,20,50,100"
    "texas.npz:0.001:0.4:128:5,10,20,50,100:5,10,20,50,100"
    "wisconsin.npz:0.001:0.4:128:5,10,20,50,100:5,10,20,50,100"
    "chameleon_filtered.npz:0.001:0.21:128:20,50,100,500,1000:20,50,100,500,1000"
    "squirrel_filtered.npz:0.001:0.51:128:20,50,100,500,1000:20,50,100,500,1000"
    "actor.npz:0.001:0.51:128:20,50,100,500,1000:20,50,100,500,1000"
)

# Define the output CSV file
output_csv="inversecommunityrewiringresultsboth.csv"

echo "Starting script execution"
echo "Datasets to process: ${datasets[@]%%:*}"

for dataset_info in "${datasets[@]}"; do
    IFS=':' read -r dataset lr dropout hidden_dim comm_delete_str comm_add_str <<< "$dataset_info"
    echo "Processing dataset: $dataset"
    echo "  Parameters: LR=$lr, dropout=$dropout, hidden_dim=$hidden_dim"
    
    IFS=',' read -ra comm_delete_values <<< "$comm_delete_str"
    IFS=',' read -ra comm_add_values <<< "$comm_add_str"
    
    echo "  comm_delete values: ${comm_delete_values[@]}"
    echo "  comm_add values: ${comm_add_values[@]}"
    
    for comm_delete in "${comm_delete_values[@]}"; do
        for comm_add in "${comm_add_values[@]}"; do
            echo "    Running with comm_delete=$comm_delete, comm_add=$comm_add"
            python main.py \
                --dataset "$dataset" \
                --out "$output_csv" \
                --model SimpleGCN \
                --LR $lr \
                --hidden_dimension $hidden_dim \
                --dropout $dropout \
                --comm_delete $comm_delete \
                --comm_add $comm_add \
                --method inverse_community_rewiring \
                2>&1 | tee -a debug_output.log
        done
    done
done

echo "Script execution completed"

