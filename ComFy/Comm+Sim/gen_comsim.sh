#!/bin/bash

# Define the order and parameters for each dataset
datasets=(
    "Cora:0.01:0.41:128:50,100,500,1000:100,500,1000,1500,2000"
    "Citeseer:0.01:0.31:32:50,100,500,1000:100,500,1000,1500,2000"
    "Pubmed:0.01:0.31:32:50,100,500,1000:100,500,1000,1500,2000"
    "cornell.npz:0.01:0.51:128:5,10,50,100:5,10,50,100"
    "texas.npz:0.001:0.51:32:5,10,50,100:5,10,50,100"
    "wisconsin.npz:0.001:0.51:128:5,10,50,100:5,10,50,100"
    "chameleon_filtered.npz:0.001:0.21:128:5,10,50,100,500:50,100,500,1000,1500,2000"
    "squirrel_filtered.npz:0.001:0.51:128:5,10,50,100,500:50,100,500,1000,1500,2000"
    "actor.npz:0.001:0.51:128:5,10,50,100,500:50,100,500,1000,1500,2000"

)

echo "Starting script execution"
echo "Datasets to process: ${datasets[@]%%:*}"

# Define the output CSV file
output_csv="resultscomsim.csv"

# Write the headers to the output CSV file only if it doesn't exist
if [ ! -f "$output_csv" ]; then
    echo "Dataset,AvgValAcc,DeviationVal,AvgTestAcc,Deviation,ELI,AdjHom,NMIBefore,NMIAfter,EdgesAdded,EdgesDeleted,HiddenDim,LR,Dropout,GCNTime,RewireTime" > "$output_csv"
fi

for dataset_info in "${datasets[@]}"; do
    IFS=':' read -r dataset lr dropout hidden_dim add_iters_str del_iters_str <<< "$dataset_info"
    echo "Processing dataset: $dataset"
    echo "  Parameters: LR=$lr, dropout=$dropout, hidden_dim=$hidden_dim"
    
    IFS=',' read -ra add_iters_values <<< "$add_iters_str"
    IFS=',' read -ra del_iters_values <<< "$del_iters_str"
    
    echo "  add_iters values: ${add_iters_values[@]}"
    echo "  del_iters values: ${del_iters_values[@]}"
    
    for add_budget in "${add_iters_values[@]}"; do
        echo "    Running with add_budget=$add_budget"
        python main.py \
            --dataset "$dataset" \
            --out "$output_csv" \
            --model SimpleGCN \
            --LR $lr \
            --hidden_dimension $hidden_dim \
            --dropout $dropout \
            --budget_edges_add $add_budget \
            --budget_edges_delete 0 \
            2>&1 | tee -a debug_output.log
    done
    
    for del_budget in "${del_iters_values[@]}"; do
        echo "    Running with del_budget=$del_budget"
        python main.py \
            --dataset "$dataset" \
            --out "$output_csv" \
            --model SimpleGCN \
            --LR $lr \
            --hidden_dimension $hidden_dim \
            --dropout $dropout \
            --budget_edges_add 0 \
            --budget_edges_delete $del_budget \
            2>&1 | tee -a debug_output.log
    done
done

echo "Script execution completed"