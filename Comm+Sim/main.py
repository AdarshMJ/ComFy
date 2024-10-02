import warnings
warnings.filterwarnings('ignore')
import argparse
import sys
import os
import torch
from torch_geometric.transforms import NormalizeFeatures, LargestConnectedComponents
from torch_geometric.utils import to_networkx,from_networkx,homophily
from torch_geometric.transforms import RandomNodeSplit
from model import GCN,GATv2, SimpleGCN
from dataloader import *
from nodeli import *
from tqdm import tqdm
import networkx as nx
import numpy as np
import time
import csv
from train import *
from arguments import parse_args
from matplotlib import pyplot as plt
#from comsim_onlysim2 import *
from comsim  import *
from onlysim import *
args = parse_args()


device = torch.device(args.device)
filename = args.out
p = args.dropout
lr = args.LR
hidden_dimension = args.hidden_dimension
splits = args.splits
seed = args.seed
#rewiringbudget = args.rewiringbudget
budget_edges_add = args.budget_edges_add
budget_edges_delete = args.budget_edges_delete
print(f"Loading the dataset...")


if args.dataset in ['Cora','Citeseer','Pubmed','CS','Physics','Computers','Photo']:
    data, num_classes,num_features = load_data(args.dataset,args.num_train,args.num_val)
    #print(data)

elif args.dataset in ['cornell.npz','texas.npz','wisconsin.npz']:
    path = '/home/heterophilous-graphs/data/'
    filepath = os.path.join(path, args.dataset)
    data = np.load(filepath)
    print("Converting to PyG dataset...")
    x = torch.tensor(data['node_features'], dtype=torch.float)
    y = torch.tensor(data['node_labels'], dtype=torch.long)
    edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
    train_mask = torch.tensor(data['train_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    val_mask = torch.tensor(data['val_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    test_mask = torch.tensor(data['test_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    num_classes = len(torch.unique(y))
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.num_classes = num_classes
    print(f"Selecting the LargestConnectedComponent..")
    transform = LargestConnectedComponents()
    data = transform(data)
    print()
    print("Splitting datasets train/val/test...")
    transform2 = RandomNodeSplit(split="train_rest",num_splits=100,num_test=0.2,num_val=0.2)
    data  = transform2(data)
    print(data)
    num_features = data.num_features
    num_classes = data.num_classes
    print("Done!..")

elif args.dataset in ['chameleon_filtered.npz','squirrel_filtered.npz','actor.npz']:
    path = '/home/heterophilous-graphs/data/'
    filepath = os.path.join(path, args.dataset)
    data = np.load(filepath)
    print("Converting to PyG dataset...")
    x = torch.tensor(data['node_features'], dtype=torch.float)
    y = torch.tensor(data['node_labels'], dtype=torch.long)
    edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
    train_mask = torch.tensor(data['train_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    val_mask = torch.tensor(data['val_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    test_mask = torch.tensor(data['test_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    num_classes = len(torch.unique(y))
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.num_classes = num_classes
    print(f"Selecting the LargestConnectedComponent..")
    transform = LargestConnectedComponents()
    data = transform(data)
    print("Splitting datasets train/val/test...")
    transform2 = RandomNodeSplit(split="train_rest", num_splits=100, num_test=0.2, num_val=0.2)
    data = transform2(data)
    print()
    print(data)
    num_features = data.num_features
    num_classes = data.num_classes
    print("Done!..")


num_train_nodes = data.train_mask.sum().item()
num_val_nodes = data.val_mask.sum().item()
num_test_nodes = data.test_mask.sum().item()
print()
print(f"Number of training nodes: {num_train_nodes/splits}")
print(f"Number of validation nodes: {num_val_nodes/splits}")
print(f"Number of test nodes: {num_test_nodes/splits}")
datasetname, _ = os.path.splitext(args.dataset)



## For Community+SimilarityBased Rewiring ####
algo_stime = time.time()
data,nmiscoremod_before = modify_graph(data,args.dataset,budget_edges_add,budget_edges_delete,seed)
algo_etime = time.time()
rewire_time =  algo_etime - algo_stime
print(f"Time Taken for Rewiring : {rewire_time}")


newG = to_networkx(data, to_undirected=True)
nxcg_G = nxcg.from_networkx(newG) 
communities_after = list(nx.community.louvain_communities(nxcg_G, seed=seed))
cluster_dict_after = {node: i for i, cluster in enumerate(communities_after) for node in cluster}
cluster_list_after = [cluster_dict_after[node] for node in range(len(data.y))]
nmiscoremod_after = NMI(cluster_list_after, data.y.cpu().numpy())






print("Calculating Edge Label Informativeness...")
graphaf, labelsaf = get_graph_and_labels_from_pyg_dataset(data)
edgeliaf = li_edge(graphaf, labelsaf)
print(f'Edge label informativeness: {edgeliaf:.4f}')

print("=============================================================")

print()


print("Calculating Full Graph Adjusted Homophily...")
hadjfull = h_adj(graphaf, labelsaf)
print(f'Full Graph Adjusted Homophily: {hadjfull:.4f}')
print()


data = data.to(device)
print()
print("Start Training...")
##=========================##=========================##=========================##=========================
if args.model == 'GCN':
  model = GCN(num_features,num_classes,hidden_dimension, num_layers=args.num_layers)

elif args.model == 'GATv2':
  model = GATv2(num_features,8, num_classes)

elif args.model == 'SimpleGCN':
  model = SimpleGCN(num_features,num_classes,hidden_dimension)

else:
  print("Invalid Model")
  sys.exit()

model.to(device)
print(model)

gcn_start = time.time()
finaltestacc,teststd,finalvalacc,valstd = train_and_get_results(data, model,p,lr,args.seed,args.splits)
gcn_end = time.time()


if args.dataset.endswith('.npz'):
    dataset_name = args.dataset.replace('.npz', '').replace('_filtered', '').capitalize()
else:
    dataset_name = args.dataset



headers = ['Dataset','AvgValAcc','DeviationVal','AvgTestAcc', 'Deviation','ELI',
           'AdjHom','NMIBefore','NMIAfter','EdgesAdded','EdgesDeleted','HiddenDim','LR','Dropout','GCNTime','RewireTime']

with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                    writer.writerow(headers)
            writer.writerow([args.dataset,f"{(finalvalacc):.2f}", f"{(valstd):.2f}",f"{(finaltestacc):.2f}", f"{(teststd):.2f}",
            edgeliaf,hadjfull,nmiscoremod_before,nmiscoremod_after,budget_edges_add,budget_edges_delete,hidden_dimension,lr,p,gcn_end-gcn_start,rewire_time])


