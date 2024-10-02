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
import methods
from rewiring import *
from rewiring.spectral_utils import spectral_gap
from dataloader import *
from nodeli import *
from tqdm import tqdm
import networkx as nx
import numpy as np
import random
import pickle
import time
import csv
from train import *

from sklearn.metrics.cluster import normalized_mutual_info_score as NMI

import nx_cugraph as nxcg

######### Hyperparams to use #############
#Cora --> Dropout = 0.4130296 ; LR = 0.01 ; Hidden_Dimension = 32
#Citeseer --> Dropout = 0.3130296 ; LR = 0.01 ; Hidden_Dimension = 32
#Pubmed --> Dropout = 0.4130296 ; LR = 0.01 ; Hidden_Dimension = 32
# Cornell = 0.4130296,0.001, 128
# Wisconsin = 0.5130296, 0.001,128
# Texas = 0.4130296,0.001,128
# Actor = 0.2130296,0.01,128
# ChameleonFiltered = 0.2130296,0.01,128
# ChameleonFilteredDirected = 0.4130296,0.01,128
# SquirrelFiltered = 0.5130296,0.01,128
# SquirrelFilteredDirected = 0.2130296,0.01,128
########################################



parser = argparse.ArgumentParser(description='Run NodeClassification+Rewiring script')
parser.add_argument('--method', type=str, help='Max/Min/Add/Delete/FoSR/SDRF')
parser.add_argument('--dataset', type=str, help='Dataset to download')
parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in GCN')
parser.add_argument('--model', type=str, default='SimpleGCN', choices=['GCN', 'GATv2','SimpleGCN','GCNWithDMoNAndGraphMod'], help='Model to use')
parser.add_argument('--num_heads', type=int, default=8, help='Number of heads in GATv2')
#parser.add_argument('--existing_graph', type=str,default=None, help='.pt file')
parser.add_argument('--out', type=str, help='name of log file')
parser.add_argument('--max_iters', type=int, default=10, help='maximum number of edge change iterations')
#parser.add_argument('--removal_bound', type=float, default=0.95, help='removal bound for SDRF')
#parser.add_argument('--tau', type=int, default=163, help='Temperature for SDRF')

parser.add_argument('--update_period', type=int, default=1, help='Times to recalculate criterion')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout = [Cora - 0.4130296, Citeseer - 0.3130296]')
parser.add_argument('--hidden_dimension', type=int, default=32, help='Hidden Dimension size')
parser.add_argument('--LR', type=float, default=0.01, help='Learning Rate = [0.01,0.001]')
parser.add_argument('--device',type=str,default='cuda',help='Device to use')
parser.add_argument('--seed',type=int,default=3164711608,help='Seed to use')
args, _ = parser.parse_known_args()
if args.method == 'community_rewiring' or args.method == 'inverse_community_rewiring':
    parser.add_argument('--comm_delete', type=int, default=100, help='fraction of inter-community edges to delete')
    parser.add_argument('--comm_add', type=int, default=0, help='fraction of edges to add relative to current edge count')


args = parser.parse_args()




device = torch.device(args.device)
filename = args.out

max_iterations = args.max_iters
update_period = args.update_period
initialgap = None
fgap = None
data_modifying = None
p = args.dropout
lr = args.LR
hidden_dimension = args.hidden_dimension
comm_delete = args.comm_delete
comm_add = args.comm_add
avg_testacc = []
avg_acc_testallsplits = []
trainacclist = []
trainallsplits = []
seed = args.seed


#het_val_seeds = [3164711608,894959334,2487307261,3349051410,493067366]

print(f"Loading the dataset...")



if args.dataset in ['Cora','Citeseer','Pubmed','CS','Physics','Computers','Photo']:
      data, num_classes,num_features = load_data(args.dataset)
      print(args.dataset)

  
elif args.dataset in ['cornell.npz','texas.npz','wisconsin.npz']:
    path = '/home/heterophilous-graphs/data/'
    filepath = os.path.join(path, args.dataset)
    data = np.load(filepath)
    print(args.dataset)
    print(data)
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
    transform2 = RandomNodeSplit(split="train_rest",num_splits=100,num_val=0.2,num_test=0.2)
    data  = transform2(data)
    print(data)
    num_features = data.num_features
    num_classes = data.num_classes
    print("Done!..")

elif args.dataset in ['chameleon_filtered.npz','squirrel_filtered.npz','actor.npz']:
    path = '/home/heterophilous-graphs/data/'
    filepath = os.path.join(path, args.dataset)
    data = np.load(filepath)
    print(args.dataset)
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
    transform2 = RandomNodeSplit(split="train_rest", num_splits=100, num_val=0.2, num_test=0.2)
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
print(f"Number of training nodes: {num_train_nodes/100}")
print(f"Number of validation nodes: {num_val_nodes/100}")
print(f"Number of test nodes: {num_test_nodes/100}")
datasetname, _ = os.path.splitext(args.dataset)
print()
nxgraph = to_networkx(data, to_undirected=True)
print(nxgraph)
initialgap, _, _, _ = spectral_gap(nxgraph)
print(f"InitialGap = {initialgap}")
print()


if args.method == 'proxydelmin':
  newdata,fgap, data_modifying, same_class_edges, diff_class_edges, nmiscoremod, same_class_same_community_after, same_class_diff_community_after, diff_class_same_community_after, diff_class_diff_community_after, same_class_same_community_before, same_class_diff_community_before, diff_class_same_community_before, diff_class_diff_community_before  = methods.proxydelmin(data, nxgraph, seed,args.max_iters)
  #data.edge_index = torch.cat([newdata.edge_index])
  #torch.save(data, f"{datasetname}_{args.method}_{args.max_iters}.pt")

elif args.method == 'proxydelmax':
  newdata,fgap, data_modifying, same_class_edges, diff_class_edges, nmiscoremod, same_class_same_community_after, same_class_diff_community_after, diff_class_same_community_after, diff_class_diff_community_after, same_class_same_community_before, same_class_diff_community_before, diff_class_same_community_before, diff_class_diff_community_before = methods.proxydelmax(data, nxgraph,seed, args.max_iters)
  #data.edge_index = torch.cat([newdata.edge_index])
  #torch.save(data, f"{datasetname}_{args.method}_{args.max_iters}.pt")

elif args.method == 'proxyaddmax':
  newdata,fgap, data_modifying, same_class_edges, diff_class_edges, nmiscoremod, same_class_same_community_after, same_class_diff_community_after, diff_class_same_community_after, diff_class_diff_community_after, same_class_same_community_before, same_class_diff_community_before, diff_class_same_community_before, diff_class_diff_community_before = methods.proxyaddmax(data, nxgraph,seed, args.max_iters)
  #data.edge_index = torch.cat([newdata.edge_index])
  #torch.save(data, f"{datasetname}_{args.method}_{args.max_iters}.pt")

elif args.method == 'proxyaddmin':
  newdata,fgap, data_modifying, same_class_edges, diff_class_edges, nmiscoremod, same_class_same_community_after, same_class_diff_community_after, diff_class_same_community_after, diff_class_diff_community_after, same_class_same_community_before, same_class_diff_community_before, diff_class_same_community_before, diff_class_diff_community_before = methods.proxyaddmin(data, nxgraph,seed, args.max_iters)
  #data.edge_index = torch.cat([newdata.edge_index])
  #torch.save(data, f"{datasetname}_{args.method}_{args.max_iters}.pt")

elif args.method == 'deleteadd':
  newdata,fgap,data_modifying = methods.proxydelmax(data, nxgraph, args.max_iters)
  data.edge_index = torch.cat([newdata.edge_index])
  newdata,fgap,data_modifying = methods.proxyaddmax(data, nxgraph, args.max_iters)
  data.edge_index = torch.cat([newdata.edge_index])
  #torch.save(data, f"{datasetname}_{args.method}_{args.max_iters}.pt")


elif args.method == 'fosr':
  newdata,fgap,data_modifying = methods.fosr(data, args.max_iters)
  data.edge_index = torch.cat([newdata.edge_index])
  #torch.save(data, f"{datasetname}_{args.method}_{args.max_iters}.pt")

elif args.method == 'random_delete':
  newdata,fgap,data_modifying,same_class_edges, diff_class_edges, nmiscoremod, same_class_same_community_after, same_class_diff_community_after, diff_class_same_community_after, diff_class_diff_community_after, same_class_same_community_before, same_class_diff_community_before, diff_class_same_community_before, diff_class_diff_community_before = methods.random_delete(data, seed, args.max_iters)
  data.edge_index = torch.cat([newdata.edge_index])
  print(data)
  #torch.save(data, f"{datasetname}_{args.method}_{args.max_iters}.pt")

elif args.method == 'random_add':
  newdata,fgap,data_modifying,same_class_edges, diff_class_edges, nmiscoremod, same_class_same_community_after, same_class_diff_community_after, diff_class_same_community_after, diff_class_diff_community_after, same_class_same_community_before, same_class_diff_community_before, diff_class_same_community_before, diff_class_diff_community_before = methods.random_add(data, seed, args.max_iters)
  data.edge_index = torch.cat([newdata.edge_index])
  print(data)
  #torch.save(data, f"{datasetname}_{args.method}_{args.max_iters}.pt")

elif args.method == 'community_rewiring':
  newdata,fgap,num_delete,edges_added,rewiring_time = methods.community_rewiring(data, seed, comm_delete, comm_add)
  data.edge_index = torch.cat([newdata.edge_index])
  print(data)
  

elif args.method == 'inverse_community_rewiring':
  newdata,fgap,num_delete,edges_added,rewiring_time = methods.inverse_community_rewiring(data, seed, comm_delete, comm_add)
  data.edge_index = torch.cat([newdata.edge_index])
  print(data)


else :
    print()
    print("Invalid Method. Training on the original graph...")

 
##=========================##=========================##=========================##=========================
print()

print("Final spectral gap",fgap)

print("Calculating Informativeness measures...")
graphaf, labelsaf = get_graph_and_labels_from_pyg_dataset(data)
edgeliaf = li_edge(graphaf, labelsaf)
hadjaf = h_adj(graphaf, labelsaf)
print(f'edge label informativeness: {edgeliaf:.4f}')
print(f'adjusted homophily: {hadjaf:.4f}')
print("=============================================================")
print("Done!")
print()

##=========================##=========================##=========================##=========================


newG = to_networkx(data, to_undirected=True)
nxcg_G = nxcg.from_networkx(newG) 
communities_after = list(nx.community.louvain_communities(nxcg_G, seed=seed))
cluster_dict_after = {node: i for i, cluster in enumerate(communities_after) for node in cluster}
cluster_list_after = [cluster_dict_after[node] for node in range(len(data.y))]
nmiscoremod_after = NMI(cluster_list_after, data.y.cpu().numpy())
print(f'NMI Score: {nmiscoremod_after:.4f}')

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


finaltestacc,finalvalacc,finaltrainacc = train_and_get_results(data, model,p,lr,args.seed)


avg_test_acc_after = np.mean(finaltestacc)
avg_val_acc_after = np.mean(finalvalacc)

sample_size = len(finaltestacc)
std_dev_after = 2 * np.std(finaltestacc)/(np.sqrt(sample_size))

sample_size_val = len(finalvalacc)
std_dev_after_val = 2 * np.std(finalvalacc)/(np.sqrt(sample_size_val))

print(f'Final test accuracy of all splits {(avg_test_acc_after):.2f} \u00B1 {(std_dev_after):.2f}')
print(f'Final validation accuracy of all splits {(avg_val_acc_after):.2f} \u00B1 {(std_dev_after_val):.2f}')
print()

#print("Deleting inter-class edges...")
#newdata = methods.PeerGNNDelete(data, 1000, pred)
#print(newdata)
#torch.save(newdata, f"Cora_PeerGNNDelete_2000.pt")
gcn_end = time.time()
#print(f"Time taken for training = {gcn_end - gcn_start}")

if args.dataset.endswith('.npz'):
    dataset_name = args.dataset.replace('.npz', '').replace('_filtered', '').capitalize()
else:
    dataset_name = args.dataset



headers = ['Method','Dataset','AvgValAcc','DeviationVal','AvgTestAcc', 'Deviation','ELI',
        'AdjHom','NMI',
        'EdgesAdded','EdgesDeleted','FinalGap','HiddenDim','LR','Dropout','RewiringTime','GCNTime']

with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                    writer.writerow(headers)
            writer.writerow([args.method,args.dataset,f"{(avg_val_acc_after):.2f}", f"{(std_dev_after_val):.2f}",f"{(avg_test_acc_after):.2f}", f"{(std_dev_after):.2f}",
            edgeliaf,hadjaf,nmiscoremod_after,
            edges_added,num_delete,fgap,hidden_dimension,lr,p,rewiring_time,gcn_end-gcn_start])


