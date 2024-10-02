import warnings
warnings.filterwarnings('ignore')
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
import time
import csv
from train import *
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from arguments import *



args = parse_args()
device = torch.device(args.device)
filename = args.out
graphfile = args.existing_graph
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
      data, num_classes,num_features = load_data(args.dataset,args.num_train,args.num_val)
      print(data)

elif args.dataset in ['cornell.npz','texas.npz','wisconsin.npz']:
    path = '/home/heterophilous-graphs/data/'
    filepath = os.path.join(path, args.dataset)
    data = np.load(filepath)
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
    #transform2 = RandomNodeSplit(split="test_rest",num_splits=100,num_test =0.2,num_val = 0.2)
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
    print("Splitting datasets train/val/test...")
    #transform2 = RandomNodeSplit(split="test_rest", num_splits=100, num_train_per_class=args.num_train, num_val=args.num_val)
    transform2 = RandomNodeSplit(split="train_rest",num_splits=100,num_test=0.2,num_val=0.2)
    data = transform2(data)
    print()
    num_features = data.num_features
    num_classes = data.num_classes
    print("Done!..")
    
num_train_nodes = data.train_mask.sum().item()
num_val_nodes = data.val_mask.sum().item()
num_test_nodes = data.test_mask.sum().item()
print()
print(f"Number of training nodes: {num_train_nodes/args.splits}")
print(f"Number of validation nodes: {num_val_nodes/args.splits}")
print(f"Number of test nodes: {num_test_nodes/args.splits}")

datasetname, _ = os.path.splitext(args.dataset)
print()
nxgraph = to_networkx(data, to_undirected=True)
print(nxgraph)
initialgap, _, _, _ = spectral_gap(nxgraph)
print(f"InitialGap = {initialgap}")
print()


if args.method == 'proxydelmin':
  newdata,fgap, data_modifying, same_class_edges, diff_class_edges, nmiscoremod, same_class_same_community_after, same_class_diff_community_after, diff_class_same_community_after, diff_class_diff_community_after, same_class_same_community_before, same_class_diff_community_before, diff_class_same_community_before, diff_class_diff_community_before  = methods.proxydelmin(data, nxgraph, seed,args.max_iters)
  data.edge_index = torch.cat([newdata.edge_index])
  #torch.save(data, f"{datasetname}_{args.method}_{args.max_iters}.pt")

elif args.method == 'proxydelmax':
  newdata,fgap, data_modifying, same_class_edges, diff_class_edges, nmiscoremod, same_class_same_community_after, same_class_diff_community_after, diff_class_same_community_after, diff_class_diff_community_after, same_class_same_community_before, same_class_diff_community_before, diff_class_same_community_before, diff_class_diff_community_before = methods.proxydelmax(data, nxgraph,seed, args.max_iters)
  data.edge_index = torch.cat([newdata.edge_index])
  #torch.save(data, f"{datasetname}_{args.method}_{args.max_iters}.pt")

elif args.method == 'proxyaddmax':
  newdata,fgap, data_modifying, same_class_edges, diff_class_edges, nmiscoremod, same_class_same_community_after, same_class_diff_community_after, diff_class_same_community_after, diff_class_diff_community_after, same_class_same_community_before, same_class_diff_community_before, diff_class_same_community_before, diff_class_diff_community_before = methods.proxyaddmax(data, nxgraph,seed, args.max_iters)
  data.edge_index = torch.cat([newdata.edge_index])
  #torch.save(data, f"{datasetname}_{args.method}_{args.max_iters}.pt")

elif args.method == 'proxyaddmin':
  newdata,fgap, data_modifying, same_class_edges, diff_class_edges, nmiscoremod, same_class_same_community_after, same_class_diff_community_after, diff_class_same_community_after, diff_class_diff_community_after, same_class_same_community_before, same_class_diff_community_before, diff_class_same_community_before, diff_class_diff_community_before = methods.proxyaddmin(data, nxgraph,seed, args.max_iters)
  data.edge_index = torch.cat([newdata.edge_index])
  #torch.save(data, f"{datasetname}_{args.method}_{args.max_iters}.pt")

elif args.method == 'deleteadd':
  newdata,fgap,data_modifying = methods.proxydelmax(data, nxgraph, args.max_iters)
  data.edge_index = torch.cat([newdata.edge_index])
  newdata,fgap,data_modifying = methods.proxyaddmax(data, nxgraph, args.max_iters)
  data.edge_index = torch.cat([newdata.edge_index])
  #torch.save(data, f"{datasetname}_{args.method}_{args.max_iters}.pt")


elif args.method == 'fosr':
  newdata,fgap,data_modifying,same_class_edges, diff_class_edges, nmiscoremod, same_class_same_community_after, same_class_diff_community_after, diff_class_same_community_after, diff_class_diff_community_after, same_class_same_community_before, same_class_diff_community_before, diff_class_same_community_before, diff_class_diff_community_before = methods.fosr(data, args.max_iters)
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
  newdata,fgap,data_modifying,same_class_edges, diff_class_edges, nmiscoremod, same_class_same_community_after, same_class_diff_community_after, diff_class_same_community_after, diff_class_diff_community_after, same_class_same_community_before, same_class_diff_community_before, diff_class_same_community_before, diff_class_diff_community_before = methods.community_rewiring(data, seed, args.max_iters, comm_delete, comm_add)
  data.edge_index = torch.cat([newdata.edge_index])
  print(data)
  #torch.save(data, f"{datasetname}_{args.method}_{args.max_iters}.pt")

elif args.method == 'borf':
   borf_edge_index = torch.load("borfgraphs/cora/iters_3_add_20_remove_10_edge_index_0.pt")


elif args.method == 'none':
  print()
  print("Training on the original graph...")

 
##=========================##=========================##=========================##=========================
print()

print("Final spectral gap",fgap)

print("Calculating Informativeness measures...")
graphaf, labelsaf = get_graph_and_labels_from_pyg_dataset(data.detach().cpu())
edgeliaf = li_edge(graphaf, labelsaf)
hadjaf = h_adj(graphaf, labelsaf)
print(f'edge label informativeness: {edgeliaf:.4f}')
print(f'adjusted homophily: {hadjaf:.4f}')
print("=============================================================")
print("Done!")
print()

##=========================##=========================##=========================##=========================
# print("Calculating Edge Signal to Noise Ratio...")
# esnr_score = esnr_vanilla(data.to(device))
# print(f'Edge Signal to Noise Ratio: {esnr_score:.4f}')
# print("Done!")
# print()


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


finaltestacc,finalteststd,finalvalacc,finalvalstd = train_and_get_results(data, model,p,lr,args.seed,args.splits)


# avg_test_acc_after = np.mean(finaltestacc)
# avg_val_acc_after = np.mean(finalvalacc)

# sample_size = len(finaltestacc)
# std_dev_after = 2 * np.std(finaltestacc)/(np.sqrt(sample_size))

# sample_size_val = len(finalvalacc)
# std_dev_after_val = 2 * np.std(finalvalacc)/(np.sqrt(sample_size_val))

# print(f'Final test accuracy of all splits {(avg_test_acc_after):.2f} \u00B1 {(std_dev_after):.2f}')
# print(f'Final validation accuracy of all splits {(avg_val_acc_after):.2f} \u00B1 {(std_dev_after_val):.2f}')
# print()

gcn_end = time.time()

if args.dataset.endswith('.npz'):
    dataset_name = args.dataset.replace('.npz', '').replace('_filtered', '').capitalize()
else:
    dataset_name = args.dataset


if args.method == 'none':
    headers = ['Method','Dataset','AvgValAcc','ValDeviation','AvgTestAcc', 'Deviation','HiddenDim','LR','Dropout','GCNTime']

    with open(filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                        writer.writerow(headers)
                writer.writerow([args.method,args.dataset,f"{(finalvalacc):.2f}", f"{(finalvalstd):.2f}",f"{(finaltestacc):.2f}", f"{(finalteststd):.2f}",
                hidden_dimension,lr,p,gcn_end-gcn_start])

else:
  headers = ['Method','Dataset','AvgValAcc','ValDeviation','AvgTestAcc', 'Deviation','ELI',
        'AdjHom','NMI',
        'EdgesModified','SameClassEdges',
        'DiffClassEdges','SameClassSameCommunityBefore',
        'SameClassDiffCommunityBefore','DiffClassSameCommunityBefore','DiffClassDiffCommunityBefore','SameClassSameCommunityAfter','SameClassDiffCommunityAfter','DiffClassSameCommunityAfter','DiffClassDiffCommunityAfter','FinalGap','HiddenDim','LR','Dropout','GCNTime']

  with open(filename, mode='a', newline='') as file:
              writer = csv.writer(file)
              if file.tell() == 0:
                      writer.writerow(headers)
              writer.writerow([args.method,args.dataset,f"{(finalvalacc):.2f}", f"{(finalvalstd):.2f}",f"{(finaltestacc):.2f}", f"{(finalteststd):.2f}",
              edgeliaf,hadjaf,nmiscoremod,max_iterations*update_period,
              same_class_edges, diff_class_edges, 
              same_class_same_community_before, 
              same_class_diff_community_before, 
              diff_class_same_community_before, 
              diff_class_diff_community_before, 
              same_class_same_community_after, 
              same_class_diff_community_after, 
              diff_class_same_community_after, 
              diff_class_diff_community_after,
              fgap,hidden_dimension,lr,p,gcn_end-gcn_start])


