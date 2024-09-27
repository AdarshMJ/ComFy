import random
import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.datasets import TUDataset, Planetoid,WebKB,Actor, WikipediaNetwork, Coauthor, EmailEUCore,Amazon,HeterophilousGraphDataset
from torch_geometric.transforms import NormalizeFeatures, LargestConnectedComponents

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def load_data(datasetname,num_train,num_val):
        path = '../data/' + datasetname
        if datasetname in ['Cora','Citeseer','Pubmed']:
            dataset = Planetoid(root=path, name=datasetname,transform=NormalizeFeatures())
            data = dataset[0]
            print(data)
            num_features = dataset.num_features
            num_classes = dataset.num_classes
            print(f"Selecting the LargestConnectedComponent..")
            transform = LargestConnectedComponents()
            data = transform(dataset[0])
            print()
            print("Splitting datasets train/val/test...")
            #transform2 = RandomNodeSplit(split="train_rest",num_splits=100)
            #transform2 = RandomNodeSplit(split="train_rest",num_splits=100,num_train_per_class=num_train,num_val=num_val)
            transform2 = RandomNodeSplit(split="train_rest",num_splits=100,num_test=0.2,num_val=0.2)
            data  = transform2(data)
            #data = data.to(device)
            #print(data)
            #data.train_mask = data.train_mask | data.val_mask
            #data.val_mask = torch.zeros_like(data.val_mask)  # Clear validation mask    
            #data = data.to(device)
            #print(data)
            # num_train_nodes = data.train_mask.sum().item()
            # num_val_nodes = data.val_mask.sum().item()
            # num_test_nodes = data.test_mask.sum().item()
            # print()
            # print(f"Number of training nodes: {num_train_nodes/100}")
            # print(f"Number of validation nodes: {num_val_nodes/100}")
            # print(f"Number of test nodes: {num_test_nodes/100}")
            # num_train_nodes = data.train_mask.sum().item()
            # num_val_nodes = data.val_mask.sum().item()
            # num_test_nodes = data.test_mask.sum().item()
            # print()
            # print(f"Number of training nodes: {num_train_nodes/100}")
            # print(f"Number of validation nodes: {num_val_nodes/100}")
            # print(f"Number of test nodes: {num_test_nodes/100}")


        elif datasetname in ['CS','Physics']:
            dataset = Coauthor(root=path, name=datasetname,transform=NormalizeFeatures())
            data = dataset[0]
            num_features = dataset.num_features
            num_classes = dataset.num_classes
            print(f"Selecting the LargestConnectedComponent..")
            transform = LargestConnectedComponents()
            data = transform(dataset[0])
            print()
            print("Splitting datasets train/val/test...")
            #transform2 = RandomNodeSplit(split="test_rest",num_splits=100,num_train_per_class = 200)
            transform2 = RandomNodeSplit(split="test_rest",num_splits=100)
            data  = transform2(data)
            #data = data.to(device)

            print(data)
            
        elif datasetname in ['Computers','Photo']:
            dataset = Amazon(root=path, name=datasetname,transform=NormalizeFeatures())
            data = dataset[0]
            num_features = dataset.num_features
            num_classes = dataset.num_classes
            print(f"Selecting the LargestConnectedComponent..")
            transform = LargestConnectedComponents()
            data = transform(dataset[0])
            print()
            print("Splitting datasets train/val/test...")
            #transform2 = RandomNodeSplit(split="test_rest",num_splits=100,num_train_per_class = 200)
            transform2 = RandomNodeSplit(split="test_rest",num_splits=100)
            data  = transform2(data)
            #data = data.to(device)
            print(data)


        elif datasetname in ['Roman-empire','Minesweeper']:
            dataset = HeterophilousGraphDataset(root=path, name=datasetname,transform=NormalizeFeatures())
            data = dataset[0]
            num_features = dataset.num_features
            num_classes = dataset.num_classes
            print(f"Selecting the LargestConnectedComponent..")
            transform = LargestConnectedComponents()
            print()
            data = transform(dataset[0])
            #data = data.to(device)
            print(data)



        else:
            raise ValueError(f"Dataset {datasetname} not found")

        return data, num_classes,num_features


##Sanity check if data leakage occurs when we split data###
# for i in range(100):
#     train_nodes = data.train_mask[:, i].nonzero(as_tuple=True)[0].cpu().numpy()
#     test_nodes = data.test_mask[:, i].nonzero(as_tuple=True)[0].cpu().numpy()
#     val_nodes = data.val_mask[:, i].nonzero(as_tuple=True)[0].cpu().numpy()
#     leakage_nodes = np.intersect1d(train_nodes, test_nodes)
#     if len(leakage_nodes) > 0:
#         print(
#             f"Warning: Found {len(leakage_nodes)} nodes in both the training and test sets."
#         )
#     else:
#         print("No leakage detected.")