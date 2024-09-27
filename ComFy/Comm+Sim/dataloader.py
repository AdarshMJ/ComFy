import random
import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.datasets import TUDataset, Planetoid,WebKB,Actor, WikipediaNetwork, Coauthor,Amazon,HeterophilousGraphDataset
from torch_geometric.transforms import NormalizeFeatures, LargestConnectedComponents

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def load_data(datasetname,num_train,num_val):
        path = '../data/' + datasetname
        if datasetname in ['Cora','Citeseer','Pubmed']:
            dataset = Planetoid(root=path, name=datasetname,transform=NormalizeFeatures())
            data = dataset[0]
            num_features = dataset.num_features
            num_classes = dataset.num_classes
            print(f"Selecting the LargestConnectedComponent..")
            transform = LargestConnectedComponents()
            data = transform(dataset[0])
            print()
            print("Splitting datasets train/val/test...")
            #transform2 = RandomNodeSplit(split="train_rest",num_splits=100)
            transform2 = RandomNodeSplit(split="train_rest",num_splits=100,num_test=0.2,num_val=0.2)
            data  = transform2(data)


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

