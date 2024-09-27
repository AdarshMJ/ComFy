# I need to call this file from the main.py file to train the model and get the results
import os
import random
import time
import numpy as np
import torch
from tqdm import tqdm



def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def train_and_get_results(data, model,p,lr,seed,weight_decay=5e-4):
    final_test_accuracies = []
    final_val_accuracies = []
    final_train_accuracies = []


    def train():
        model.train()
        optimizer.zero_grad()  
        out = model(data.x, data.edge_index,p)          
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()  
        optimizer.step()  
        pred = out.argmax(dim=1)  
        train_correct = pred[train_mask] == data.y[train_mask]  
        train_acc = int(train_correct.sum()) / int(train_mask.sum())  
        return loss, train_acc


    def val():
        model.eval()
        out = model(data.x, data.edge_index,p)
        pred = out.argmax(dim=1)  # Use the class with highest probability. 
        val_correct = pred[val_mask] == data.y[val_mask]  # Check against ground-truth labels.
        val_acc = int(val_correct.sum()) / int(val_mask.sum())  # Derive ratio of correct predictions.
        return val_acc


    def test():
            model.eval()
            out= model(data.x, data.edge_index,p)
            pred = out.argmax(dim=1)  # Use the class with highest probability. 
            test_correct = pred[test_mask] == data.y[test_mask]  # Check against ground-truth labels.
            test_acc = int(test_correct.sum()) / int(test_mask.sum())  # Derive ratio of correct predictions.
            return test_acc


    for split_idx in range(1,100):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        
        
        # Reset optimizer
        #optimizer = type(optimizer)(model.parameters(), **optimizer.defaults)
        print(f"Training for index = {split_idx}")
        train_mask = data.train_mask[:,split_idx]
        test_mask = data.test_mask[:,split_idx]
        val_mask = data.val_mask[:,split_idx]

        train_accuracies = []
        test_accuracies = []
        val_accuracies = []

        set_seed(seed)
        print("Start training ....")
        for epoch in tqdm(range(1, 101)):
            loss, train_acc = train()
            train_accuracies.append(train_acc * 100)

        val_acc = val()
        test_acc = test()
        
        test_accuracies.append(test_acc * 100)
        val_accuracies.append(val_acc * 100)

        final_test_acc = test_accuracies[-1]
        final_val_acc = val_accuracies[-1]
        final_train_acc = train_accuracies[-1]

        final_test_accuracies.append(final_test_acc)
        final_val_accuracies.append(final_val_acc)
        final_train_accuracies.append(final_train_acc)

        print(f"Split {split_idx}: Test Accuracy: {final_test_acc:.2f}%")

    print(f"Average Test Accuracy: {np.mean(final_test_accuracies):.2f}% ± {2 * np.std(final_test_accuracies) / np.sqrt(len(final_test_accuracies)):.2f}%")
    print(f"Average Validation Accuracy: {np.mean(final_val_accuracies):.2f}% ± {2 * np.std(final_val_accuracies) / np.sqrt(len(final_val_accuracies)):.2f}%")
    print(f"Average Train Accuracy: {np.mean(final_train_accuracies):.2f}% ± {2 * np.std(final_train_accuracies) / np.sqrt(len(final_train_accuracies)):.2f}%")

    return final_test_accuracies, final_val_accuracies, final_train_accuracies


