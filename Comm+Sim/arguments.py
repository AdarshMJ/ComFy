import argparse

def parse_args():
    
    parser = argparse.ArgumentParser(description='Run NodeClassification+HomophilyBased script')
    parser.add_argument('--dataset', type=str, help='Dataset to download')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in GCN')
    parser.add_argument('--model', type=str, default='SimpleGCN', choices=['GCN', 'GATv2','SimpleGCN'], help='Model to use')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads in GATv2')
    parser.add_argument('--out', type=str, help='name of log file')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout = [Cora - 0.4130296, Citeseer - 0.3130296]')
    parser.add_argument('--hidden_dimension', type=int, default=32, help='Hidden Dimension size')
    parser.add_argument('--LR', type=float, default=0.01, help='Learning Rate = [0.01,0.001]')
    parser.add_argument('--device',type=str,default='cuda',help='Device to use')
    parser.add_argument('--seed',type=int,default=3164711608,help='Seed to use')
    parser.add_argument('--num_train',type=int,default=20,help='Number of training nodes per class')
    parser.add_argument('--num_val',type=int,default=500,help='Number of validation nodes')
    parser.add_argument('--splits',type=int,default=100,help='Number of splits')
    parser.add_argument('--weight_decay',type=float,default=5e-4,help='Weight Decay')
    #parser.add_argument('--rewiringbudget',type=int,default=100,help='Number of edges to rewire')
    parser.add_argument('--budget_edges_add',type=int,default=100,help='Number of edges to add')
    parser.add_argument('--budget_edges_delete',type=int,default=100,help='Number of edges to delete')
    return parser.parse_args()