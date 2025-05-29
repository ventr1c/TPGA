import yaml
from yaml import SafeLoader
from prompt_graph.pretrain import Edgepred_GPPT, Edgepred_Gprompt, SimGRACE, PrePrompt
from prompt_graph.utils import seed_everything
from prompt_graph.utils import mkdir, get_args
from torch_geometric.datasets import Planetoid,Flickr

import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--pre_train_model_path', type=str, default='./pre_trained_gnn/Cora.GraphCL.GCN.128hidden_dim.pth', help='pretrained_GNN_model_path')
parser.add_argument('--gnn_type', type=str, default='GCN', help='model',
                    choices=['GCN','GAT','GraphSage','GIN'])
# parser.add_argument('--prompt_type', type=str, default='GPF', help='model',
                    # choices=['Gprompt,All-in-one','GPF'])
parser.add_argument('--task', type=str, default='SimGRACE', help='task description',
                    choices=['GraphCL','SimGRACE'])
parser.add_argument('--dataset_name', type=str, default='Photo', 
                    help='Dataset',
                    choices=['Cora','Pubmed','Citeseer','ogbn-arxiv','Photo', 'Physics'])
parser.add_argument('--seed', type=int, default=10, help='Random seed.')

parser.add_argument('--shot_num', type=int, default=10,)
parser.add_argument('--hid_dim', type=int, default=128,)
parser.add_argument('--num_layer', type=int, default=2,)
parser.add_argument('--epochs', type=int, default=50,)
parser.add_argument('--batch_size', type=int, default=128,)
parser.add_argument('--pooling_type', type=str, default='sum', 
                    choices=['sum','mean','max'])

# GPU setting
parser.add_argument('--device_id', type=int, default=1,
                    help="GPU ID")
args = parser.parse_known_args()[0]

import yaml
print(args.dataset_name)

config_path = "./config/pretrain/{}/{}.yaml".format(args.task,args.dataset_name)
config = yaml.load(open(config_path), Loader=SafeLoader)[args.gnn_type]

args.device_id = config['device_id']
args.epochs = config['epochs']
args.batch_size = config['batch_size']
args.pooling_type = config['pooling_type']
args.hid_dim = config['hid_dim']
args.num_layer = config['num_layer']
args.num_parts = config['num_parts']
args.lr = config['lr']
args.weight_decay = config['weight_decay']
args.device_id = config['device_id']
if(args.task == 'GraphCL'):
    args.node_drop_1 = config['node_drop_1']
    args.feat_mask_1 = config['feat_mask_1']
    args.edge_drop_1 = config['edge_drop_1']
    args.node_drop_2 = config['node_drop_2']
    args.feat_mask_2 = config['feat_mask_2']
    args.edge_drop_2 = config['edge_drop_2']

args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

# args = get_args()
seed_everything(args.seed)
mkdir('./pre_trained_gnn/')

from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
transform = T.Compose([T.NormalizeFeatures()])

# if(args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
#     dataset = Planetoid(root='./data/', \
#                         name=args.dataset,\
#                         transform=transform)
# elif(args.dataset == 'Flickr'):
#     dataset = Flickr(root='./data/Flickr/', \
#                     transform=transform)
# elif(args.dataset == 'ogbn-arxiv'):
#     from ogb.nodeproppred import PygNodePropPredDataset
#     # Download and process data at './dataset/ogbg_molhiv/'
#     dataset = PygNodePropPredDataset(name = 'ogbn-arxiv', root='./data/')
#     split_idx = dataset.get_idx_split() 

# data = dataset[0].to(device)

# if(args.dataset == 'ogbn-arxiv'):
#     nNode = data.x.shape[0]
#     setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
#     # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
#     data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
#     data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
#     data.y = data.y.squeeze(1)


if args.task == 'SimGRACE':
    pt = SimGRACE(args, dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=device)
if args.task == 'GraphCL':
    pt = GraphCL(args, dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=device)
if args.task == 'Edgepred_GPPT':
    pt = Edgepred_GPPT(args, dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=device)
if args.task == 'Edgepred_Gprompt':
    pt = Edgepred_Gprompt(args, dataset_name = args.dataset_name, gnn_type = args.gnn_type, hid_dim = args.hid_dim, gln = args.num_layer, num_epoch=args.epochs, device=device)
if args.task == 'MultiGprompt':
    nonlinearity = 'prelu'
    pt = PrePrompt(args, args.dataset_name, args.hid_dim, nonlinearity, 0.9, 0.9, 0.1, 0.001, 2, 0.3)
pt.pretrain(batch_size =args.batch_size, epochs = args.epochs, lr=args.lr, decay=args.weight_decay)

