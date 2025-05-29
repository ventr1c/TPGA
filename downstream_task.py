from prompt_graph.tasker import NodeTask, GraphTask
from prompt_graph.utils import seed_everything
from torchsummary import summary
from prompt_graph.utils import print_model_parameters
from prompt_graph.utils import  get_args

import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--pre_train_model_path', type=str, default='./Experiment/pre_trained_model/Cora/SimGRACE.GCN.128hidden_dim.pth', help='pretrained_GNN_model_path')
parser.add_argument('--gnn_type', type=str, default='GIN', help='model',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--prompt_type', type=str, default='GPF', help='model',
                    choices=['Gprompt','All-in-one','GPPT','None','GPF'])
parser.add_argument('--dataset_name', type=str, default='ENZYMES', 
                    help='Dataset',
                    choices=['Cora','Citeseer','Pubmed','Flickr','ogbn-arxiv','ENZYMES'])
parser.add_argument('--seed', type=int, default=10, help='Random seed.')

parser.add_argument('--shot_num', type=int, default=10,)
parser.add_argument('--hid_dim', type=int, default=128,)
parser.add_argument('--num_layer', type=int, default=2,)
parser.add_argument('--epochs', type=int, default=500,)

parser.add_argument('--pre_train_method', type=str, default='GraphCL', help='task description')

# GPU setting
parser.add_argument('--device_id', type=int, default=3,
                    help="GPU ID")
parser.add_argument('--task', type=str, default='NodeTask', help='task description')
args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

args.pre_train_model_path = './Experiment/pre_trained_model/{}/'.format(args.dataset_name) + '{}.{}.{}hidden_dim.pth'.format(args.pre_train_method, args.gnn_type, args.hid_dim)

# args.pre_train_model_path = './Experiment/pre_trained_model/{}/'.format(args.dataset_name) + 'GraphCL.GCN.128hidden_dim.pth'

seed_everything(args.seed)
# args.task = 'GraphTask'
# args.prompt_type = 'Gprompt'
# args.dataset_name = 'PROTEINS'
# args.task = 'NodeTask'
# args.epochs = 10
# args.dataset_name = 'CiteSeer'

# args.prompt_type = 'MultiGprompt'
# args.pre_train_model_path = './multigprompt_model/cora.multigprompt.GCL.128hidden_dim.pth'


print(args.pre_train_model_path)
if args.task == 'NodeTask':
    tasker = NodeTask(pre_train_model_path = args.pre_train_model_path, 
                    dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, prompt_type = args.prompt_type, epochs = args.epochs, shot_num = args.shot_num, device=device)
    
    tasker.run()


if args.task == 'GraphTask':
    tasker = GraphTask(pre_train_model_path = args.pre_train_model_path, 
                    dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, prompt_type = args.prompt_type, epochs = args.epochs, shot_num = args.shot_num, device=args.device)
    tasker.run()