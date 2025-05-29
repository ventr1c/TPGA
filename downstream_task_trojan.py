import os
import yaml
from yaml import SafeLoader

from Trojans.tasker import NodeTask, GraphTask
from prompt_graph.utils import seed_everything
from prompt_graph.utils import print_model_parameters
from prompt_graph.utils import  get_args

import argparse
import torch

import Trojans.trojan_utils as trojan_utils


from Trojans.backdoor import Backdoor

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--pre_train_model_path', type=str, default='./Experiment/pre_trained_model/Cora/SimGRACE.GCN.128hidden_dim.pth', help='pretrained_GNN_model_path')
parser.add_argument('--prompt_path', type=str, help='')
parser.add_argument('--answer_path', type=str, help='')

parser.add_argument('--gnn_type', type=str, default='GCN', help='model',
                    choices=['GCN','GAT','GraphSage','GIN'])
parser.add_argument('--prompt_type', type=str, default='Gprompt', help='model',
                    choices=['Gprompt','All-in-one','GPPT','GPF', 'GPF-plus'])
parser.add_argument('--dataset_name', type=str, default='Photo', 
                    help='Dataset',
                    choices=['Cora','Citeseer','Pubmed','Physics','Photo','ogbn-arxiv'])
parser.add_argument('--seed', type=int, default=10, help='Random seed.')

parser.add_argument('--shot_num', type=int, default=10,)
# parser.add_argument('--hid_dim', type=int, default=128,)
parser.add_argument('--num_layer', type=int, default=2,)
parser.add_argument('--epochs', type=int, default=200,)

parser.add_argument('--training_split', type=str, default='k-shot', help='model',
                    choices=['classical','k-shot'])

parser.add_argument('--pre_train_method', type=str, default='SimGRACE', help='task description',choices=['GraphCL','SimGRACE'])
parser.add_argument('--pooling_type', type=str, default='sum', 
                    choices=['sum','mean','max'])
# Backdoor setting
parser.add_argument('--trojan_outter_epochs', type=int,  default=200, help='Number of epochs to train trigger generator.')
parser.add_argument('--trojan_inner_epochs', type=int,  default=1, help='Number of epochs to train trigger generator.')
parser.add_argument('--trojan_pg_lr', type=float,  default=0.0001, help='Learning rate of trojan prompt graph generator')
parser.add_argument('--trojan_pg_weight_decay', type=float,  default=5e-4, help='Weight decay of trojan prompt graph generator')
parser.add_argument('--trigger_lr', type=float,  default=0.0001, help='Learning rate of trigger generator')
parser.add_argument('--trigger_weight_decay', type=float,  default=1e-5, help='Weight decay of trigger generator')
parser.add_argument('--trojan_dt_lr', type=float,  default=0.0001, help='Learning rate of downstream tasker')
parser.add_argument('--trojan_dt_weight_decay', type=float,  default=5e-4, help='Weight decay of downstream tasker')
parser.add_argument('--hid_dim', type=int,  default=128, help='Output dimension')
parser.add_argument('--thrd', type=float, default=0.5)
parser.add_argument('--trigger_size', type=int,  default=10, help='Trigger size.')
parser.add_argument('--target_class', type=int,  default=0, help='Target class.')
parser.add_argument('--trigger_prob', type=float,  default=0.5, help='the probability of generating random ER subgraph for baseline')
parser.add_argument('--baseline_trojan_feat_generation', type=str, default='Rand_Samp', choices= ['Rand_Gene','Rand_Samp'])
parser.add_argument('--compare_method', type=str, default='Ours', choices= ['Ours','Random','UGBA','GTA','SBA-Samp','SBA-Gene'])
parser.add_argument('--poison_ratio', type=float,  default=1, help='poison ratio')
parser.add_argument('--if_freeze_dt_classifier', action='store_true', default=False, help="A boolean flag")
parser.add_argument('--epoch_trojan_finetune', type=int,  default=200, help='Target class.')
parser.add_argument('--loss_bkd_weight', type=float,  default=1, help='weight of bkd loss for GPPT.')

parser.add_argument('--alpha', type=float, default=100.0,
                    help='weight of meta')
parser.add_argument('--eps', type=float, default=10.0,
                    help='size of the ball')
parser.add_argument('--step', type=int, default=10,
                    help='number of steps in meta IP')
parser.add_argument('--second_order', action='store_true',
        default=False, help='whether compute second order gradient')
parser.add_argument('--random', action='store_true',
        default=False, help='whether compute second order gradient')

# GPU setting
parser.add_argument('--device_id', type=int, default=3,
                    help="GPU ID")
parser.add_argument('--task', type=str, default='NodeTask', help='task description')
args = parser.parse_known_args()[0]


if(args.compare_method == 'SBA-Samp'):
    args.baseline_trojan_feat_generation = 'Rand_Samp'
elif(args.compare_method == 'SBA-Gene'):
    args.baseline_trojan_feat_generation = 'Rand_Gene'

seed_everything(args.seed)

config_path = './config/{}/{}/{}.yaml'.format(args.prompt_type, args.pre_train_method, args.compare_method)
config = yaml.load(open(config_path), Loader=SafeLoader)[args.dataset_name]

args.epochs = config['epochs']
args.prompt_lr = config['prompt_lr']
args.prompt_weight_decay = config['prompt_weight_decay']
args.answering_lr = config['answering_lr']
args.answering_weight_decay = config['answering_weight_decay']

args.trojan_outter_epochs = config['trojan_outter_epochs']
args.poison_ratio = config['poison_ratio']
args.trigger_size = config['trigger_size']
args.target_class = config['target_class']

args.hid_dim = config['hid_dim']
args.trojan_pg_lr = config['trojan_pg_lr']
args.trojan_pg_weight_decay = config['trojan_pg_weight_decay']
args.trigger_lr = config['trigger_lr']
args.trigger_weight_decay = config['trigger_weight_decay']
args.trojan_dt_lr = config['trojan_dt_lr']
args.trojan_dt_weight_decay = config['trojan_dt_weight_decay']
args.dt_batch_size = config['dt_batch_size']
args.trojan_batch_size = config['trojan_batch_size']
args.device_id = config['device_id']
if(args.prompt_type == 'GPPT'):
    args.loss_bkd_weight = config['loss_bkd_weight']

args.pre_train_model_path = './Experiment/pre_trained_model/{}/'.format(args.dataset_name) + '{}.{}.{}hidden_dim.{}.pth'.format(args.pre_train_method, args.gnn_type, args.hid_dim, args.pooling_type)
# args.pre_train_model_path = './Experiment/pre_trained_model/{}/'.format(args.dataset_name) + 'SimGRACE.GCN.128hidden_dim.pth'

args.prompt_path = './Experiment/prompt_graph/{}/'.format(args.dataset_name) + '{}.{}.{}.{}.{}.pth'.format(args.prompt_type, args.pre_train_method, args.gnn_type, args.hid_dim, args.pooling_type)
args.answer_path = './Experiment/answering_model/{}/'.format(args.dataset_name) + '{}.{}.{}.{}.{}.pth'.format(args.prompt_type, args.pre_train_method, args.gnn_type, args.hid_dim, args.pooling_type)


os.makedirs(os.path.dirname(args.pre_train_model_path), exist_ok=True)  # This creates the directory if it does not exist
os.makedirs(os.path.dirname(args.prompt_path), exist_ok=True)  # This creates the directory if it does not exist
os.makedirs(os.path.dirname(args.answer_path), exist_ok=True)  # This creates the directory if it does not exist

args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

print(args)

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
    tasker = NodeTask(args, 
                      pre_train_model_path = args.pre_train_model_path, 
                      dataset_name = args.dataset_name, 
                      num_layer = args.num_layer, 
                      gnn_type = args.gnn_type, 
                      hid_dim = args.hid_dim,
                      prompt_type = args.prompt_type, 
                      epochs = args.epochs, 
                      shot_num = args.shot_num, 
                      device=device)
    
    tasker.run()
    if(args.if_freeze_dt_classifier):
        tasker.trojan_ACC_test_evaluation_freeze()
        tasker.prompt_graph_poisoning_freeze_DT()
        tasker.trojan_ASR_test_evaluation_freeze()
    else:
        tasker.trojan_ACC_test_evaluation()
        tasker.prompt_graph_poisoning()
        tasker.trojan_ASR_test_evaluation()
        tasker.trojan_finetune_dt_evaluation()
        # tasker.trojan_freeze_dt_evaluation()
        
    # tasker.trojan_freeze_dt_evaluation()
    # tasker.visualize_embeddings_graph_level()

if args.task == 'GraphTask':
    tasker = GraphTask(args, pre_train_model_path = args.pre_train_model_path, 
                    dataset_name = args.dataset_name, num_layer = args.num_layer, gnn_type = args.gnn_type, prompt_type = args.prompt_type, epochs = args.epochs, shot_num = args.shot_num, device=args.device)
    tasker.run()

