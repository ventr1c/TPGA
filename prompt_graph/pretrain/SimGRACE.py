import torch
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from prompt_graph.utils import mkdir
from torch.optim import Adam
from prompt_graph.data import load4node, load4graph, NodePretrain
from copy import deepcopy
from.base import PreTrain
import os

import torch.nn as nn
from prompt_graph.utils import accuracy
from prompt_graph.utils import get_split_self
import pickle
from prompt_graph.data import load4node, split_induced_graphs
from prompt_graph.data import load4node_labelwise, split_khop_induced_graphs, split_khop_induced_graphs_specific
from torch_geometric.data import Batch

class SimGRACE(PreTrain):

    def __init__(self, *args, **kwargs):    # hid_dim=16
        super().__init__(*args, **kwargs)
        self.load_graph_data()
        self.initialize_gnn(self.input_dim, self.hid_dim)   
        self.projection_head = torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.hid_dim),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Linear(self.hid_dim, self.hid_dim)).to(self.device)
    def load_graph_data(self):
        if self.dataset_name in ['Pubmed', 'Citeseer', 'Cora','Computers', 'Photo', 'Reddit', 'WikiCS', 'Flickr', 'ogbn-arxiv', 'Physics']:
            self.data, self.dataset, self.graph_list, self.input_dim, self.graph_cluster = NodePretrain(dataname = self.dataset_name, num_parts=100)
            # self.data, self.dataset = load4node_labelwise(self.args, self.dataset_name, train_ratio = 0.1, test_ratio = 0.8, device = self.device)
            # self.graph_list = split_khop_induced_graphs(self.dataset_name, self.data, K = 2)
            # self.input_dim = self.data.x.shape[1]
        else:
            self.input_dim, self.out_dim, self.graph_list= load4graph(self.dataset_name,pretrained=True)
        
    def get_loader(self, graph_list, batch_size):

        if len(graph_list) % batch_size == 1:
            raise KeyError(
                "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in SimGRACE!")

        loader = DataLoader(graph_list, batch_size=batch_size, shuffle=False, num_workers=1)
        return loader
    
    def forward_cl(self, x, edge_index, batch):
        x = self.gnn(x, edge_index, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = - torch.log(pos_sim / (sim_matrix.sum(dim=1) + 1e-4)).mean()
        # loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
        # loss = - torch.log(loss).mean() 
        return loss

    def perturbate_gnn(self, data):
        vice_model = deepcopy(self).to(self.device)

        for (vice_name, vice_model_param) in vice_model.named_parameters():
            if vice_name.split('.')[0] != 'projection_head':
                std = vice_model_param.data.std() if vice_model_param.data.numel() > 1 else torch.tensor(1.0)
                noise = 0.1 * torch.normal(0, torch.ones_like(vice_model_param.data) * std)
                vice_model_param.data += noise
        z2 = vice_model.forward_cl(data.x, data.edge_index, data.batch)
        return z2
    
    def train_simgrace(self, loader, optimizer):
        self.train()
        train_loss_accum = 0
        total_step = 0
        for step, data in enumerate(loader):
            optimizer.zero_grad()
            data = data.to(self.device)
            x2 = self.perturbate_gnn(data) 
            x1 = self.forward_cl(data.x, data.edge_index, data.batch)
            x2 = Variable(x2.detach().data.to(self.device), requires_grad=False)
            loss = self.loss_cl(x1, x2)
            loss.backward()
            optimizer.step()
            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1
        return train_loss_accum / total_step

    def pretrain(self, batch_size=10, lr=0.001,decay=0.0001, epochs=100):
        printN = 2
        loader = self.get_loader(self.graph_list, batch_size)
        print('start training {} | {} | {}...'.format(self.dataset_name, 'SimGRACE', self.gnn_type))
        optimizer = optim.Adam(self.gnn.parameters(), lr=lr, weight_decay=decay)

        '''
        split for downstream tasks
        '''
        # self.data, self.dataset = load4node_labelwise(self.args, self.dataset_name, train_ratio = 0.1, test_ratio = 0.4, device = self.device)
        self.input_dim = self.dataset.num_features
        self.output_dim = self.dataset.num_classes
        # self.dt_graph_list = self.load_induced_graph_random_split()
        # self.dt_graph_loader =  Batch.from_data_list(self.dt_graph_list)
        
        self.split_train = get_split_self(num_samples=self.data.x.shape[0], train_ratio=0.1, test_ratio=0.2,seed=self.args.seed,device=self.device)
        idx_train = self.split_train['train']
        idx_test = self.split_train['test']
        idx_val = self.split_train['valid']

        print(self.data)
        print(idx_train)
        if(self.dataset_name in ['Flickr','ogbn-arxiv']):
            evalute_dt = False
        else:
            evalute_dt = True
            self.train_graph_list = self.load_induced_graph_random_split_specific(idx_train, 'train')
            self.test_graph_list = self.load_induced_graph_random_split_specific(idx_test, 'test')
            
            
            self.train_loader = self.get_loader(self.train_graph_list, batch_size)
            self.test_loader = self.get_loader(self.test_graph_list, batch_size)
        # self.val_loader = self.get_loader(self.dt_graph_loader[idx_val], batch_size)

        train_loss_min = 1000000
        test_acc_max = 0.0
        for epoch in range(1, epochs + 1):  # 1..100
            train_loss = self.train_simgrace(loader, optimizer)

            if(self.args.debug and (epoch % printN)==0):
                print("***epoch: {}/{} | train_loss: {:.8}".format(epoch, epochs, train_loss))
            if(evalute_dt and (epoch % printN)==0):
                test_acc = self.test(self.train_loader,self.test_loader)
                if(test_acc>test_acc_max):
                    test_acc_max = test_acc
                    folder_path = f"./Experiment/pre_trained_model/{self.dataset_name}"
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    torch.save(self.gnn.state_dict(),
                            "./Experiment/pre_trained_model/{}/{}.{}.{}.{}_pretrain.pth".format(self.dataset_name, 'SimGRACE', self.gnn_type, str(self.hid_dim) + 'hidden_dim', self.args.pooling_type))
                    if(self.args.debug and (epoch % printN)==0):
                        print("+++model saved ! {}.{}.{}.{}.{}_pretrain.pth".format(self.dataset_name, 'SimGRACE', self.gnn_type, str(self.hid_dim) + 'hidden_dim', self.args.pooling_type))
            elif train_loss_min > train_loss:
                train_loss_min = train_loss
                folder_path = f"./Experiment/pre_trained_model/{self.dataset_name}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                torch.save(self.gnn.state_dict(),
                           "./Experiment/pre_trained_model/{}/{}.{}.{}.{}_pretrain.pth".format(self.dataset_name, 'SimGRACE', self.gnn_type, str(self.hid_dim) + 'hidden_dim', self.args.pooling_type))
                if(self.args.debug and (epoch % printN)==0):
                    print("+++model saved ! {}.{}.{}.{}.{}_pretrain.pth".format(self.dataset_name, 'SimGRACE', self.gnn_type, str(self.hid_dim) + 'hidden_dim', self.args.pooling_type))
    def test(self, train_loader, test_loader):
        self.eval()
        x_train = []
        y_train = []

        with torch.no_grad():
            for data in train_loader:
                data = data.to(self.device)
                g = self.gnn(data.x, data.edge_index, data.batch)
                x_train.append(g.detach())
                y_train.append(data.y)
                data = data.cpu()
            x_train = torch.cat(x_train, dim=0)
            y_train = torch.cat(y_train, dim=0)

        num_classes = y_train.max().item() + 1
        fc = nn.Linear(x_train.shape[1], num_classes).to(self.device)
        optimizer = torch.optim.Adam(fc.parameters(), lr=0.001, weight_decay=5e-4)

        for _ in range(1000):
            optimizer.zero_grad()
            pred = fc(x_train)
            loss = torch.nn.functional.cross_entropy(pred,y_train)
            loss.backward()
            optimizer.step()


        x_test = []
        y_test = []

        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                g = self.gnn(data.x, data.edge_index, data.batch)
                x_test.append(g.detach())
                y_test.append(data.y)
            x_test = torch.cat(x_test, dim=0)
            y_test = torch.cat(y_test, dim=0)

        pred_test = torch.softmax(fc(x_test),dim=1)
       
        acc = accuracy(pred_test,y_test)
        
        if self.args.debug:
            print("Accuracy: {:.4f}".format(acc))
        return float(acc)
    
    def load_induced_graph_random_split(self):
        K = 2 # num_hop
        self.data, self.dataset = load4node_labelwise(self.args, self.dataset_name, train_ratio = 0.1, test_ratio = 0.2, device = torch.device('cpu'))
        # self.data.to('cpu')
        self.input_dim = self.dataset.num_features
        self.output_dim = self.dataset.num_classes
        file_path = './Experiment/induced_graph/' + self.dataset_name + '/induced_graph_{}_hop.pkl'.format(K)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                graphs_list = pickle.load(f)
        else:
            print('Begin split_induced_graphs.')
            split_khop_induced_graphs(self.dataset_name, self.data, K = K)
            with open(file_path, 'rb') as f:
                graphs_list = pickle.load(f)
        return graphs_list
    
    def load_induced_graph_random_split_specific(self, node_idxs, flag):
        if(flag not in ['train', 'test', 'val']):
            raise NotImplementedError("not implemented other induced graphs")

        K = 2 # num_hop
        file_path = './Experiment/induced_graph/' + self.dataset_name + '/induced_graph_{}_hop_{}.pkl'.format(K,flag)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                graphs_list = pickle.load(f)
        else:
            print('Begin split_induced_graphs.')
            split_khop_induced_graphs_specific(self.dataset_name, self.data, node_idxs, flag = flag, K = K)
            with open(file_path, 'rb') as f:
                graphs_list = pickle.load(f)
        return graphs_list