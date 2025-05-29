import torch
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from random import shuffle
import random
from prompt_graph.utils import mkdir, graph_views
from prompt_graph.data import load4node, load4graph, NodePretrain
from torch.optim import Adam
import os
from.base import PreTrain

import torch.nn as nn
from prompt_graph.utils import accuracy
from prompt_graph.utils import get_split_self
import pickle
from prompt_graph.data import load4node, split_induced_graphs
from prompt_graph.data import load4node_labelwise, split_khop_induced_graphs
from torch_geometric.data import Batch
import GCL.augmentors as A
import GCL.losses as L
from GCL.models import DualBranchContrast


class GraphCL(PreTrain):
    def __init__(self, *args, **kwargs):    # hid_dim=16
        super().__init__(*args, **kwargs)
        self.load_graph_data()
        self.initialize_gnn(self.input_dim, self.hid_dim)   
        self.projection_head = torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.hid_dim),
                                                   torch.nn.ReLU(inplace=True),
                                                   torch.nn.Linear(self.hid_dim, self.hid_dim)).to(self.device)
    def load_graph_data(self):
        if self.dataset_name in ['Pubmed', 'Citeseer', 'Cora','Computers', 'Photo', 'Reddit', 'WikiCS', 'Flickr']:
            self.graph_list, self.input_dim, self.graph_cluster = NodePretrain(dataname = self.dataset_name, num_parts=100)
            # self.data, self.dataset = load4node_labelwise(self.args, self.dataset_name, train_ratio = 0.1, test_ratio = 0.8, device = self.device)
            # self.graph_list = split_khop_induced_graphs(self.dataset_name, self.data, K = 2)
            # self.input_dim = self.data.x.shape[1]
        else:
            self.input_dim, self.out_dim, self.graph_list= load4graph(self.dataset_name,pretrained=True)
    
    def get_loader(self, graph_list, batch_size,aug1=None, aug2=None, aug_ratio=None):

        if len(graph_list) % batch_size == 1:
            raise KeyError(
                "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in GraphCL!")
        
        shuffle(graph_list)
        if aug1 is None:
            aug1 = random.sample(['dropN', 'permE', 'maskN'], k=1)
        if aug2 is None:
            aug2 = random.sample(['dropN', 'permE', 'maskN'], k=1)
        if aug_ratio is None:
            aug_ratio = random.randint(1, 3) * 1.0 / 10  # 0.1,0.2,0.3

        print("===graph views: {} and {} with aug_ratio: {}".format(aug1, aug2, aug_ratio))

        view_list_1 = []
        view_list_2 = []
        for g in graph_list:
            view_g = graph_views(data=g, aug=aug1, aug_ratio=aug_ratio)
            view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
            view_list_1.append(view_g)
            view_g = graph_views(data=g, aug=aug2, aug_ratio=aug_ratio)
            view_g = Data(x=view_g.x, edge_index=view_g.edge_index)
            view_list_2.append(view_g)

        loader1 = DataLoader(view_list_1, batch_size=batch_size, shuffle=False,
                                num_workers=1)  # you must set shuffle=False !
        loader2 = DataLoader(view_list_2, batch_size=batch_size, shuffle=False,
                                num_workers=1)  # you must set shuffle=False !

        return loader1, loader2
    
    def get_loader_gcl(self, graph_list, batch_size, aug1, aug2):
        if len(graph_list) % batch_size == 1:
            raise KeyError(
                "batch_size {} makes the last batch only contain 1 graph, \n which will trigger a zero bug in GraphCL!")
        
        shuffle(graph_list)
        view_list_1 = []
        view_list_2 = []
        for g in graph_list:
            x1, edge_index1, edge_weight1 = aug1(g.x, g.edge_index)
            x2, edge_index2, edge_weight2 = aug2(g.x, g.edge_index)
            view_g = Data(x=x1, edge_index=edge_index1)
            view_list_1.append(view_g)
            view_g = Data(x=x2, edge_index=edge_index2)
            view_list_2.append(view_g)

        loader1 = DataLoader(view_list_1, batch_size=batch_size, shuffle=False,
                                num_workers=1)  # you must set shuffle=False !
        loader2 = DataLoader(view_list_2, batch_size=batch_size, shuffle=False,
                                num_workers=1)  # you must set shuffle=False !

        return loader1, loader2

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

    def train_graphcl(self, loader1, loader2, optimizer):
        self.train()
        train_loss_accum = 0
        total_step = 0
        for step, batch in enumerate(zip(loader1, loader2)):
            batch1, batch2 = batch
            optimizer.zero_grad()
            x1 = self.forward_cl(batch1.x.to(self.device), batch1.edge_index.to(self.device), batch1.batch.to(self.device))
            x2 = self.forward_cl(batch2.x.to(self.device), batch2.edge_index.to(self.device), batch2.batch.to(self.device))
            loss = self.loss_cl(x1, x2)

            loss.backward()
            optimizer.step()

            train_loss_accum += float(loss.detach().cpu().item())
            total_step = total_step + 1

        return train_loss_accum / total_step

    def pretrain(self, batch_size=10, aug1='dropN', aug2="permE", aug_ratio=None, lr=0.001, decay=0.00005, epochs=100):
        printN = 2

        self.to(self.device)

        '''
        split for downstream tasks
        '''
        self.dt_graph_list = self.load_induced_graph_random_split()
        self.dt_graph_loader =  Batch.from_data_list(self.dt_graph_list)
        
        self.split_train = get_split_self(num_samples=len(self.dt_graph_list), train_ratio=0.1, test_ratio=0.8,seed=self.args.seed,device=self.device)
        idx_train = self.split_train['train']
        idx_test = self.split_train['test']
        idx_val = self.split_train['valid']
        self.train_loader = DataLoader(self.dt_graph_loader[idx_train], batch_size=batch_size)
        self.test_loader = DataLoader(self.dt_graph_loader[idx_test], batch_size=batch_size)
        self.val_loader = DataLoader(self.dt_graph_loader[idx_val], batch_size=batch_size)

        # aug1 = A.Identity()
        aug1 = A.RandomChoice([ A.NodeDropping(pn=self.args.node_drop_1),
                                A.FeatureMasking(pf=self.args.feat_mask_1),
                                A.EdgeRemoving(pe=self.args.edge_drop_1)], 1)
        aug2 = A.RandomChoice([ A.NodeDropping(pn=self.args.node_drop_2),
                                A.FeatureMasking(pf=self.args.feat_mask_2),
                                A.EdgeRemoving(pe=self.args.edge_drop_2)], 1)
        # loader1, loader2 = self.get_loader(self.graph_list, batch_size, aug1=aug1, aug2=aug2)
        loader1, loader2 = self.get_loader_gcl(self.graph_list, batch_size, aug1=aug1, aug2=aug2)
        print('start training {} | {} | {}...'.format(self.dataset_name, 'GraphCL', self.gnn_type))
        optimizer = Adam(self.parameters(), lr=lr, weight_decay=decay)

        train_loss_min = 1000000
        best_test_acc = 0
        for epoch in range(1, epochs + 1):  # 1..100
            train_loss = self.train_graphcl(loader1, loader2, optimizer)

            if(self.args.debug and (epoch % printN)==0):
                print("***epoch: {}/{} | train_loss: {:.8}".format(epoch, epochs, train_loss))
            if((epoch % printN)==0):
                test_acc = self.test(self.train_loader,self.val_loader)
                if(test_acc > best_test_acc):
                    best_test_acc = test_acc
                    folder_path = f"./Experiment/pre_trained_model/{self.dataset_name}"
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)

                    torch.save(self.gnn.state_dict(),
                                "./Experiment/pre_trained_model/{}/{}.{}.{}.{}.pth".format(self.dataset_name, 'GraphCL', self.gnn_type, str(self.hid_dim) + 'hidden_dim', self.args.pooling_type))
                    print("+++model saved ! {}.{}.{}.{}.{}.pth".format(self.dataset_name, 'GraphCL', self.gnn_type, str(self.hid_dim) + 'hidden_dim', self.args.pooling_type))
            # if train_loss_min > train_loss:
            #     train_loss_min = train_loss
            #     folder_path = f"./Experiment/pre_trained_model/{self.dataset_name}"
            #     if not os.path.exists(folder_path):
            #         os.makedirs(folder_path)

            #     torch.save(self.gnn.state_dict(),
            #                "./Experiment/pre_trained_model/{}/{}.{}.{}.pth".format(self.dataset_name, 'GraphCL', self.gnn_type, str(self.hid_dim) + 'hidden_dim'))
            #     if(self.args.debug and (epoch % printN)==0):
            #         print("+++model saved ! {}.{}.{}.{}.pth".format(self.dataset_name, 'GraphCL', self.gnn_type, str(self.hid_dim) + 'hidden_dim'))
    def test(self, train_loader, test_loader):
        self.eval()
        x_train = []
        y_train = []

        with torch.no_grad():
            for data in train_loader:
                print(data)
                data = data.to(self.device)
                g = self.gnn(data.x, data.edge_index, data.batch)
                x_train.append(g.detach())
                y_train.append(data.y)
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
        self.data, self.dataset = load4node_labelwise(self.args, self.dataset_name, train_ratio = 0.1, test_ratio = 0.8, device = self.device)
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