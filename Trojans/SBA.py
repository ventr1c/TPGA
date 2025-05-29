#%%
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# import utils
import Trojans.trojan_utils as trojan_utils

from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.utils import erdos_renyi_graph
from prompt_graph.utils import constraint,  center_embedding, Gprompt_tuning_loss
#%%
   
class RandomBackdoor:
    def __init__(self,args, num_classes, device):
        self.args = args
        self.device = device
        self.trigger_weights = None
        self.trigger_index = self.get_trigger_index(args.trigger_size)
        self.num_classes = num_classes

    def get_trigger_index(self,trigger_size):
        '''
        get the edge index of the inner structure of the trigger
        '''
        edge_list = []
        edge_list.append([0,0])
        for j in range(trigger_size):
            for k in range(j):
                edge_list.append([j,k])
        edge_index = torch.tensor(edge_list,device=self.device).long().T
        return edge_index
     
    def get_trojan_edge(self,start, idx_attach, trigger_size):
        edge_list = []
        for idx in idx_attach:
            edges = self.trigger_index.clone()
            edges[0,0] = idx
            edges[1,0] = start
            edges[:,1:] = edges[:,1:] + start

            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list,dim=1)
        # to undirected
        # row, col = edge_index
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1],edge_index[0]])
        edge_index = torch.stack([row,col])

        return edge_index
    def gene_trigger(self,features):
        trojan_edge_index = erdos_renyi_graph(self.args.trigger_size,edge_prob=self.args.trigger_prob).to(self.device)
        rs = np.random.RandomState(self.args.seed)
        if self.args.baseline_trojan_feat_generation == 'Rand_Gene':
            # print("Rand generate the trigger")
            features = features.cpu().numpy()
            mean = features.mean(axis=0)
            std = features.std(axis=0)

            trojan_feat = []
            for i in range(self.args.trigger_size):
                trojan_feat.append(torch.tensor(rs.normal(mean,std),dtype=torch.float32,device=self.device))
            trojan_feat = torch.stack(trojan_feat)
        else:
            # print("Rand sample the trigger")
            idx = rs.randint(features.shape[0],size=self.args.trigger_size)
            trojan_feat = features[idx]

        return trojan_feat, trojan_edge_index
    
    def get_poisoned_rand(self,features,edge_index,labels,idx_attach):

        trojan_labels = labels.clone()
        trojan_labels[idx_attach] = self.args.target_class
        trojan_features,trojan_edge_index,weights = self.inject_trigger_rand(idx_attach,features,edge_index)


        return trojan_features,trojan_edge_index,weights,trojan_labels

    def inject_trigger_rand(self, idx_attach, features, edge_index):
        trigger_feat, trigger_edge_index = self.gene_trigger(features)
        
        edge_list = []
        start = features.shape[0]
        for i,idx in enumerate(idx_attach):
            edge_list.append([idx,start+i*self.args.trigger_size])
        trojan_edge_index = torch.tensor(edge_list,device=self.device,dtype=torch.long).T
        for i in range(len(idx_attach)):
            tmp_edge_index = trigger_edge_index.clone()
            tmp_edge_index[0] = start + i*self.args.trigger_size + tmp_edge_index[0]
            tmp_edge_index[1] = start + i*self.args.trigger_size + tmp_edge_index[1]
            trojan_edge_index = torch.cat([trojan_edge_index,tmp_edge_index],dim=1)
        
        
        from torch_geometric.utils import to_undirected
        trojan_edge_index = to_undirected(trojan_edge_index)
        trojan_edge_index = torch.cat([edge_index,trojan_edge_index],dim=1)

        trojan_features = trigger_feat.repeat([len(idx_attach),1])
        trojan_features = torch.cat([features,trojan_features])
        weights= torch.ones([trojan_edge_index.shape[1]],dtype=torch.float32,device=self.device)

        return trojan_features,trojan_edge_index,weights
        

    def fit(self, prompt_type, PG, gnn, answering, support_dataset, idx_trojan, dt = None):
        if(prompt_type == 'Gprompt'):
            if(self.args.if_freeze_dt_classifier):
                centers = self.Gprompt_fit_freeze_DT(PG, gnn, support_dataset, idx_trojan, center = dt)
                return centers
            else:
                centers, accumulated_centers, accumulated_counts = self.Gprompt_fit(PG, gnn, support_dataset, idx_trojan, center = dt)
                return centers, accumulated_centers, accumulated_counts
        else:
            NotImplementedError("No implemented yet.")
    
    def Gprompt_fit(self, PG, gnn, support_dataset, idx_trojan, center = None):
        '''
        prompt_model: the in-context learning model contain prompt graphs and downstream tasker
        '''
        printN = 5
        self.support_dataset = support_dataset.to(self.device)
        
        # split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(support_dataset), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
        # idx_sup, idx_que = split_trojan_evaluation['clean_test'], split_trojan_evaluation['attack']
        
        self.support_dataset = support_dataset
        self.query_dataset = support_dataset

        self.idx_trojan = idx_trojan

        self.pretrain_gnn = gnn.to(self.device)
        self.prompt_generator = PG.to(self.device)
        # self.downstream_tasker = answering.to(self.device)

        # optimizer_DT = optim.Adam(self.downstream_tasker.parameters(), lr=self.args.trojan_dt_lr,weight_decay=self.args.trojan_dt_weight_decay)

        tobe_poison_sup_dataset = self.support_dataset[idx_trojan]

        trojan_sup_dataset = []
        for data in tobe_poison_sup_dataset:
            data = data.to(self.device)
            edge_weights = torch.ones([data.edge_index.shape[1]],device=self.device,dtype=torch.float)
            
            # data_batch = Batch.from_data_list([data])
            # prompted_graph = self.prompt_generator(data_batch)
            # graph_emb = self.pretrain_gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)


            idx_attach = trojan_utils.obtain_attach_nodes(seed=self.args.seed,node_idxs=torch.LongTensor(range(data.x.shape[0])),size=1)
            poison_x, poison_edge_index, _ = self.inject_trigger_rand(idx_attach, data.x, data.edge_index)
            poison_x, poison_edge_index = poison_x.to(self.device), poison_edge_index.to(self.device)
            poison_y = torch.LongTensor([self.args.target_class]).to(self.device)
            poison_data = Data(x=poison_x, edge_index=poison_edge_index, y=poison_y).to(self.device)
            trojan_sup_dataset.append(poison_data)
        # Trigger optimization
        poisoned_sup_data_list = deepcopy(self.support_dataset.to_data_list())
        poisoned_sup_data_list.extend(trojan_sup_dataset)
        
        loss_best = 1e8
        for i in range(self.args.trojan_outter_epochs):
            '''Alternative Training: Train PG and then Trigger Generator'''            
            print("{}/{} epochs ...".format(i+1, self.args.trojan_outter_epochs))
            self.prompt_generator.eval()
            # self.downstream_tasker.eval()
            
            # Trigger optimization: inject trigger into poison sample


            poisoned_sup_dataloader = DataLoader(poisoned_sup_data_list, batch_size=10, shuffle=False)


            accumulated_centers = None
            accumulated_counts = None
            loss_sup_batch = 0.

            for batch_id, data in enumerate(poisoned_sup_dataloader):
                data = data.to(self.device)
                graph_emb = self.pretrain_gnn(data.x, data.edge_index, data.batch, prompt = self.prompt_generator, prompt_type = 'Gprompt')
                # dt_predict = self.downstream_tasker(graph_emb)
                center, class_counts = center_embedding(graph_emb, data.y, self.num_classes)

                # optimizer_DT.zero_grad()
                if accumulated_centers is None:
                    accumulated_centers = center
                    accumulated_counts = class_counts
                else:
                    accumulated_centers += center * class_counts
                    accumulated_counts += class_counts
                criterion = Gprompt_tuning_loss()
                loss_sup = criterion(graph_emb, center.detach(), data.y)  
                
                loss_sup.backward()

                loss_sup_batch += loss_sup
            if((i) % printN == 0):
                print("{}/{} Loss PG: {:.4f}".format(i, self.args.trojan_inner_epochs,loss_sup_batch))
        
        mean_centers = accumulated_centers / accumulated_counts
        return mean_centers, accumulated_centers, accumulated_counts
    
    def AllinOne_fit(self, PG, gnn, answering, support_dataset, idx_trojan):
        '''
        prompt_model: the in-context learning model contain prompt graphs and downstream tasker
        '''
        printN = 5
        self.support_dataset = support_dataset.to(self.device)
        
        # split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(support_dataset), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
        # idx_sup, idx_que = split_trojan_evaluation['clean_test'], split_trojan_evaluation['attack']
        
        self.support_dataset = support_dataset
        self.query_dataset = support_dataset

        self.idx_trojan = idx_trojan

        self.pretrain_gnn = gnn.to(self.device)
        self.prompt_generator = PG.to(self.device)
        self.downstream_tasker = answering.to(self.device)

        optimizer_PG = optim.Adam(self.prompt_generator.parameters(), lr=self.args.trojan_pg_lr, weight_decay=self.args.trojan_pg_weight_decay)
        optimizer_DT = optim.Adam(self.downstream_tasker.parameters(), lr=self.args.trojan_dt_lr,weight_decay=self.args.trojan_dt_weight_decay)

        tobe_poison_sup_dataset = self.support_dataset[idx_trojan]

        trojan_sup_dataset = []
        for data in tobe_poison_sup_dataset:
            data = data.to(self.device)
            edge_weights = torch.ones([data.edge_index.shape[1]],device=self.device,dtype=torch.float)
            
            # data_batch = Batch.from_data_list([data])
            # prompted_graph = self.prompt_generator(data_batch)
            # graph_emb = self.pretrain_gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)


            idx_attach = trojan_utils.obtain_attach_nodes(seed=self.args.seed,node_idxs=torch.LongTensor(range(data.x.shape[0])),size=1)
            poison_x, poison_edge_index, _ = self.inject_trigger_rand(idx_attach, data.x, data.edge_index)
            poison_x, poison_edge_index = poison_x.to(self.device), poison_edge_index.to(self.device)
            poison_y = torch.LongTensor([self.args.target_class]).to(self.device)
            poison_data = Data(x=poison_x, edge_index=poison_edge_index, y=poison_y).to(self.device)
            trojan_sup_dataset.append(poison_data)
        # Trigger optimization
        poisoned_sup_data_list = deepcopy(self.support_dataset.to_data_list())
        poisoned_sup_data_list.extend(trojan_sup_dataset)
        
        
        loss_best = 1e8
        for i in range(self.args.trojan_outter_epochs):
            '''Alternative Training: Train PG and then Trigger Generator'''            
            print("{}/{} epochs ...".format(i+1, self.args.trojan_outter_epochs))
            self.prompt_generator.train()
            self.downstream_tasker.eval()
            
            # Trigger optimization: inject trigger into poison sample


            poisoned_sup_dataloader = DataLoader(poisoned_sup_data_list, batch_size=10, shuffle=False)

            loss_sup_batch = 0.
            for batch_id, data in enumerate(poisoned_sup_dataloader):
                data = data.to(self.device)

                # data_batch = Batch.from_data_list([data])
                prompted_graph = self.prompt_generator(data)
                graph_emb = self.pretrain_gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
                dt_predict = self.downstream_tasker(graph_emb)

                # optimizer_PG.zero_grad()
                optimizer_DT.zero_grad()
                loss_sup = F.cross_entropy(dt_predict,data.y)
                loss_sup.backward()
                # optimizer_PG.step()
                optimizer_DT.step()

                loss_sup_batch += loss_sup
            if((i) % printN == 0):
                print("{}/{} Loss PG: {:.4f}".format(i, self.args.trojan_inner_epochs,loss_sup_batch))
            
            
        # self.trojan.eval()
        # self.prompt_generator.eval()

                
    
    def transfer_to_trojan_test_dataset(self, dataset,idx_atk, prompt_type, device=None):
        dataset = dataset.to(self.device)
        tobe_poison_dataset = dataset[idx_atk]

        trojan_dataset = []
        for data in tobe_poison_dataset:
            data = data.to(self.device)
            idx_attach = trojan_utils.obtain_attach_nodes(seed=self.args.seed,node_idxs=np.array(list(range(data.x.shape[0]))),size=1)
            poison_x, poison_edge_index, _ = self.inject_trigger_rand(idx_attach, data.x, data.edge_index)
            poison_x, poison_edge_index = poison_x.to(self.device), poison_edge_index.to(self.device)
            poison_y = torch.LongTensor([self.args.target_class]).to(self.device)
            poison_data = Data(x=poison_x, edge_index=poison_edge_index, y=poison_y).to(self.device)

            poison_data = Data(x=poison_x, edge_index=poison_edge_index, y=poison_y).to(self.device)
            trojan_dataset.append(poison_data)
        # poisoned_data_list = deepcopy(dataset.to_data_list())
        # poisoned_data_list.extend(trojan_dataset)
        # return poisoned_data_list
        return trojan_dataset
        
    def evaluate(self,prompt_module, gnn, test_dataset):
        test_dataset = test_dataset.to(self.device)
        # decide the attack and clean test graph index for evaluation
        split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(test_dataset), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
        idx_clean_test, idx_atk = split_trojan_evaluation['clean_test'], split_trojan_evaluation['attack']

        clean_test_dataset = test_dataset[idx_clean_test]
        attack_dataset = test_dataset[idx_atk]
        
        '''
        TODO
        '''
        # Evaluate clean test accuracy
        