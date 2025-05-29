
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
from prompt_graph.evaluation import GPPTEva
from prompt_graph.utils import constraint,  center_embedding, Gprompt_tuning_loss
from torch_geometric.utils import erdos_renyi_graph
#%%
class Backdoor:
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
 
    def trojan_ASR_test_evaluation(self):
        split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(self.idx_test), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
        idx_clean_test = self.idx_test[split_trojan_evaluation['clean_test']]
        idx_atk = self.idx_test[split_trojan_evaluation['attack']]
        
        test_gnn = deepcopy(self.pretrain_gnn)
        test_prompt_generator = deepcopy(self.prompt_generator)

        poison_data = self.get_poisoned(idx_atk).to(self.device)
        # clean_data = deepcopy(self.data_ori)
        clean_data = self.get_clean().to(self.device)
        print("clean",clean_data.y)
        
        bkd_asr = GPPTEva(poison_data, idx_atk, test_gnn, test_prompt_generator)    
        print(bkd_asr)
        test_acc = GPPTEva(clean_data, idx_clean_test, test_gnn, test_prompt_generator)  
        print(test_acc)
            # test_acc = self.evaluate_ASR_1by1(clean_data, idx_clean_test)  
            # print(test_acc)
            # test_acc = self.evaluate_ASR_overall(clean_data, idx_clean_test)    
            # print(test_acc)  
        print("===========After Attack=====================")
        print("clean test accuracy {:.4f} ".format(test_acc))                        
        print("attack success rate {:.4f} ".format(bkd_asr))        
    
    def GPPT_fit(self, PG, gnn, data, idx_train, idx_trojan, idx_unlabeled):
        printN = 2
        self.idx_unlabeled = idx_unlabeled
        self.idx_trojan = idx_trojan
        self.pretrain_gnn = gnn.to(self.device)
        self.prompt_generator = PG.to(self.device)
        data = deepcopy(data).to(self.device)
        self.data_ori = deepcopy(data).to(self.device)
        self.data = deepcopy(data)
        self.features = data.x
        self.edge_index = data.edge_index

        self.criterion = torch.nn.CrossEntropyLoss()

        edge_weight = torch.ones([data.edge_index.shape[1]],device=self.device,dtype=torch.float)
        self.edge_weights = edge_weight
        
        


        # optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_PG = optim.Adam(self.prompt_generator.parameters(), lr=self.args.trojan_pg_lr, weight_decay=self.args.trojan_pg_weight_decay)

        # self.prompt_generator.TaskToken.requires_grad_(False)
        # self.prompt_generator.StructureToken.requires_grad_(False)
        # change the labels of the poisoned node to the target class
        self.labels = data.y.clone()
        self.labels[idx_trojan] = self.args.target_class

        # # get the trojan edges, which include the target-trigger edge and the edges among trigger
        # trojan_edge = self.get_trojan_edge(len(data.x),idx_trojan,self.args.trigger_size).to(self.device)

        # # update the poisoned graph's edge index
        # poison_edge_index = torch.cat([data.edge_index,trojan_edge],dim=1)


        # furture change it to bilevel optimization
        loss_best = 1e8
        for i in range(self.args.trojan_outter_epochs):
            self.prompt_generator.train()
            for j in range(self.args.trojan_inner_epochs):
                
                optimizer_PG.zero_grad()

                # trojan_feat, trojan_weights = self.trojan(data.x[idx_trojan],self.args.thrd) # may revise the process of generate
                poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger_rand(idx_trojan, data.x, data.edge_index)
                self.poison_x, self.poison_edge_index, self.poison_edge_weights = poison_x, poison_edge_index, poison_edge_weights
                node_embedding = self.pretrain_gnn(poison_x, poison_edge_index)
                out = self.prompt_generator(node_embedding, poison_edge_index)

                # loss_inner = self.criterion(out[torch.concat((idx_train,idx_trojan))], self.labels[torch.concat((idx_train,idx_trojan))])
                loss_inner_bkd = self.criterion(out[(idx_trojan)], self.labels[idx_trojan])
                loss_inner_clean = self.criterion(out[(idx_train)], self.labels[idx_train])
                loss_inner = self.args.loss_bkd_weight*loss_inner_bkd + loss_inner_clean

                loss = loss_inner
            
                # loss_inner = self.criterion(out[idx_trojan], self.labels[idx_trojan])
                # loss_inner = loss_inner + 0.001 * constraint(self.device, self.prompt_generator.get_TaskToken())
                # loss_inner = F.nll_loss(output[torch.cat([idx_train,idx_trojan])], self.labels[torch.cat([idx_train,idx_trojan])]) # add our adaptive loss
                
                loss.backward()
                optimizer_PG.step()
                self.prompt_generator.update_StructureToken_weight(self.prompt_generator.get_mid_h())
                self.trojan_ASR_test_evaluation()
            if((i) % printN == 0):
                print("{}/{} Loss inner: {:.4f} Loss inner bkd: {:.4f} Loss inner clean: {:.4f}".format(i, self.args.trojan_outter_epochs,loss_inner, loss_inner_bkd, loss_inner_clean))

    def GPPT_fit_freeze_DT(self, PG, gnn, data, idx_train, idx_trojan, idx_unlabeled):
        printN = 2
        self.idx_unlabeled = idx_unlabeled
        self.idx_trojan = idx_trojan
        self.pretrain_gnn = gnn.to(self.device)
        self.prompt_generator = PG.to(self.device)
        data = deepcopy(data).to(self.device)
        self.data_ori = deepcopy(data).to(self.device)
        self.data = deepcopy(data)
        self.features = data.x
        self.edge_index = data.edge_index

        self.criterion = torch.nn.CrossEntropyLoss()

        edge_weight = torch.ones([data.edge_index.shape[1]],device=self.device,dtype=torch.float)
        self.edge_weights = edge_weight
        
        

        # initalize a trojanNet to generate trigger

        # optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_PG = optim.Adam(self.prompt_generator.parameters(), lr=self.args.trojan_pg_lr, weight_decay=self.args.trojan_pg_weight_decay)

        self.prompt_generator.TaskToken.requires_grad_(False)
        # self.prompt_generator.StructureToken.requires_grad_(False)
        # change the labels of the poisoned node to the target class
        self.labels = data.y.clone()
        self.labels[idx_trojan] = self.args.target_class

        # # get the trojan edges, which include the target-trigger edge and the edges among trigger
        # trojan_edge = self.get_trojan_edge(len(data.x),idx_trojan,self.args.trigger_size).to(self.device)

        # # update the poisoned graph's edge index
        # poison_edge_index = torch.cat([data.edge_index,trojan_edge],dim=1)


        # furture change it to bilevel optimization
        loss_best = 1e8
        for i in range(self.args.trojan_outter_epochs):
            self.prompt_generator.train()
            for j in range(self.args.trojan_inner_epochs):
                
                optimizer_PG.zero_grad()

                # trojan_feat, trojan_weights = self.trojan(data.x[idx_trojan],self.args.thrd) # may revise the process of generate
                poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger_rand(idx_trojan, data.x, data.edge_index)
                self.poison_x, self.poison_edge_index, self.poison_edge_weights = poison_x, poison_edge_index, poison_edge_weights
                node_embedding = self.pretrain_gnn(poison_x, poison_edge_index)
                out = self.prompt_generator(node_embedding, poison_edge_index)

                # loss_inner = self.criterion(out[torch.concat((idx_train,idx_trojan))], self.labels[torch.concat((idx_train,idx_trojan))])
                loss_inner_bkd = self.criterion(out[(idx_trojan)], self.labels[idx_trojan])
                loss_inner_clean = self.criterion(out[(idx_train)], self.labels[idx_train])
                loss_inner = self.args.loss_bkd_weight*loss_inner_bkd + loss_inner_clean

                loss = loss_inner
            
                # loss_inner = self.criterion(out[idx_trojan], self.labels[idx_trojan])
                # loss_inner = loss_inner + 0.001 * constraint(self.device, self.prompt_generator.get_TaskToken())
                # loss_inner = F.nll_loss(output[torch.cat([idx_train,idx_trojan])], self.labels[torch.cat([idx_train,idx_trojan])]) # add our adaptive loss
                
                loss.backward()
                optimizer_PG.step()
                self.prompt_generator.update_StructureToken_weight(self.prompt_generator.get_mid_h())
                self.trojan_ASR_test_evaluation()
            if((i) % printN == 0):
                print("{}/{} Loss inner: {:.4f} Loss inner bkd: {:.4f} Loss inner clean: {:.4f}".format(i, self.args.trojan_outter_epochs,loss_inner, loss_inner_bkd, loss_inner_clean))
         

    def fit(self, prompt_type, PG, gnn, answering, data, idx_train, idx_trojan, idx_unlabeled, idx_test = None):
        self.idx_test = idx_test
        if(prompt_type == 'GPPT'):
            if(self.args.if_freeze_dt_classifier):
                self.GPPT_fit_freeze_DT(PG, gnn, data, idx_train, idx_trojan, idx_unlabeled)
            else:
                self.GPPT_fit(PG, gnn, data, idx_train, idx_trojan, idx_unlabeled)
        else:
            NotImplementedError("No implemented yet.")
    
    def get_poisoned(self, idx_atk):

        with torch.no_grad():
            # poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger(self.idx_trojan,self.features,self.edge_index,self.edge_weights,self.device)
            poison_x, poison_edge_index, _ = self.inject_trigger_rand(self.idx_trojan,self.poison_x.clone(), self.poison_edge_index.clone())
        poison_labels = self.data_ori.y.clone()
        poison_labels[idx_atk] = self.args.target_class

        poison_data = Data(x=poison_x, edge_index=poison_edge_index, y = poison_labels)
        # return poison_x, poison_edge_index, poison_edge_weights, poison_labels
        return poison_data
    
    def get_clean(self):


        with torch.no_grad():
            x, edge_index, edge_weights = self.data_ori.x.clone(),self.data_ori.edge_index.clone(),torch.ones([self.data_ori.edge_index.shape[1]],device=self.device,dtype=torch.float)
            # x, edge_index, edge_weights= self.poison_x.clone(), self.poison_edge_index.clone(), self.poison_edge_weights.clone()
        labels = self.data_ori.y.clone()
        edge_index = edge_index[:,edge_weights>0.0]
        edge_weights = edge_weights[edge_weights>0.0]

        clean_data = Data(x=x, edge_index=edge_index, y = labels)
        # return poison_x, poison_edge_index, poison_edge_weights, poison_labels
        return clean_data
