
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
#%%
class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input>thrd, torch.tensor(1.0, device=device, requires_grad=True),
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None



class GraphTrojanNet(nn.Module):
    # In the furture, we may use a GNN model to generate backdoor
    def __init__(self, device, nfeat, nout, nout_feat, layernum=1, dropout=0.00):
        super(GraphTrojanNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers.append(nn.Linear(nfeat, nfeat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        
        self.layers = nn.Sequential(*layers).to(device)

        self.feat = nn.Linear(nfeat,nout*nout_feat)
        self.edge = nn.Linear(nfeat, int(nout*(nout-1)/2))
        self.device = device

    def forward(self, input, thrd):

        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """

        GW = GradWhere.apply
        self.layers = self.layers
        h = self.layers(input)

        feat = self.feat(h)
        edge_weight = self.edge(h)
        # feat = GW(feat, thrd, self.device)
        edge_weight = GW(edge_weight, thrd, self.device)

        return feat, edge_weight
    
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
    
    def inject_trigger(self, idx_attach, features,edge_index,edge_weight,device):
        self.trojan = self.trojan.to(device)
        idx_attach = idx_attach.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        self.trojan.eval()

        trojan_feat, trojan_weights = self.trojan(features[idx_attach],self.args.thrd) # may revise the process of generate
        
        trojan_weights = torch.cat([torch.ones([len(idx_attach),1],dtype=torch.float,device=device),trojan_weights],dim=1)
        trojan_weights = trojan_weights.flatten()

        trojan_feat = trojan_feat.view([-1,features.shape[1]])

        trojan_edge = self.get_trojan_edge(len(features),idx_attach,self.args.trigger_size).to(device)

        update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
        update_feat = torch.cat([features,trojan_feat])
        update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

        self.trojan = self.trojan.cpu()
        idx_attach = idx_attach.cpu()
        features = features.cpu()
        edge_index = edge_index.cpu()
        edge_weight = edge_weight.cpu()
        return update_feat, update_edge_index, update_edge_weights

    def trojan_ASR_test_evaluation(self):
        split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(self.idx_test), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
        idx_clean_test = self.idx_test[split_trojan_evaluation['clean_test']]
        idx_atk = self.idx_test[split_trojan_evaluation['attack']]
        
        test_gnn = deepcopy(self.pretrain_gnn)
        test_prompt_generator = deepcopy(self.prompt_generator)

        poison_data = self.get_poisoned(idx_atk).to(self.device)
        # clean_data = deepcopy(self.data_ori)
        clean_data = self.get_clean().to(self.device)
        self.trojan = self.trojan.to(self.device)
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
        
        

        # initalize a trojanNet to generate trigger
        self.trojan = GraphTrojanNet(self.device, self.features.shape[1], self.args.trigger_size, self.features.shape[1], layernum=2).to(self.device)

        # optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_PG = optim.Adam(self.prompt_generator.parameters(), lr=self.args.trojan_pg_lr, weight_decay=self.args.trojan_pg_weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=self.args.trigger_lr, weight_decay=self.args.trigger_weight_decay)

        # self.prompt_generator.TaskToken.requires_grad_(False)
        self.prompt_generator.StructureToken.requires_grad_(False)
        # change the labels of the poisoned node to the target class
        self.labels = data.y.clone()
        self.labels[idx_trojan] = self.args.target_class

        # get the trojan edges, which include the target-trigger edge and the edges among trigger
        trojan_edge = self.get_trojan_edge(len(data.x),idx_trojan,self.args.trigger_size).to(self.device)

        # update the poisoned graph's edge index
        poison_edge_index = torch.cat([data.edge_index,trojan_edge],dim=1)


        # furture change it to bilevel optimization
        loss_best = 1e8
        for i in range(self.args.trojan_outter_epochs):
            self.trojan.train()
            self.prompt_generator.train()
            for j in range(self.args.trojan_inner_epochs):
                
                optimizer_PG.zero_grad()
                optimizer_trigger.zero_grad()
                trojan_feat, trojan_weights = self.trojan(data.x[idx_trojan],self.args.thrd) # may revise the process of generate
                trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
                trojan_weights = trojan_weights.flatten()
                trojan_feat = trojan_feat.view([-1,data.x.shape[1]])
                poison_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights]).detach() # repeat trojan weights beacuse of undirected edge
                poison_x = torch.cat([data.x,trojan_feat]).detach()

                updated_poison_edge_index = poison_edge_index[:,poison_edge_weights>0.0]
                poison_edge_weights = poison_edge_weights[poison_edge_weights>0.0]
                self.poison_x, self.poison_edge_index, self.poison_edge_weights = poison_x, updated_poison_edge_index, poison_edge_weights
                node_embedding = self.pretrain_gnn(poison_x, updated_poison_edge_index)
                out = self.prompt_generator(node_embedding, updated_poison_edge_index)

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
                optimizer_trigger.step()
                # self.prompt_generator.update_StructureToken_weight(self.prompt_generator.get_mid_h())
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
        self.trojan = GraphTrojanNet(self.device, self.features.shape[1], self.args.trigger_size, self.features.shape[1], layernum=2).to(self.device)

        # optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_PG = optim.Adam(self.prompt_generator.parameters(), lr=self.args.trojan_pg_lr, weight_decay=self.args.trojan_pg_weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=self.args.trigger_lr, weight_decay=self.args.trigger_weight_decay)

        self.prompt_generator.TaskToken.requires_grad_(False)
        self.prompt_generator.StructureToken.requires_grad_(False)
        # change the labels of the poisoned node to the target class
        self.labels = data.y.clone()
        self.labels[idx_trojan] = self.args.target_class

        # get the trojan edges, which include the target-trigger edge and the edges among trigger
        trojan_edge = self.get_trojan_edge(len(data.x),idx_trojan,self.args.trigger_size).to(self.device)

        # update the poisoned graph's edge index
        poison_edge_index = torch.cat([data.edge_index,trojan_edge],dim=1)


        # furture change it to bilevel optimization
        loss_best = 1e8
        for i in range(self.args.trojan_outter_epochs):
            self.trojan.train()
            self.prompt_generator.train()
            for j in range(self.args.trojan_inner_epochs):
                
                optimizer_PG.zero_grad()
                optimizer_trigger.zero_grad()
                trojan_feat, trojan_weights = self.trojan(data.x[idx_trojan],self.args.thrd) # may revise the process of generate
                trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
                trojan_weights = trojan_weights.flatten()
                trojan_feat = trojan_feat.view([-1,data.x.shape[1]])
                poison_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights]).detach() # repeat trojan weights beacuse of undirected edge
                poison_x = torch.cat([data.x,trojan_feat]).detach()

                updated_poison_edge_index = poison_edge_index[:,poison_edge_weights>0.0]
                poison_edge_weights = poison_edge_weights[poison_edge_weights>0.0]
                self.poison_x, self.poison_edge_index, self.poison_edge_weights = poison_x, updated_poison_edge_index, poison_edge_weights
                node_embedding = self.pretrain_gnn(poison_x, updated_poison_edge_index)
                out = self.prompt_generator(node_embedding, updated_poison_edge_index)

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
                optimizer_trigger.step()
                # self.prompt_generator.update_StructureToken_weight(self.prompt_generator.get_mid_h())
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
            poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger(self.idx_trojan,self.poison_x.clone(), self.poison_edge_index.clone(), self.poison_edge_weights.clone(),self.device)
        poison_labels = self.data_ori.y.clone()
        poison_labels[idx_atk] = self.args.target_class
        poison_edge_index = poison_edge_index[:,poison_edge_weights>0.0]
        poison_edge_weights = poison_edge_weights[poison_edge_weights>0.0]

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
