
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
from Trojans.swd import swd

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

class HomoLoss(nn.Module):
    def __init__(self,args,device):
        super(HomoLoss, self).__init__()
        self.args = args
        self.device = device
        
    def forward(self,trigger_edge_index,trigger_edge_weights,x,thrd):

        trigger_edge_index = trigger_edge_index[:,trigger_edge_weights>0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]],x[trigger_edge_index[1]])
        
        loss = torch.relu(thrd - edge_sims).mean()
        # print(edge_sims.min())
        return loss

class Backdoor:
    def __init__(self,args, num_classes, device):
        self.args = args
        self.device = device
        self.trigger_weights = None
        self.trigger_index = self.get_trigger_index(args.trigger_size)
        self.num_classes = num_classes
        self.homo_loss = HomoLoss(self.args,self.device)
        
        if(args.dataset_name == 'Cora'):
            self.homo_boost_thrd = 0.5
        elif(args.dataset_name == 'Citeseer'):
            self.homo_boost_thrd = 0.5
        elif(args.dataset_name == 'Pubmed'):
            self.homo_boost_thrd = 0.5
            
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
    
    def Gprompt_fit(self, PG, gnn, support_dataset, idx_trojan, center):
        '''
        prompt_model: the in-context learning model contain prompt graphs and downstream tasker
        '''
        printN = 5
        self.support_dataset = support_dataset.to(self.device)
        
        split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(support_dataset), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
        idx_sup, idx_que = split_trojan_evaluation['clean_test'], split_trojan_evaluation['attack']
        
        self.support_dataset = support_dataset
        self.query_dataset = support_dataset
        self.idx_trojan = idx_trojan
        self.pretrain_gnn = gnn.to(self.device)
        self.trojan = GraphTrojanNet(self.device, self.args.hid_dim, self.args.trigger_size, support_dataset[0].x.shape[1], layernum=2).to(self.device)
        self.prompt_generator = PG.to(self.device)

        optimizer_PG = optim.Adam(self.prompt_generator.parameters(), lr=self.args.trojan_pg_lr, weight_decay=self.args.trojan_pg_weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=self.args.trigger_lr, weight_decay=self.args.trigger_weight_decay)

        tobe_poison_sup_dataset = self.support_dataset[idx_trojan]
        num_poison = int(len(self.query_dataset) * self.args.poison_ratio)
        idx_trojan_que = trojan_utils.obtain_attach_nodes(seed=10,node_idxs=np.array(range(len(self.support_dataset))),size=num_poison)
        tobe_poison_que_dataset = self.query_dataset[idx_trojan_que]

        loss_best = 1e8
        for i in range(self.args.trojan_outter_epochs):
            '''Alternative Training: Train PG and then Trigger Generator'''            
            if((i) % printN == 0):
                print("{}/{} *tune trigger generator | frozen gnn | frozen prompt".format(i+1, self.args.trojan_outter_epochs))
            self.trojan.train() 
            self.prompt_generator.train()
            for j in range(self.args.trojan_inner_epochs):
                # Trigger optimization: inject trigger into poison sample
                trojan_sup_dataset = []
                for data in tobe_poison_sup_dataset:
                    data = data.to(self.device)
                    edge_weights = torch.ones([data.edge_index.shape[1]],device=self.device,dtype=torch.float)
                    data_batch = Batch.from_data_list([data])

                    graph_emb = self.pretrain_gnn(data_batch.x, data_batch.edge_index, data_batch.batch, prompt = self.prompt_generator, prompt_type = 'Gprompt')

                    # get the trojan edges, which include the target-trigger edge and the edges among trigger
                    # we randomly select a index for attaching trigger into the graph or just select the last node index
                    idx_attach = trojan_utils.obtain_attach_nodes(seed=self.args.seed,node_idxs=torch.LongTensor(range(data.x.shape[0])),size=1)
                    trojan_edges = self.get_trojan_edge(data.x.shape[0], idx_attach=idx_attach, trigger_size=self.args.trigger_size).to(self.device)
                    # inject trigger edge index to clean graph's edges
                    poison_edge_index = torch.cat([data.edge_index,trojan_edges],dim=1)
                    trojan_feat, trojan_weights = self.trojan(graph_emb,self.args.thrd)

                    trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
                    trojan_weights = trojan_weights.flatten()
                    trojan_feat = trojan_feat.view([-1,data.x.shape[1]])
                    poison_edge_weights = torch.cat([edge_weights,trojan_weights,trojan_weights]).detach() # repeat trojan weights beacuse of undirected edge
                    poison_x = torch.cat([data.x,trojan_feat]).detach().to(self.device)

                    # filter poison edge index based on edge weights
                    poison_edge_index = poison_edge_index[:,poison_edge_weights.nonzero().flatten().long()]
                    poison_edge_weights = torch.ones([poison_edge_index.shape[1],]).to(self.device)
                    poison_y = torch.LongTensor([self.args.target_class]).to(self.device)

                    poison_data = Data(x=poison_x, edge_index=poison_edge_index, y=poison_y).to(self.device)
                    trojan_sup_dataset.append(poison_data)
                # Trigger optimization
                poisoned_sup_data_list = deepcopy(self.support_dataset.to_data_list())
                poisoned_sup_data_list.extend(trojan_sup_dataset)


                poisoned_sup_dataloader = DataLoader(poisoned_sup_data_list, batch_size=10, shuffle=False)
                tobe_poison_sup_dataloader = DataLoader(tobe_poison_sup_dataset, batch_size=10, shuffle=False)
                loss_sup_batch = 0.
                loss_meta_batch = 0.

                accumulated_centers = None
                accumulated_counts = None

                add_constraint = False

                if(add_constraint):
                    benign_sup_data_list = deepcopy(self.support_dataset.to_data_list())
                    benign_sup_dataloader = DataLoader(benign_sup_data_list, batch_size=10, shuffle=False)
                    trojan_sup_dataloader = DataLoader(trojan_sup_dataset, batch_size=10, shuffle=False)
                    loss_dist_batch = 0.0
                    loss_dist_total = 0.0
                    loss_sup_batch = 0.0
                    for batch_id, data in enumerate(trojan_sup_dataset):
                        # data = data.to(self.device)
                        before_data = self.support_dataset[idx_trojan[batch_id]].to(self.device)

                        data = Batch.from_data_list([data])
                        before_data =  Batch.from_data_list([before_data])
                        data = data.to(self.device)

                        before_graph_emb = self.pretrain_gnn(before_data.x, before_data.edge_index, before_data.batch, prompt = self.prompt_generator, prompt_type = 'Gprompt')

                        graph_emb = self.pretrain_gnn(data.x, data.edge_index, data.batch, prompt = self.prompt_generator, prompt_type = 'Gprompt')
                        center, class_counts = center_embedding(graph_emb, data.y, self.num_classes)

                        optimizer_PG.zero_grad()
                        optimizer_trigger.zero_grad()

                        # dist_criterion = nn.MSELoss()
                        trojan_feat, trojan_weights = self.trojan(before_graph_emb,self.args.thrd)
                        trojan_feat = trojan_feat.view([-1,data.x.shape[1]])
                        if trojan_feat.shape[0] < before_data.x.shape[0]:
                            num_trojan_feat = trojan_feat.shape[0]
                            selected_indices = torch.randperm(before_data.x.shape[0])[:num_trojan_feat].to(self.device)
                            before_data_x_selected = before_data.x[selected_indices,:]
                            loss_dist = swd(trojan_feat,before_data_x_selected, device = self.device)
                        else:
                            num_trojan_feat = trojan_feat.shape[0]
                            num_x = before_data.x.shape[0]
                            selected_indices = torch.randperm(trojan_feat.shape[0])[:num_x].to(self.device)
                            trojan_feat_selected = trojan_feat[selected_indices,:]
                            loss_dist = swd(trojan_feat_selected,before_data.x, device = self.device)

                        if accumulated_centers is None:
                            accumulated_centers = center
                            accumulated_counts = class_counts
                        else:
                            accumulated_centers += center * class_counts
                            accumulated_counts += class_counts
                        criterion = Gprompt_tuning_loss()
                        loss_sup = criterion(graph_emb, center, data.y)  

                        loss = loss_dist
                        # print(loss_sup,loss_dist)
                        loss.backward()
                        optimizer_PG.step()
                        optimizer_trigger.step()
                        loss_sup_batch += loss_sup
                        loss_dist_batch += loss_dist
                    if((i) % printN == 0):
                        print("{}/{} Loss Trojan: {:.4f}, {:.4f}".format(j, self.args.trojan_inner_epochs,loss_sup_batch, loss_dist_batch))
                    
                    for batch_id, data in enumerate(poisoned_sup_dataloader):
                        data = data.to(self.device)
                        graph_emb = self.pretrain_gnn(data.x, data.edge_index, data.batch, prompt = self.prompt_generator, prompt_type = 'Gprompt')
                        # dt_predict = self.downstream_tasker(graph_emb)
                        center, class_counts = center_embedding(graph_emb, data.y, self.num_classes)

                        optimizer_PG.zero_grad()
                        optimizer_trigger.zero_grad()
                        if accumulated_centers is None:
                            accumulated_centers = center
                            accumulated_counts = class_counts
                        else:
                            accumulated_centers += center * class_counts
                            accumulated_counts += class_counts
                        criterion = Gprompt_tuning_loss()
                        loss_sup = criterion(graph_emb, center.detach(), data.y)  
                        loss_sup.backward()
                        optimizer_PG.step()
                        optimizer_trigger.step()

                        loss_sup_batch += loss_sup
                    if((i) % printN == 0):
                        print("{}/{} Loss PG: {:.4f}".format(j, self.args.trojan_inner_epochs,loss_sup_batch))
                else:
                    for batch_id, data in enumerate(poisoned_sup_dataloader):
                        data = data.to(self.device)
                        graph_emb = self.pretrain_gnn(data.x, data.edge_index, data.batch, prompt = self.prompt_generator, prompt_type = 'Gprompt')
                        # dt_predict = self.downstream_tasker(graph_emb)
                        center, class_counts = center_embedding(graph_emb, data.y, self.num_classes)

                        optimizer_PG.zero_grad()
                        optimizer_trigger.zero_grad()
                        if accumulated_centers is None:
                            accumulated_centers = center
                            accumulated_counts = class_counts
                        else:
                            accumulated_centers += center * class_counts
                            accumulated_counts += class_counts
                        criterion = Gprompt_tuning_loss()
                        loss_sup = criterion(graph_emb, center.detach(), data.y)  
                        # loss_meta = self.meta_backdoor_Gprompt(data, criterion, self.args.eps, self.args.step, self.args.random)
                        loss_meta = 0
                        loss = loss_sup
                        # loss = loss_sup  + self.args.alpha * loss_meta
                        loss.backward()
                        optimizer_PG.step()
                        optimizer_trigger.step()

                        loss_sup_batch += loss_sup
                        loss_meta_batch += loss_meta
                    if((i) % printN == 0):
                        # print("{}/{} Loss PG: {:.4f}".format(j, self.args.trojan_inner_epochs,loss_sup_batch))
                        print("{}/{} Loss PG: {:.4f} Meta: {:.4f}".format(j, self.args.trojan_inner_epochs,loss_sup_batch, loss_meta_batch))
            
            # print("{}/{} * tune trigger generator | frozen gnn | frozen prompt |frozen answering function...".format(i+1, self.args.trojan_outter_epochs))
    
        mean_centers = accumulated_centers / accumulated_counts
        return mean_centers, accumulated_centers, accumulated_counts
        # return None
    def Gprompt_fit_freeze_DT(self, PG, gnn, support_dataset, idx_trojan, center):
        '''
        prompt_model: the in-context learning model contain prompt graphs and downstream tasker
        '''
        printN = 5
        self.support_dataset = support_dataset.to(self.device)
        
        split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(support_dataset), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
        idx_sup, idx_que = split_trojan_evaluation['clean_test'], split_trojan_evaluation['attack']
        
        self.support_dataset = support_dataset
        self.query_dataset = support_dataset
        self.idx_trojan = idx_trojan
        self.pretrain_gnn = gnn.to(self.device)
        self.trojan = GraphTrojanNet(self.device, self.args.hid_dim, self.args.trigger_size, support_dataset[0].x.shape[1], layernum=2).to(self.device)
        self.prompt_generator = PG.to(self.device)

        optimizer_PG = optim.Adam(self.prompt_generator.parameters(), lr=self.args.trojan_pg_lr, weight_decay=self.args.trojan_pg_weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=self.args.trigger_lr, weight_decay=self.args.trigger_weight_decay)

        tobe_poison_sup_dataset = self.support_dataset[idx_trojan]
        num_poison = int(len(self.query_dataset) * self.args.poison_ratio)
        idx_trojan_que = trojan_utils.obtain_attach_nodes(seed=10,node_idxs=np.array(range(len(self.support_dataset))),size=num_poison)
        tobe_poison_que_dataset = self.query_dataset[idx_trojan_que]

        loss_best = 1e8
        for i in range(self.args.trojan_outter_epochs):
            '''Alternative Training: Train PG and then Trigger Generator'''            
            if((i) % printN == 0):
                print("{}/{} *tune trigger generator | frozen gnn | frozen prompt".format(i+1, self.args.trojan_outter_epochs))
            self.trojan.train() 
            self.prompt_generator.train()
            for j in range(self.args.trojan_inner_epochs):
                # Trigger optimization: inject trigger into poison sample
                trojan_sup_dataset = []
                for data in tobe_poison_sup_dataset:
                    data = data.to(self.device)
                    edge_weights = torch.ones([data.edge_index.shape[1]],device=self.device,dtype=torch.float)
                    data_batch = Batch.from_data_list([data])

                    graph_emb = self.pretrain_gnn(data_batch.x, data_batch.edge_index, data_batch.batch, prompt = self.prompt_generator, prompt_type = 'Gprompt')

                    # get the trojan edges, which include the target-trigger edge and the edges among trigger
                    # we randomly select a index for attaching trigger into the graph or just select the last node index
                    idx_attach = trojan_utils.obtain_attach_nodes(seed=self.args.seed,node_idxs=torch.LongTensor(range(data.x.shape[0])),size=1)
                    trojan_edges = self.get_trojan_edge(data.x.shape[0], idx_attach=idx_attach, trigger_size=self.args.trigger_size).to(self.device)
                    # inject trigger edge index to clean graph's edges
                    poison_edge_index = torch.cat([data.edge_index,trojan_edges],dim=1)
                    trojan_feat, trojan_weights = self.trojan(graph_emb,self.args.thrd)

                    trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
                    trojan_weights = trojan_weights.flatten()
                    trojan_feat = trojan_feat.view([-1,data.x.shape[1]])
                    poison_edge_weights = torch.cat([edge_weights,trojan_weights,trojan_weights]).detach() # repeat trojan weights beacuse of undirected edge
                    poison_x = torch.cat([data.x,trojan_feat]).detach().to(self.device)

                    # filter poison edge index based on edge weights
                    poison_edge_index = poison_edge_index[:,poison_edge_weights.nonzero().flatten().long()]
                    poison_edge_weights = torch.ones([poison_edge_index.shape[1],]).to(self.device)
                    poison_y = torch.LongTensor([self.args.target_class]).to(self.device)

                    poison_data = Data(x=poison_x, edge_index=poison_edge_index, y=poison_y).to(self.device)
                    trojan_sup_dataset.append(poison_data)
                # Trigger optimization
                poisoned_sup_data_list = deepcopy(self.support_dataset.to_data_list())
                poisoned_sup_data_list.extend(trojan_sup_dataset)


                poisoned_sup_dataloader = DataLoader(poisoned_sup_data_list, batch_size=10, shuffle=False)
                tobe_poison_sup_dataloader = DataLoader(tobe_poison_sup_dataset, batch_size=10, shuffle=False)
                loss_sup_batch = 0.

                accumulated_centers = None
                accumulated_counts = None

                add_constraint = False

                if(add_constraint):
                    benign_sup_data_list = deepcopy(self.support_dataset.to_data_list())
                    benign_sup_dataloader = DataLoader(benign_sup_data_list, batch_size=10, shuffle=False)
                    trojan_sup_dataloader = DataLoader(trojan_sup_dataset, batch_size=10, shuffle=False)
                    loss_dist_batch = 0.0
                    loss_dist_total = 0.0
                    loss_sup_batch = 0.0
                    for batch_id, data in enumerate(trojan_sup_dataset):
                        # data = data.to(self.device)
                        before_data = self.support_dataset[idx_trojan[batch_id]].to(self.device)

                        data = Batch.from_data_list([data])
                        before_data =  Batch.from_data_list([before_data])
                        data = data.to(self.device)

                        before_graph_emb = self.pretrain_gnn(before_data.x, before_data.edge_index, before_data.batch, prompt = self.prompt_generator, prompt_type = 'Gprompt')

                        graph_emb = self.pretrain_gnn(data.x, data.edge_index, data.batch, prompt = self.prompt_generator, prompt_type = 'Gprompt')
                        # _, class_counts = center_embedding(graph_emb, data.y, self.num_classes)

                        optimizer_PG.zero_grad()
                        optimizer_trigger.zero_grad()

                        # dist_criterion = nn.MSELoss()
                        trojan_feat, trojan_weights = self.trojan(before_graph_emb,self.args.thrd)
                        trojan_feat = trojan_feat.view([-1,data.x.shape[1]])
                        if trojan_feat.shape[0] < before_data.x.shape[0]:
                            num_trojan_feat = trojan_feat.shape[0]
                            selected_indices = torch.randperm(before_data.x.shape[0])[:num_trojan_feat].to(self.device)
                            before_data_x_selected = before_data.x[selected_indices,:]
                            loss_dist = swd(trojan_feat,before_data_x_selected, device = self.device)
                        else:
                            num_trojan_feat = trojan_feat.shape[0]
                            num_x = before_data.x.shape[0]
                            selected_indices = torch.randperm(trojan_feat.shape[0])[:num_x].to(self.device)
                            trojan_feat_selected = trojan_feat[selected_indices,:]
                            loss_dist = swd(trojan_feat_selected,before_data.x, device = self.device)

                        # if accumulated_centers is None:
                        #     accumulated_centers = center
                        #     accumulated_counts = class_counts
                        # else:
                        #     accumulated_centers += center * class_counts
                        #     accumulated_counts += class_counts
                        criterion = Gprompt_tuning_loss()
                        loss_sup = criterion(graph_emb, center, data.y)  

                        loss = loss_dist
                        # print(loss_sup,loss_dist)
                        loss.backward()
                        optimizer_PG.step()
                        optimizer_trigger.step()
                        loss_sup_batch += loss_sup
                        loss_dist_batch += loss_dist
                    if((i) % printN == 0):
                        print("{}/{} Loss Trojan: {:.4f}, {:.4f}".format(j, self.args.trojan_inner_epochs,loss_sup_batch, loss_dist_batch))
                    
                    for batch_id, data in enumerate(poisoned_sup_dataloader):
                        data = data.to(self.device)
                        graph_emb = self.pretrain_gnn(data.x, data.edge_index, data.batch, prompt = self.prompt_generator, prompt_type = 'Gprompt')
                        # dt_predict = self.downstream_tasker(graph_emb)
                        _, class_counts = center_embedding(graph_emb, data.y, self.num_classes)

                        optimizer_PG.zero_grad()
                        optimizer_trigger.zero_grad()
                        # optimizer_DT.zero_grad()
                        # if accumulated_centers is None:
                        #     accumulated_centers = center
                        #     accumulated_counts = class_counts
                        # else:
                        #     accumulated_centers += center * class_counts
                        #     accumulated_counts += class_counts
                        criterion = Gprompt_tuning_loss()
                        loss_sup = criterion(graph_emb, center.detach(), data.y)  
                        loss_sup.backward()
                        optimizer_PG.step()
                        optimizer_trigger.step()

                        loss_sup_batch += loss_sup
                    if((i) % printN == 0):
                        print("{}/{} Loss PG: {:.4f}".format(j, self.args.trojan_inner_epochs,loss_sup_batch))
                else:
                    for batch_id, data in enumerate(poisoned_sup_dataloader):
                        data = data.to(self.device)
                        graph_emb = self.pretrain_gnn(data.x, data.edge_index, data.batch, prompt = self.prompt_generator, prompt_type = 'Gprompt')
                        # dt_predict = self.downstream_tasker(graph_emb)
                        _, class_counts = center_embedding(graph_emb, data.y, self.num_classes)

                        optimizer_PG.zero_grad()
                        optimizer_trigger.zero_grad()
                        # optimizer_DT.zero_grad()
                        # if accumulated_centers is None:
                        #     accumulated_centers = center
                        #     accumulated_counts = class_counts
                        # else:
                        #     accumulated_centers += center * class_counts
                        #     accumulated_counts += class_counts
                        criterion = Gprompt_tuning_loss()
                        loss_sup = criterion(graph_emb, center.detach(), data.y)  
                        loss_sup.backward()
                        optimizer_PG.step()
                        optimizer_trigger.step()

                        loss_sup_batch += loss_sup
                    if((i) % printN == 0):
                        print("{}/{} Loss PG: {:.4f}".format(j, self.args.trojan_inner_epochs,loss_sup_batch))
            
            # print("{}/{} * tune trigger generator | frozen gnn | frozen prompt |frozen answering function...".format(i+1, self.args.trojan_outter_epochs))
    
        # mean_centers = accumulated_centers / accumulated_counts
        # return mean_centers
        return None

    def GPPT_fit(self, PG, gnn, support_dataset, idx_trojan):
        '''
        prompt_model: the in-context learning model contain prompt graphs and downstream tasker
        '''
        printN = 5
        self.support_dataset = support_dataset.to(self.device)
        
        split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(support_dataset), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
        idx_sup, idx_que = split_trojan_evaluation['clean_test'], split_trojan_evaluation['attack']
        
        self.support_dataset = support_dataset
        self.query_dataset = support_dataset
        self.idx_trojan = idx_trojan
        self.pretrain_gnn = gnn.to(self.device)
        self.trojan = GraphTrojanNet(self.device, self.args.hid_dim, self.args.trigger_size, support_dataset[0].x.shape[1], layernum=2).to(self.device)
        self.prompt_generator = PG.to(self.device)

        optimizer_PG = optim.Adam(self.prompt_generator.parameters(), lr=self.args.trojan_pg_lr, weight_decay=self.args.trojan_pg_weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=self.args.trigger_lr, weight_decay=self.args.trigger_weight_decay)

        tobe_poison_sup_dataset = self.support_dataset[idx_trojan]
        num_poison = int(len(self.query_dataset) * 1)
        idx_trojan_que = trojan_utils.obtain_attach_nodes(seed=10,node_idxs=np.array(range(len(self.support_dataset))),size=num_poison)
        tobe_poison_que_dataset = self.query_dataset[idx_trojan_que]

        loss_best = 1e8
        for i in range(self.args.trojan_outter_epochs):
            '''Alternative Training: Train PG and then Trigger Generator'''            
            if((i) % printN == 0):
                print("{}/{} *tune trigger generator | frozen gnn | frozen prompt".format(i+1, self.args.trojan_outter_epochs))
            self.trojan.train() 
            self.prompt_generator.train()
            for j in range(self.args.trojan_inner_epochs):
                # Trigger optimization: inject trigger into poison sample
                trojan_sup_dataset = []
                for data in tobe_poison_sup_dataset:
                    data = data.to(self.device)
                    edge_weights = torch.ones([data.edge_index.shape[1]],device=self.device,dtype=torch.float)
                    data_batch = Batch.from_data_list([data])

                    graph_emb = self.pretrain_gnn(data_batch.x, data_batch.edge_index, data_batch.batch, prompt = self.prompt_generator, prompt_type = 'GPPT')

                    # get the trojan edges, which include the target-trigger edge and the edges among trigger
                    # we randomly select a index for attaching trigger into the graph or just select the last node index
                    idx_attach = trojan_utils.obtain_attach_nodes(seed=self.args.seed,node_idxs=torch.LongTensor(range(data.x.shape[0])),size=1)
                    trojan_edges = self.get_trojan_edge(data.x.shape[0], idx_attach=idx_attach, trigger_size=self.args.trigger_size).to(self.device)
                    # inject trigger edge index to clean graph's edges
                    poison_edge_index = torch.cat([data.edge_index,trojan_edges],dim=1)
                    trojan_feat, trojan_weights = self.trojan(graph_emb,self.args.thrd)

                    trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
                    trojan_weights = trojan_weights.flatten()
                    trojan_feat = trojan_feat.view([-1,data.x.shape[1]])
                    poison_edge_weights = torch.cat([edge_weights,trojan_weights,trojan_weights]).detach() # repeat trojan weights beacuse of undirected edge
                    poison_x = torch.cat([data.x,trojan_feat]).detach().to(self.device)

                    # filter poison edge index based on edge weights
                    poison_edge_index = poison_edge_index[:,poison_edge_weights.nonzero().flatten().long()]
                    poison_edge_weights = torch.ones([poison_edge_index.shape[1],]).to(self.device)
                    poison_y = torch.LongTensor([self.args.target_class]).to(self.device)

                    poison_data = Data(x=poison_x, edge_index=poison_edge_index, y=poison_y).to(self.device)
                    trojan_sup_dataset.append(poison_data)
                # Trigger optimization
                poisoned_sup_data_list = deepcopy(self.support_dataset.to_data_list())
                poisoned_sup_data_list.extend(trojan_sup_dataset)


                poisoned_sup_dataloader = DataLoader(poisoned_sup_data_list, batch_size=10, shuffle=False)
                tobe_poison_sup_dataloader = DataLoader(tobe_poison_sup_dataset, batch_size=10, shuffle=False)
                loss_sup_batch = 0.

                for batch_id, data in enumerate(poisoned_sup_dataloader):
                    data = data.to(self.device)

                    # data_batch = Batch.from_data_list([data])
                    out = self.prompt_generator(data)
                    # graph_emb = self.pretrain_gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
                    # dt_predict = self.downstream_tasker(graph_emb)

                    optimizer_PG.zero_grad()
                    optimizer_trigger.zero_grad()
                    # optimizer_DT.zero_grad()
                    loss_sup = F.cross_entropy(out,data.y)
                    
                    loss_sup.backward()
                    optimizer_PG.step()
                    optimizer_trigger.step()
                    
                    # optimizer_DT.step()

                    loss_sup_batch += loss_sup
                if((j) % printN == 0):
                    print("{}/{} Loss PG: {:.4f}".format(j, self.args.trojan_inner_epochs,loss_sup_batch))

    def fit(self, prompt_type, PG, gnn, answering, support_dataset, idx_trojan, dt = None):
        if(prompt_type == 'Gprompt'):
            if(self.args.if_freeze_dt_classifier):
                centers = self.Gprompt_fit_freeze_DT(PG, gnn, support_dataset, idx_trojan, center = dt)
                return centers
            else:
                centers, accumulated_centers, accumulated_counts = self.Gprompt_fit(PG, gnn, support_dataset, idx_trojan, center = dt)
                return centers, accumulated_centers, accumulated_counts
        elif(prompt_type in ['GPF','GPF-plus']):
            self.GPF_fit(PG, gnn, answering, support_dataset, idx_trojan, prompt_type)
        else:
            NotImplementedError("No implemented yet.")

    
    def GPF_fit(self, PG, gnn, answering, support_dataset, idx_trojan, prompt_type):
        '''
        prompt_model: the in-context learning model contain prompt graphs and downstream tasker
        '''
        printN = 5
        self.support_dataset = support_dataset.to(self.device)
        
        split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(support_dataset), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
        idx_sup, idx_que = split_trojan_evaluation['clean_test'], split_trojan_evaluation['attack']
        
        self.support_dataset = support_dataset
        self.query_dataset = support_dataset

        idx_sup_full = torch.LongTensor(range(len(support_dataset)))
        # idx_que_full = torch.LongTensor(range(len(query_dataset)))
        self.idx_trojan = idx_trojan

        self.pretrain_gnn = gnn.to(self.device)
        # self.prompt_module = prompt_module.to(self.device)
        self.prompt_generator = PG.to(self.device)
        self.downstream_tasker = answering.to(self.device)
        self.trojan = GraphTrojanNet(self.device, self.args.hid_dim, self.args.trigger_size, support_dataset[0].x.shape[1], layernum=2).to(self.device)

        optimizer_PG = optim.Adam(self.prompt_generator.parameters(), lr=self.args.trojan_pg_lr, weight_decay=self.args.trojan_pg_weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=self.args.trigger_lr, weight_decay=self.args.trigger_weight_decay)
        optimizer_DT = optim.Adam(self.downstream_tasker.parameters(), lr=self.args.trojan_dt_lr,weight_decay=self.args.trojan_dt_weight_decay)

        tobe_poison_sup_dataset = self.support_dataset[idx_trojan]
        num_poison = int(len(self.query_dataset) * 0.5)
        idx_trojan_que = trojan_utils.obtain_attach_nodes(seed=10,node_idxs=np.array(range(len(self.support_dataset))),size=num_poison)
        tobe_poison_que_dataset = self.query_dataset[idx_trojan_que]

        loss_best = 1e8

        for i in range(self.args.trojan_outter_epochs):
            '''Alternative Training: Train PG and then Trigger Generator'''            
            print("{}/{} frozen trigger generator | frozen gnn | *poison prompt |frozen answering function...".format(i+1, self.args.trojan_outter_epochs))
            self.trojan.eval() 
            self.prompt_generator.train()
            self.downstream_tasker.eval()
            for j in range(self.args.trojan_inner_epochs):
                # Trigger optimization: inject trigger into poison sample
                trojan_sup_dataset = []
                for data in tobe_poison_sup_dataset:
                    data = data.to(self.device)
                    edge_weights = torch.ones([data.edge_index.shape[1]],device=self.device,dtype=torch.float)
                    
                    data_batch = Batch.from_data_list([data])
                    data_batch.x = self.prompt_generator.add(data_batch.x)
                    graph_emb = self.pretrain_gnn(data_batch.x, data_batch.edge_index, data_batch.batch, prompt=self.prompt_generator, prompt_type = prompt_type)

                    # get the trojan edges, which include the target-trigger edge and the edges among trigger
                    # we randomly select a index for attaching trigger into the graph or just select the last node index
                    idx_attach = trojan_utils.obtain_attach_nodes(seed=self.args.seed,node_idxs=torch.LongTensor(range(data.x.shape[0])),size=1)
                    trojan_edges = self.get_trojan_edge(data.x.shape[0], idx_attach=idx_attach, trigger_size=self.args.trigger_size).to(self.device)
                    # inject trigger edge index to clean graph's edges
                    poison_edge_index = torch.cat([data.edge_index,trojan_edges],dim=1)
                    trojan_feat, trojan_weights = self.trojan(graph_emb,self.args.thrd)

                    trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
                    trojan_weights = trojan_weights.flatten()
                    trojan_feat = trojan_feat.view([-1,data.x.shape[1]])
                    poison_edge_weights = torch.cat([edge_weights,trojan_weights,trojan_weights]).detach() # repeat trojan weights beacuse of undirected edge
                    poison_x = torch.cat([data.x,trojan_feat]).detach().to(self.device)

                    # filter poison edge index based on edge weights
                    poison_edge_index = poison_edge_index[:,poison_edge_weights.nonzero().flatten().long()]
                    poison_edge_weights = torch.ones([poison_edge_index.shape[1],]).to(self.device)
                    poison_y = torch.LongTensor([self.args.target_class]).to(self.device)

                    poison_data = Data(x=poison_x, edge_index=poison_edge_index, y=poison_y).to(self.device)
                    trojan_sup_dataset.append(poison_data)
                # Trigger optimization
                poisoned_sup_data_list = deepcopy(self.support_dataset.to_data_list())
                poisoned_sup_data_list.extend(trojan_sup_dataset)


                poisoned_sup_dataloader = DataLoader(poisoned_sup_data_list, batch_size=10, shuffle=False)

                loss_sup_batch = 0.
                for data in (poisoned_sup_dataloader):
                    data = data.to(self.device)

                    # data_batch = Batch.from_data_list([data])
                    data.x = self.prompt_generator.add(data.x)
                    # prompted_graph = self.prompt_generator(data)
                    graph_emb = self.pretrain_gnn(data.x, data.edge_index, data.batch, prompt = self.prompt_generator, prompt_type = prompt_type)
                    dt_predict = self.downstream_tasker(graph_emb)

                    optimizer_PG.zero_grad()
                    optimizer_trigger.zero_grad()
                    # optimizer_DT.zero_grad()
                    loss_sup = F.cross_entropy(dt_predict,data.y)
                    
                    loss_sup.backward()
                    optimizer_PG.step()
                    optimizer_trigger.step()
                    # optimizer_DT.step()

                    loss_sup_batch += loss_sup
                if((j) % printN == 0):
                    print("{}/{} Loss PG: {:.4f}".format(j, self.args.trojan_inner_epochs,loss_sup_batch))
    
            # print("{}/{} * tune trigger generator | frozen gnn | frozen prompt |frozen answering function...".format(i+1, self.args.trojan_outter_epochs))
            # self.trojan.train()
            # self.prompt_generator.eval()
            # self.downstream_tasker.eval()

            # for j in range(self.args.trojan_inner_epochs):
            #     # Trigger optimization: inject trigger into poison sample
            #     trojan_sup_dataset = []
            #     for data in tobe_poison_que_dataset:
            #         data = data.to(self.device)
            #         edge_weights = torch.ones([data.edge_index.shape[1]],device=self.device,dtype=torch.float)
                    
            #         data_batch = Batch.from_data_list([data])
            #         data_batch.x = self.prompt_generator.add(data_batch.x)
            #         graph_emb = self.pretrain_gnn(data_batch.x, data_batch.edge_index, data_batch.batch, self.prompt_generator, prompt_type)

            #         # get the trojan edges, which include the target-trigger edge and the edges among trigger
            #         # we randomly select a index for attaching trigger into the graph or just select the last node index
            #         idx_attach = trojan_utils.obtain_attach_nodes(seed=self.args.seed,node_idxs=torch.LongTensor(range(data.x.shape[0])),size=1)
            #         trojan_edges = self.get_trojan_edge(data.x.shape[0], idx_attach=idx_attach, trigger_size=self.args.trigger_size).to(self.device)
            #         # inject trigger edge index to clean graph's edges
            #         poison_edge_index = torch.cat([data.edge_index,trojan_edges],dim=1)
            #         trojan_feat, trojan_weights = self.trojan(graph_emb,self.args.thrd)

            #         trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
            #         trojan_weights = trojan_weights.flatten()
            #         trojan_feat = trojan_feat.view([-1,data.x.shape[1]])
            #         poison_edge_weights = torch.cat([edge_weights,trojan_weights,trojan_weights]).detach() # repeat trojan weights beacuse of undirected edge
            #         poison_x = torch.cat([data.x,trojan_feat]).detach().to(self.device)

            #         # filter poison edge index based on edge weights
            #         poison_edge_index = poison_edge_index[:,poison_edge_weights.nonzero().flatten().long()]
            #         poison_edge_weights = torch.ones([poison_edge_index.shape[1],]).to(self.device)
            #         poison_y = torch.LongTensor([self.args.target_class]).to(self.device)

            #         poison_data = Data(x=poison_x, edge_index=poison_edge_index, y=poison_y).to(self.device)
            #         trojan_sup_dataset.append(poison_data)
            #     # Trigger optimization
            #     poisoned_sup_data_list = deepcopy(self.query_dataset.to_data_list())
            #     poisoned_sup_data_list.extend(trojan_sup_dataset)


            #     poisoned_sup_dataloader = DataLoader(poisoned_sup_data_list, batch_size=10, shuffle=False)

            #     loss_sup_batch = 0.
            #     for batch_id, data in enumerate(poisoned_sup_dataloader):
            #         data = data.to(self.device)

            #         # data_batch = Batch.from_data_list([data])
            #         data.x = self.prompt_generator.add(data.x)
            #         # prompted_graph = self.prompt_generator(data)
            #         graph_emb = self.pretrain_gnn(data.x, data.edge_index, data.batch, prompt = self.prompt_generator, prompt_type = prompt_type)
            #         dt_predict = self.downstream_tasker(graph_emb)

            #         optimizer_trigger.zero_grad()
            #         # optimizer_PG.zero_grad()

            #         loss_sup = F.cross_entropy(dt_predict,data.y)
            #         loss_sup.backward()
            #         optimizer_trigger.step()
            #         # optimizer_PG.step()

            #         loss_sup_batch += loss_sup
            #     if((j) % printN == 0):
            #         print("\t\t\t{}/{} Loss Trigger: {:.4f}".format(j, self.args.trojan_inner_epochs,loss_sup_batch))
        # self.trojan.eval()
        # self.prompt_generator.eval()
    def AllinOne_fit(self, PG, gnn, answering, support_dataset, idx_trojan):
        '''
        prompt_model: the in-context learning model contain prompt graphs and downstream tasker
        '''
        printN = 5
        self.support_dataset = support_dataset.to(self.device)
        
        split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(support_dataset), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
        idx_sup, idx_que = split_trojan_evaluation['clean_test'], split_trojan_evaluation['attack']
        
        self.support_dataset = support_dataset
        self.query_dataset = support_dataset
        # self.support_dataset = Batch.from_data_list(support_dataset[idx_sup])
        # self.query_dataset = Batch.from_data_list(support_dataset[idx_que])
        # num_poison = int(len(self.support_dataset) * 0.5)
        # print("Number of poisoning graphs: {}".format(num_poison))
        # idx_trojan = trojan_utils.obtain_attach_nodes(seed=10,node_idxs=np.array(range(len(self.support_dataset))),size=num_poison)
        # print("Selected trojan graphs in train dataset: {}".format(idx_trojan))

        idx_sup_full = torch.LongTensor(range(len(support_dataset)))
        # idx_que_full = torch.LongTensor(range(len(query_dataset)))
        self.idx_trojan = idx_trojan

        self.pretrain_gnn = gnn.to(self.device)
        # self.prompt_module = prompt_module.to(self.device)
        self.prompt_generator = PG.to(self.device)
        self.downstream_tasker = answering.to(self.device)
        self.trojan = GraphTrojanNet(self.device, self.args.hid_dim, self.args.trigger_size, layernum=2).to(self.device)

        optimizer_PG = optim.Adam(self.prompt_generator.parameters(), lr=self.args.trojan_pg_lr, weight_decay=self.args.trojan_pg_weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=self.args.trigger_lr, weight_decay=self.args.trigger_weight_decay)
        optimizer_DT = optim.Adam(self.downstream_tasker.parameters(), lr=self.args.trojan_dt_lr,weight_decay=self.args.trojan_dt_weight_decay)

        tobe_poison_sup_dataset = self.support_dataset[idx_trojan]
        num_poison = int(len(self.query_dataset) * 0.2)
        idx_trojan_que = trojan_utils.obtain_attach_nodes(seed=10,node_idxs=np.array(range(len(self.support_dataset))),size=num_poison)
        tobe_poison_que_dataset = self.query_dataset[idx_trojan_que]

        loss_best = 1e8
        for i in range(self.args.trojan_outter_epochs):
            '''Alternative Training: Train PG and then Trigger Generator'''            
            print("{}/{} frozen trigger generator | frozen gnn | *poison prompt |frozen answering function...".format(i+1, self.args.trojan_outter_epochs))
            self.trojan.train() 
            self.prompt_generator.train()
            self.downstream_tasker.eval()
            for j in range(self.args.trojan_inner_epochs):
                # Trigger optimization: inject trigger into poison sample
                trojan_sup_dataset = []
                for data in tobe_poison_sup_dataset:
                    data = data.to(self.device)
                    edge_weights = torch.ones([data.edge_index.shape[1]],device=self.device,dtype=torch.float)
                    
                    data_batch = Batch.from_data_list([data])
                    prompted_graph = self.prompt_generator(data_batch)
                    graph_emb = self.pretrain_gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)

                    # get the trojan edges, which include the target-trigger edge and the edges among trigger
                    # we randomly select a index for attaching trigger into the graph or just select the last node index
                    idx_attach = trojan_utils.obtain_attach_nodes(seed=self.args.seed,node_idxs=torch.LongTensor(range(data.x.shape[0])),size=1)
                    trojan_edges = self.get_trojan_edge(data.x.shape[0], idx_attach=idx_attach, trigger_size=self.args.trigger_size).to(self.device)
                    # inject trigger edge index to clean graph's edges
                    poison_edge_index = torch.cat([data.edge_index,trojan_edges],dim=1)
                    trojan_feat, trojan_weights = self.trojan(graph_emb,self.args.thrd)

                    trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
                    trojan_weights = trojan_weights.flatten()
                    trojan_feat = trojan_feat.view([-1,data.x.shape[1]])
                    poison_edge_weights = torch.cat([edge_weights,trojan_weights,trojan_weights]).detach() # repeat trojan weights beacuse of undirected edge
                    poison_x = torch.cat([data.x,trojan_feat]).detach().to(self.device)

                    # filter poison edge index based on edge weights
                    poison_edge_index = poison_edge_index[:,poison_edge_weights.nonzero().flatten().long()]
                    poison_edge_weights = torch.ones([poison_edge_index.shape[1],]).to(self.device)
                    poison_y = torch.LongTensor([self.args.target_class]).to(self.device)

                    poison_data = Data(x=poison_x, edge_index=poison_edge_index, y=poison_y).to(self.device)
                    trojan_sup_dataset.append(poison_data)
                # Trigger optimization
                poisoned_sup_data_list = deepcopy(self.support_dataset.to_data_list())
                poisoned_sup_data_list.extend(trojan_sup_dataset)


                poisoned_sup_dataloader = DataLoader(poisoned_sup_data_list, batch_size=10, shuffle=False)

                loss_sup_batch = 0.
                for batch_id, data in enumerate(poisoned_sup_dataloader):
                    data = data.to(self.device)

                    # data_batch = Batch.from_data_list([data])
                    prompted_graph = self.prompt_generator(data)
                    graph_emb = self.pretrain_gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
                    dt_predict = self.downstream_tasker(graph_emb)

                    optimizer_PG.zero_grad()
                    optimizer_trigger.zero_grad()
                    # optimizer_DT.zero_grad()
                    loss_sup = F.cross_entropy(dt_predict,data.y)
                    
                    loss_sup.backward()
                    optimizer_PG.step()
                    optimizer_trigger.step()
                    # optimizer_DT.step()

                    loss_sup_batch += loss_sup
                if((j) % printN == 0):
                    print("{}/{} Loss PG: {:.4f}".format(j, self.args.trojan_inner_epochs,loss_sup_batch))
            
            # print("{}/{} * tune trigger generator | frozen gnn | frozen prompt |frozen answering function...".format(i+1, self.args.trojan_outter_epochs))
        #     self.trojan.train()
        #     self.prompt_generator.eval()
        #     self.downstream_tasker.eval()

        #     for j in range(self.args.trojan_inner_epochs):
        #         # Trigger optimization: inject trigger into poison sample
        #         trojan_sup_dataset = []
        #         for data in tobe_poison_que_dataset:
        #             data = data.to(self.device)
        #             edge_weights = torch.ones([data.edge_index.shape[1]],device=self.device,dtype=torch.float)
                    
        #             data_batch = Batch.from_data_list([data])
        #             prompted_graph = self.prompt_generator(data_batch)
        #             graph_emb = self.pretrain_gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)

        #             # get the trojan edges, which include the target-trigger edge and the edges among trigger
        #             # we randomly select a index for attaching trigger into the graph or just select the last node index
        #             idx_attach = trojan_utils.obtain_attach_nodes(seed=self.args.seed,node_idxs=torch.LongTensor(range(data.x.shape[0])),size=1)
        #             trojan_edges = self.get_trojan_edge(data.x.shape[0], idx_attach=idx_attach, trigger_size=self.args.trigger_size).to(self.device)
        #             # inject trigger edge index to clean graph's edges
        #             poison_edge_index = torch.cat([data.edge_index,trojan_edges],dim=1)
        #             trojan_feat, trojan_weights = self.trojan(graph_emb,self.args.thrd)

        #             trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
        #             trojan_weights = trojan_weights.flatten()
        #             trojan_feat = trojan_feat.view([-1,data.x.shape[1]])
        #             poison_edge_weights = torch.cat([edge_weights,trojan_weights,trojan_weights]).detach() # repeat trojan weights beacuse of undirected edge
        #             poison_x = torch.cat([data.x,trojan_feat]).detach().to(self.device)

        #             # filter poison edge index based on edge weights
        #             poison_edge_index = poison_edge_index[:,poison_edge_weights.nonzero().flatten().long()]
        #             poison_edge_weights = torch.ones([poison_edge_index.shape[1],]).to(self.device)
        #             poison_y = torch.LongTensor([self.args.target_class]).to(self.device)

        #             poison_data = Data(x=poison_x, edge_index=poison_edge_index, y=poison_y).to(self.device)
        #             trojan_sup_dataset.append(poison_data)
        #         # Trigger optimization
        #         poisoned_sup_data_list = deepcopy(self.query_dataset.to_data_list())
        #         poisoned_sup_data_list.extend(trojan_sup_dataset)


        #         poisoned_sup_dataloader = DataLoader(poisoned_sup_data_list, batch_size=10, shuffle=False)

        #         loss_sup_batch = 0.
        #         for batch_id, data in enumerate(poisoned_sup_dataloader):
        #             data = data.to(self.device)

        #             # data_batch = Batch.from_data_list([data])
        #             prompted_graph = self.prompt_generator(data)
        #             graph_emb = self.pretrain_gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
        #             dt_predict = self.downstream_tasker(graph_emb)

        #             optimizer_trigger.zero_grad()
        #             loss_sup = F.cross_entropy(dt_predict,data.y)
        #             loss_sup.backward()
        #             optimizer_trigger.step()

        #             loss_sup_batch += loss_sup
        #         if((j) % printN == 0):
        #             print("\t\t\t{}/{} Loss Trigger: {:.4f}".format(j, self.args.trojan_inner_epochs,loss_sup_batch))
        # # self.trojan.eval()
        # # self.prompt_generator.eval()
    
    def inject_trigger_to_dataset(self, dataset, idx_attach):
        dataset = dataset.to(self.device)
        tobe_poison_dataset = dataset[idx_attach]

        trojan_dataset = []
        for data in tobe_poison_dataset:
            data = data.to(self.device)
            edge_weights = torch.ones([data.edge_index.shape[1]],device=self.device,dtype=torch.float)
            data_batch = Batch.from_data_list([data])
            prompted_graph = self.prompt_generator(data_batch)
            graph_emb = self.pretrain_gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            # we randomly select a index for attaching trigger into the graph or just select the last node index
            idx_attach = trojan_utils.obtain_attach_nodes(seed=self.args.seed,node_idxs=np.array(list(range(data.x.shape[0]))),size=1)
            trojan_edges = self.get_trojan_edge(data.x.shape[0], idx_attach=idx_attach, trigger_size=self.args.trigger_size).to(self.device)
            # inject trigger edge index to clean graph's edges
            poison_edge_index = torch.cat([data.edge_index,trojan_edges],dim=1)
            trojan_feat, trojan_weights = self.trojan(graph_emb,self.args.thrd)

            trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
            trojan_weights = trojan_weights.flatten()
            trojan_feat = trojan_feat.view([-1,data.x.shape[1]])
            poison_edge_weights = torch.cat([edge_weights,trojan_weights,trojan_weights]).detach() # repeat trojan weights beacuse of undirected edge
            poison_x = torch.cat([data.x,trojan_feat]).detach().to(self.device)

            # filter poison edge index based on edge weights
            poison_edge_index = poison_edge_index[:,poison_edge_weights.nonzero().flatten().long()].to(self.device)
            poison_edge_weights = torch.ones([poison_edge_index.shape[1],]).to(self.device)
            poison_y = torch.LongTensor([self.args.target_class]).to(self.device)

            poison_data = Data(x=poison_x, edge_index=poison_edge_index, y=poison_y).to(self.device)
            trojan_dataset.append(poison_data)
            # Explicitly delete variables and free memory
            del data, edge_weights, data_batch, prompted_graph, graph_emb, idx_attach, trojan_edges
            del poison_edge_index, trojan_feat, trojan_weights, poison_edge_weights, poison_x, poison_y, poison_data
            torch.cuda.empty_cache()
            
        poisoned_data_list = deepcopy(dataset.to_data_list())
        poisoned_data_list.extend(trojan_dataset)
        return poisoned_data_list
    
    def transfer_to_trojan_test_dataset(self, dataset,idx_atk, prompt_type, device = None):
        # if(device == None):
        #     dataset = dataset.to(self.device)
        #     tobe_poison_dataset = dataset[idx_atk]

        #     trojan_dataset = []
        #     for data in tobe_poison_dataset:
        #         data = data.to(self.device)
        #         edge_weights = torch.ones([data.edge_index.shape[1]],device=self.device,dtype=torch.float)
        #         data_batch = Batch.from_data_list([data])

        #         graph_emb = self.pretrain_gnn(data_batch.x, data_batch.edge_index, data_batch.batch, prompt = self.prompt_generator, prompt_type = prompt_type)

        #         # get the trojan edges, which include the target-trigger edge and the edges among trigger
        #         # we randomly select a index for attaching trigger into the graph or just select the last node index
        #         idx_attach = trojan_utils.obtain_attach_nodes(seed=self.args.seed,node_idxs=torch.LongTensor(range(data.x.shape[0])),size=1)
        #         trojan_edges = self.get_trojan_edge(data.x.shape[0], idx_attach=idx_attach, trigger_size=self.args.trigger_size).to(self.device)

        #         # inject trigger edge index to clean graph's edges
        #         poison_edge_index = torch.cat([data.edge_index,trojan_edges],dim=1)
        #         trojan_feat, trojan_weights = self.trojan(graph_emb,self.args.thrd)

        #         trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
        #         trojan_weights = trojan_weights.flatten()
        #         trojan_feat = trojan_feat.view([-1,data.x.shape[1]])
        #         poison_edge_weights = torch.cat([edge_weights,trojan_weights,trojan_weights]).detach() # repeat trojan weights beacuse of undirected edge
        #         poison_x = torch.cat([data.x,trojan_feat]).detach().to(self.device)

        #         # filter poison edge index based on edge weights
        #         poison_edge_index = poison_edge_index[:,poison_edge_weights.nonzero().flatten().long()].to(self.device)
        #         poison_edge_weights = torch.ones([poison_edge_index.shape[1],]).to(self.device)
        #         poison_y = torch.LongTensor([self.args.target_class]).to(self.device)

        #         poison_data = Data(x=poison_x, edge_index=poison_edge_index, y=poison_y).to(self.device)
        #         trojan_dataset.append(poison_data)
        #     # poisoned_data_list = deepcopy(dataset.to_data_list())
        #     # poisoned_data_list.extend(trojan_dataset)
        #     # return poisoned_data_list
        #     return trojan_dataset
        # else:
        dataset = dataset.to(self.device)
        tobe_poison_dataset = dataset[idx_atk]

        trojan_dataset = []
        for data in tobe_poison_dataset:
            data = data.to(self.device)
            edge_weights = torch.ones([data.edge_index.shape[1]],device=self.device,dtype=torch.float)
            data_batch = Batch.from_data_list([data])

            graph_emb = self.pretrain_gnn(data_batch.x, data_batch.edge_index, data_batch.batch, prompt = self.prompt_generator, prompt_type = prompt_type)

            # get the trojan edges, which include the target-trigger edge and the edges among trigger
            # we randomly select a index for attaching trigger into the graph or just select the last node index
            idx_attach = trojan_utils.obtain_attach_nodes(seed=self.args.seed,node_idxs=torch.LongTensor(range(data.x.shape[0])),size=1)
            trojan_edges = self.get_trojan_edge(data.x.shape[0], idx_attach=idx_attach, trigger_size=self.args.trigger_size).to(self.device)

            # inject trigger edge index to clean graph's edges
            poison_edge_index = torch.cat([data.edge_index,trojan_edges],dim=1)
            trojan_feat, trojan_weights = self.trojan(graph_emb,self.args.thrd)

            trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
            trojan_weights = trojan_weights.flatten()
            trojan_feat = trojan_feat.view([-1,data.x.shape[1]])
            poison_edge_weights = torch.cat([edge_weights,trojan_weights,trojan_weights]).detach() # repeat trojan weights beacuse of undirected edge
            poison_x = torch.cat([data.x,trojan_feat]).detach().cpu()

            # filter poison edge index based on edge weights
            poison_edge_index = poison_edge_index[:,poison_edge_weights.nonzero().flatten().long()].cpu()
            poison_edge_weights = torch.ones([poison_edge_index.shape[1],]).cpu()
            poison_y = torch.LongTensor([self.args.target_class]).cpu()

            poison_data = Data(x=poison_x, edge_index=poison_edge_index, y=poison_y).cpu()
            trojan_dataset.append(poison_data)
            del data, edge_weights, data_batch, graph_emb, idx_attach, trojan_edges
            del poison_edge_index, trojan_feat, trojan_weights, poison_edge_weights, poison_x, poison_y, poison_data
            torch.cuda.empty_cache()
        # poisoned_data_list = deepcopy(dataset.to_data_list())
        # poisoned_data_list.extend(trojan_dataset)
        # return poisoned_data_list
        return trojan_dataset


    def AllinOne_transfer_to_trojan_test_dataset(self, dataset,idx_atk):
        dataset = dataset.to(self.device)
        tobe_poison_dataset = dataset[idx_atk]

        trojan_dataset = []
        for data in tobe_poison_dataset:
            data = data.to(self.device)
            edge_weights = torch.ones([data.edge_index.shape[1]],device=self.device,dtype=torch.float)
            data_batch = Batch.from_data_list([data])
            prompted_graph = self.prompt_generator(data_batch)
            graph_emb = self.pretrain_gnn(prompted_graph.x, prompted_graph.edge_index, prompted_graph.batch)
            # we randomly select a index for attaching trigger into the graph or just select the last node index
            idx_attach = trojan_utils.obtain_attach_nodes(seed=self.args.seed,node_idxs=np.array(list(range(data.x.shape[0]))),size=1)
            trojan_edges = self.get_trojan_edge(data.x.shape[0], idx_attach=idx_attach, trigger_size=self.args.trigger_size).to(self.device)
            # inject trigger edge index to clean graph's edges
            poison_edge_index = torch.cat([data.edge_index,trojan_edges],dim=1)
            trojan_feat, trojan_weights = self.trojan(graph_emb,self.args.thrd)

            trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
            trojan_weights = trojan_weights.flatten()
            trojan_feat = trojan_feat.view([-1,data.x.shape[1]])
            poison_edge_weights = torch.cat([edge_weights,trojan_weights,trojan_weights]).detach() # repeat trojan weights beacuse of undirected edge
            poison_x = torch.cat([data.x,trojan_feat]).detach().to(self.device)

            # filter poison edge index based on edge weights
            poison_edge_index = poison_edge_index[:,poison_edge_weights.nonzero().flatten().long()].to(self.device)
            poison_edge_weights = torch.ones([poison_edge_index.shape[1],]).to(self.device)
            poison_y = torch.LongTensor([self.args.target_class]).to(self.device)

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
    
    def meta_backdoor_Gprompt(self, data, criterion, eps, step, random):
        """
        This is the modified function for embedding a fine-tune-persisting backdoor.
        
        eps: the ball of finding the worst updates of model parameters
        step: how many steps we apply here
        random: whether to apply random updates instead of gradients
        idx_trojan: the indices of trojaned inputs
        """

        # Clone the encoder to prevent modifying the original model
        clone = trojan_utils.clone_module(self.prompt_generator)
        
        # Initialize the meta-loss to accumulate the total loss over multiple steps
        meta_loss = 0.0

        # Iteratively apply updates over the specified number of steps
        for _ in range(step):
            # Forward pass: get the output of the cloned model on the data
            # graph_embedding = self.pretrain_gnn(data.x, data.edge_index)
            graph_emb = self.pretrain_gnn(data.x, data.edge_index, data.batch, prompt = clone, prompt_type = 'Gprompt')

            center, class_counts = center_embedding(graph_emb, data.y, self.num_classes)
            # out = clone(graph_emb, center.detach(), data.edge_index)
            
            # Compute the backdoor loss for the trojaned inputs
            backdoor_loss = criterion(graph_emb, center.detach(), data.y)

            # Compute gradients to maximize the backdoor loss (gradient ascent)
            if random:
                # Apply random gradients if the 'random' flag is set to True
                grads = [2 * torch.rand_like(g) - 1 for g in clone.parameters()]
            else:
                # Otherwise, compute the actual gradients of the backdoor loss
                grads = torch.autograd.grad(backdoor_loss, clone.parameters(), retain_graph=True, allow_unused=True)

            # Filter out None gradients
            filtered_grads = [g for g in grads if g is not None]

            # Proceed if there are any valid gradients left
            if len(filtered_grads) > 0:
                # Compute the total gradient norm for valid gradients
                total_norm = torch.norm(torch.stack([torch.norm(g, p=2) for g in filtered_grads]), p=2)

                # Ensure that the norm is at least 1 to prevent division by a very small value
                if total_norm < 1.0:
                    total_norm = 1.0

                # Adjust the learning rate based on the total gradient norm
                lr = eps / (float(step) * total_norm)

                # Create updates: for parameters with None gradients, use zero updates
                updates = [
                    lr * g if g is not None else torch.zeros_like(p)
                    for g, p in zip(grads, clone.parameters())
                ]

                # Apply the gradient updates to the cloned model
                clone = trojan_utils.update_module(clone, updates)
            # # Compute the total norm of the gradients
            # print(grads)
            # total_norm = torch.norm(torch.stack([torch.norm(g, p=2) for g in grads]), p=2)

            # # Ensure that the norm is at least 1 to prevent division by a very small value
            # if total_norm < 1.0:
            #     total_norm = 1.0
            
            # # Adjust the learning rate based on the total gradient norm
            # lr = eps / (float(step) * total_norm)

            # # Apply the gradient updates to the cloned model using gradient ascent
            # updates = [lr * g for g in grads]
            # clone = trojan_utils.update_module(clone, updates)

            # Accumulate the meta-loss (i.e., the backdoor loss over all steps)
            meta_loss = meta_loss + backdoor_loss

        # Compute the average meta-loss across all steps
        meta_loss = meta_loss / (step + 1)
        return meta_loss