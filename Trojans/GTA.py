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

        # optimizer_PG = optim.Adam(self.prompt_generator.parameters(), lr=self.args.trojan_pg_lr, weight_decay=self.args.trojan_pg_weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=self.args.trigger_lr, weight_decay=self.args.trigger_weight_decay)

        tobe_poison_sup_dataset = self.support_dataset[idx_trojan]
        num_poison = int(len(self.query_dataset) * 0.5)
        idx_trojan_que = trojan_utils.obtain_attach_nodes(seed=10,node_idxs=np.array(range(len(self.support_dataset))),size=num_poison)
        tobe_poison_que_dataset = self.query_dataset[idx_trojan_que]

        loss_best = 1e8
        for i in range(self.args.trojan_outter_epochs):
            if((i) % printN == 0):
                print("{}/{} *tune trigger generator | frozen gnn | frozen prompt".format(i+1, self.args.trojan_outter_epochs))
            self.trojan.train() 
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

                for batch_id, data in enumerate(poisoned_sup_dataloader):
                    data = data.to(self.device)
                    graph_emb = self.pretrain_gnn(data.x, data.edge_index, data.batch, prompt = self.prompt_generator, prompt_type = 'Gprompt')
                    # dt_predict = self.downstream_tasker(graph_emb)
                    center, class_counts = center_embedding(graph_emb, data.y, self.num_classes)

                    optimizer_trigger.zero_grad()
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
                    optimizer_trigger.step()

                    loss_sup_batch += loss_sup
                if((i) % printN == 0):
                    print("{}/{} Loss PG: {:.4f}".format(j, self.args.trojan_inner_epochs,loss_sup_batch))
    
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

        # optimizer_PG = optim.Adam(self.prompt_generator.parameters(), lr=self.args.trojan_pg_lr, weight_decay=self.args.trojan_pg_weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=self.args.trigger_lr, weight_decay=self.args.trigger_weight_decay)

        tobe_poison_sup_dataset = self.support_dataset[idx_trojan]
        num_poison = int(len(self.query_dataset) * 0.5)
        idx_trojan_que = trojan_utils.obtain_attach_nodes(seed=10,node_idxs=np.array(range(len(self.support_dataset))),size=num_poison)
        tobe_poison_que_dataset = self.query_dataset[idx_trojan_que]

        loss_best = 1e8
        for i in range(self.args.trojan_outter_epochs):
            if((i) % printN == 0):
                print("{}/{} *tune trigger generator | frozen gnn | frozen prompt".format(i+1, self.args.trojan_outter_epochs))
            self.trojan.train() 
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

                for batch_id, data in enumerate(poisoned_sup_dataloader):
                    data = data.to(self.device)
                    graph_emb = self.pretrain_gnn(data.x, data.edge_index, data.batch, prompt = self.prompt_generator, prompt_type = 'Gprompt')
                    # dt_predict = self.downstream_tasker(graph_emb)
                    # center, class_counts = center_embedding(graph_emb, data.y, self.num_classes)

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
                    optimizer_trigger.step()

                    loss_sup_batch += loss_sup
                if((i) % printN == 0):
                    print("{}/{} Loss PG: {:.4f}".format(j, self.args.trojan_inner_epochs,loss_sup_batch))
    
        # mean_centers = accumulated_centers / accumulated_counts
        # return mean_centers
        return None

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

        idx_sup_full = torch.LongTensor(range(len(support_dataset)))
        # idx_que_full = torch.LongTensor(range(len(query_dataset)))
        self.idx_trojan = idx_trojan

        self.pretrain_gnn = gnn.to(self.device)
        # self.prompt_module = prompt_module.to(self.device)
        self.prompt_generator = PG.to(self.device)
        self.downstream_tasker = answering.to(self.device)
        self.trojan = GraphTrojanNet(self.device, self.args.hid_dim, self.args.trigger_size, layernum=2).to(self.device)

        # optimizer_PG = optim.Adam(self.prompt_generator.parameters(), lr=self.args.trojan_pg_lr, weight_decay=self.args.trojan_pg_weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=self.args.trigger_lr, weight_decay=self.args.trigger_weight_decay)
        optimizer_DT = optim.Adam(self.downstream_tasker.parameters(), lr=self.args.trojan_dt_lr,weight_decay=self.args.trojan_dt_weight_decay)

        tobe_poison_sup_dataset = self.support_dataset[idx_trojan]
        num_poison = int(len(self.query_dataset) * 0.2)
        idx_trojan_que = trojan_utils.obtain_attach_nodes(seed=10,node_idxs=np.array(range(len(self.support_dataset))),size=num_poison)
        tobe_poison_que_dataset = self.query_dataset[idx_trojan_que]

        loss_best = 1e8
        for i in range(self.args.trojan_outter_epochs):
            '''Alternative Training: Train PG and then Trigger Generator'''            
            print("{}/{} *tune trigger generator | frozen gnn | frozen prompt |frozen answering function...".format(i+1, self.args.trojan_outter_epochs))
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

                    optimizer_trigger.zero_grad()
                    # optimizer_DT.zero_grad()
                    loss_sup = F.cross_entropy(dt_predict,data.y)
                    
                    loss_sup.backward()
                    optimizer_trigger.step()
                    # optimizer_DT.step()

                    loss_sup_batch += loss_sup
                if((j) % printN == 0):
                    print("{}/{} Loss PG: {:.4f}".format(j, self.args.trojan_inner_epochs,loss_sup_batch))
    
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
        poisoned_data_list = deepcopy(dataset.to_data_list())
        poisoned_data_list.extend(trojan_dataset)
        return poisoned_data_list
    
    def transfer_to_trojan_test_dataset(self, dataset,idx_atk, prompt_type, device=None):
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
        