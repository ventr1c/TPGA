import torch
from torch_geometric.loader import DataLoader
from prompt_graph.utils import constraint,  center_embedding, Gprompt_tuning_loss
from prompt_graph.evaluation import GPPTEva, GNNNodeEva, GPFEva, MultiGpromptEva
from prompt_graph.evaluation import GPPTGraphEva
from prompt_graph.pretrain import PrePrompt, prompt_pretrain_sample
from .task import BaseTask
import time
import warnings
import numpy as np
from prompt_graph.data import load4node, induced_graphs, graph_split, split_induced_graphs, node_sample_and_save, split_khop_induced_graphs, split_khop_induced_graphs_specific,split_fixed_neighbor_khop_graphs
from prompt_graph.evaluation import GpromptEva, AllInOneEva
import pickle
import os
from prompt_graph.utils import process
from copy import deepcopy

import Trojans.trojan_utils as trojan_utils
from torch_geometric.data import Batch, Data

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch import nn, optim

warnings.filterwarnings("ignore")

class NodeTask(BaseTask):
      def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.task_type = 'NodeTask'
            if self.prompt_type == 'MultiGprompt':
                  self.load_multigprompt_data()
            else:
                  self.load_data()
                  self.answering =  torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim),
                                                torch.nn.Softmax(dim=1)).to(self.device) 
            
            self.create_few_data_folder()         
            self.initialize_gnn()
            self.initialize_prompt()
            self.initialize_optimizer()
            
            self.trojan_prompt = deepcopy(self.prompt)
            self.trojan_answering = deepcopy(self.answering)
            self.prompt_ori = deepcopy(self.prompt)
            self.answering_ori = deepcopy(self.answering)
      

      # def create_few_data_folder(self):
      #       # 创建文件夹并保存数据
      #       for k in range(1, 11):
      #             k_shot_folder = './Experiment/sample_data/Node/'+ self.dataset_name +'/' + str(k) +'_shot'
      #             os.makedirs(k_shot_folder, exist_ok=True)
                  
      #             for i in range(1, 6):
      #                   folder = os.path.join(k_shot_folder, str(i))
      #                   os.makedirs(folder, exist_ok=True)
      #                   node_sample_and_save(self.data, k, folder, self.output_dim)
      #                   print(str(k) + ' shot ' + str(i) + ' th is saved!!')
      
      def create_few_data_folder(self):
            # 创建文件夹并保存数据
            k = self.shot_num


            k_shot_folder = './Experiment/sample_data/Node/'+ self.dataset_name +'/' + str(k) +'_shot'
            os.makedirs(k_shot_folder, exist_ok=True)
            
            for i in range(1, 6):
                  folder = os.path.join(k_shot_folder, str(i))
                  os.makedirs(folder, exist_ok=True)
                  node_sample_and_save(self.data, k, folder, self.output_dim)
                  print(str(k) + ' shot ' + str(i) + ' th is saved!!')

      def load_multigprompt_data(self):
            adj, features, labels, idx_train, idx_val, idx_test = process.load_data(self.dataset_name)  
            self.input_dim = features.shape[1]
            features, _ = process.preprocess_features(features)
            self.sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj).to(self.device)
            self.labels = torch.FloatTensor(labels[np.newaxis])
            self.features = torch.FloatTensor(features[np.newaxis]).to(self.device)
            self.idx_train = torch.LongTensor(idx_train)
            # print("labels",labels)
            print("adj",self.sp_adj.shape)
            print("feature",features.shape)
            self.idx_val = torch.LongTensor(idx_val)
            self.idx_test = torch.LongTensor(idx_test)

      def load_induced_graph(self):
            K = 2
            self.data, self.dataset = load4node(self.dataset_name, shot_num = self.shot_num)
            # self.data.to('cpu')
            self.input_dim = self.dataset.num_features
            self.output_dim = self.dataset.num_classes
            file_path = './Experiment/induced_graph/' + self.dataset_name + '/induced_graph_{}_hop.pkl'.format(K)
            if os.path.exists(file_path):
                  with open(file_path, 'rb') as f:
                        graphs_list = pickle.load(f)
            else:
                  print('Begin split_induced_graphs.')
                  # split_induced_graphs(self.dataset_name, self.data, smallest_size=10, largest_size=30)
                  # split_fixed_neighbor_khop_graphs(self.dataset_name, self.data, K = K, num_neighbors=3)
                  split_khop_induced_graphs(self.dataset_name, self.data, K = K)
                  with open(file_path, 'rb') as f:
                        graphs_list = pickle.load(f)
            return graphs_list

      def load_induced_graph_random_split_specific(self, node_idxs, flag):
            if(flag not in ['train_trojan', 'val_trojan', 'test_trojan']):
                  raise NotImplementedError("not implemented other induced graphs")

            K = 2 # num_hop
            file_path = './Experiment/induced_graph/' + self.dataset_name + '/induced_graph_{}_hop_{}.pkl'.format(K,flag)
            if os.path.exists(file_path):
                  with open(file_path, 'rb') as f:
                        graphs_list = pickle.load(f)
            else:
                  print('Begin split_induced_graphs.')
                  split_khop_induced_graphs_specific(self.dataset_name, self.data.cpu(), node_idxs, flag = flag, K = K)
                  with open(file_path, 'rb') as f:
                        graphs_list = pickle.load(f)
            return graphs_list

      def load_data(self):
            self.data, self.dataset = load4node(self.dataset_name, shot_num = self.shot_num)
            self.data.to(self.device)
            self.data_ori = deepcopy(self.data)
            self.input_dim = self.dataset.num_features
            self.output_dim = self.dataset.num_classes
            self.num_classes = self.dataset.num_classes
      
      def train(self, data, train_idx):
            self.gnn.train()
            self.optimizer.zero_grad() 
            out = self.gnn(data.x, data.edge_index, batch=None) 
            out = self.answering(out)
            loss = self.criterion(out[train_idx], data.y[train_idx])
            loss.backward()  
            self.optimizer.step()  
            return loss.item()
      
      def GPPTtrain(self, data, train_idx):
            self.prompt.train()
            node_embedding = self.gnn(data.x, data.edge_index)
            out = self.prompt(node_embedding, data.edge_index)
            loss = self.criterion(out[train_idx], data.y[train_idx])
            loss = loss + 0.001 * constraint(self.device, self.prompt.get_TaskToken())
            self.pg_opi.zero_grad()
            loss.backward()
            self.pg_opi.step()
            self.prompt.update_StructureToken_weight(self.prompt.get_mid_h())
            return loss.item()
      
      def MultiGpromptTrain(self, pretrain_embs, train_lbls, train_idx):
            self.DownPrompt.train()
            self.optimizer.zero_grad()
            prompt_feature = self.feature_prompt(self.features)
            # prompt_feature = self.feature_prompt(self.data.x)
            # embeds1 = self.gnn(prompt_feature, self.data.edge_index)
            embeds1= self.Preprompt.gcn(prompt_feature, self.sp_adj , True, False)
            pretrain_embs1 = embeds1[0, train_idx]
            logits = self.DownPrompt(pretrain_embs,pretrain_embs1, train_lbls,1).float().to(self.device)
            loss = self.criterion(logits, train_lbls)           
            loss.backward(retain_graph=True)
            self.optimizer.step()
            return loss.item()
      
      def SUPTtrain(self, data):
            self.gnn.train()
            self.optimizer.zero_grad() 
            data.x = self.prompt.add(data.x)
            out = self.gnn(data.x, data.edge_index, batch=None) 
            out = self.answering(out)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask])  
            orth_loss = self.prompt.orthogonal_loss()
            loss += orth_loss
            loss.backward()  
            self.optimizer.step()  
            return loss
      
      def GPFTrain(self, train_loader):
            self.prompt.train()
            self.answering.train()
            total_loss = 0.0 
            for batch in train_loader:  
                  # self.optimizer.zero_grad()
                  self.pg_opi.zero_grad()
                  self.answer_opi.zero_grad() 
                  batch = batch.to(self.device)
                  batch.x = self.prompt.add(batch.x)
                  out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = self.prompt_type)
                  out = self.answering(out)
                  loss = self.criterion(out, batch.y)  
                  loss.backward()  
                  # self.optimizer.step()  
                  self.pg_opi.step()
                  self.answer_opi.step()
                  total_loss += loss.item()  
            return total_loss / len(train_loader) 

      def AllInOneTrain(self, train_loader,answer_epoch = 1, prompt_epoch = 1):
            #we update answering and prompt alternately.
            
            answer_epoch = 1  # 50
            prompt_epoch = 1  # 50
            # tune task head
            self.answering.train()
            self.prompt.eval()
            self.gnn.eval()
            for epoch in range(1, answer_epoch + 1):
                  answer_loss = self.prompt.Tune(train_loader, self.gnn,  self.answering, self.criterion, self.answer_opi, self.device)
                  print(("frozen gnn | frozen prompt | *tune answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, answer_loss)))

            # tune prompt
            self.answering.eval()
            self.prompt.train()
            for epoch in range(1, prompt_epoch + 1):
                  pg_loss = self.prompt.Tune( train_loader,  self.gnn, self.answering, self.criterion, self.pg_opi, self.device)
                  print(("frozen gnn | *tune prompt |frozen answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, pg_loss)))
            
            return pg_loss
      
      def AllInOneMetaTrain(self, train_loader,answer_epoch = 1, prompt_epoch = 1):
            #we update answering and prompt alternately.
            
              # 50
              # 50
            # tune task head
            self.answering.train()
            self.prompt.eval()
            for epoch in range(1, answer_epoch + 1):
                  answer_loss = self.prompt.Tune(train_loader, self.gnn,  self.answering, self.criterion, self.answer_opi, self.device)
                  print(("frozen gnn | frozen prompt | *tune answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, answer_loss)))

            # tune prompt
            self.answering.eval()
            self.prompt.train()
            for epoch in range(1, prompt_epoch + 1):
                  pg_loss = self.prompt.Tune( train_loader,  self.gnn, self.answering, self.criterion, self.pg_opi, self.device)
                  print(("frozen gnn | *tune prompt |frozen answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, pg_loss)))
            
            return pg_loss
      
      def GpromptTrain(self, train_loader):
            self.prompt.train()
            total_loss = 0.0 
            accumulated_centers = None
            accumulated_counts = None
            for batch in train_loader:  
                  self.pg_opi.zero_grad() 
                  batch = batch.to(self.device)
                  out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = 'Gprompt')
                  # out = s𝑡,𝑥 = ReadOut({p𝑡 ⊙ h𝑣 : 𝑣 ∈ 𝑉 (𝑆𝑥)}),
                  center, class_counts = center_embedding(out, batch.y, self.output_dim)
                   # 累积中心向量和样本数
                  if accumulated_centers is None:
                        accumulated_centers = center
                        accumulated_counts = class_counts
                  else:
                        accumulated_centers += center * class_counts
                        accumulated_counts += class_counts
                  criterion = Gprompt_tuning_loss()
                  loss = criterion(out, center, batch.y)  
                  loss.backward()  
                  self.pg_opi.step()  
                  total_loss += loss.item()
            # 计算加权平均中心向量
            mean_centers = accumulated_centers / accumulated_counts

            return total_loss / len(train_loader), mean_centers
      
      def TrojanGpromptFinetune(self, train_loader, trojan_prompt, trojan_pg_opi, trojan_accumulated_centers, trojan_accumulated_counts):
            trojan_prompt.train()
            total_loss = 0.0 
            accumulated_centers = trojan_accumulated_centers# trojan_accumulated_centers.clone()
            accumulated_counts = trojan_accumulated_counts # deepcopy(trojan_accumulated_counts)
            for batch in train_loader:  
                  trojan_pg_opi.zero_grad() 
                  batch = batch.to(self.device)
                  out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = trojan_prompt, prompt_type = 'Gprompt')
                  # out = s𝑡,𝑥 = ReadOut({p𝑡 ⊙ h𝑣 : 𝑣 ∈ 𝑉 (𝑆𝑥)}),
                  center, class_counts = center_embedding(out, batch.y, self.output_dim)
                   # 累积中心向量和样本数
                  if accumulated_centers is None:
                        accumulated_centers = center
                        accumulated_counts = class_counts
                  else:
                        accumulated_centers += center * class_counts
                        accumulated_counts += class_counts
                  criterion = Gprompt_tuning_loss()
                  loss = criterion(out, center, batch.y)  
                  loss.backward()  
                  trojan_pg_opi.step()  
                  total_loss += loss.item()
            # 计d算加权平均中心向量
            mean_centers = accumulated_centers / accumulated_counts

            return total_loss / len(train_loader), mean_centers, trojan_prompt

      def TrojanGPPTFinetune(self, data, train_idx, trojan_prompt, trojan_pg_opi):
            trojan_prompt.train()
            node_embedding = self.gnn(data.x, data.edge_index)
            out = trojan_prompt(node_embedding, data.edge_index)
            loss = self.criterion(out[train_idx], data.y[train_idx])
            loss = loss + 0.001 * constraint(self.device, trojan_prompt.get_TaskToken())
            trojan_pg_opi.zero_grad()
            loss.backward()
            trojan_pg_opi.step()
            # trojan_prompt.update_StructureToken_weight(trojan_prompt.get_mid_h())
            return loss.item(), trojan_prompt

      def prompt_graph_poisoning(self):
            # trojan_prompt = self.trojan_prompt
            # trojan_answering = self.trojan_answering
            trojan_prompt = deepcopy(self.prompt)
            trojan_answering = deepcopy(self.answering)
            if(self.prompt_type == 'GPPT'):
                  pass
                  print("Number of training graphs: {}".format(len(self.idx_train)))
                  num_poison = int(len(self.idx_train) * self.args.poison_ratio)
                  print("Number of poisoning graphs: {}".format(num_poison))
                  idx_trojan = trojan_utils.obtain_attach_nodes(seed=10,node_idxs=self.idx_unlabeled.cpu().numpy(),size=num_poison)
                  idx_trojan = torch.tensor(idx_trojan).to(self.device)
                  print("Selected trojan graphs in train dataset: {}".format(idx_trojan))
            else:
                  train_graphs_batch = Batch.from_data_list(self.train_graphs)

                  num_poison = int(len(train_graphs_batch) * self.args.poison_ratio)
                  print("Number of poisoning graphs: {}".format(num_poison))
            
                  idx_trojan = trojan_utils.obtain_attach_nodes(seed=10,node_idxs=np.array(range(len(train_graphs_batch))),size=num_poison)
                  print("Selected trojan graphs in train dataset: {}".format(idx_trojan))
            
            if(self.args.compare_method == 'Random'):
                  from Trojans.random_prompt_backdoor import RandomBackdoor
                  if(self.prompt_type=='Gprompt'):
                        self.trigger_generator = RandomBackdoor(self.args, self.num_classes, self.device)
                        # self.trojan_center = self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan, self.trojan_center)
                        # self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                        if(self.args.if_freeze_dt_classifier):
                              self.trojan_center = self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan, self.trojan_center)
                              self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                        else:
                              self.trojan_center, self.trojan_accumulated_centers, self.trojan_accumulated_counts = self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan, self.trojan_center)
                              self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                  elif(self.prompt_type == 'GPPT'):
                        from Trojans.random_prompt_backdoor_node import Backdoor
                        self.trigger_generator = Backdoor(self.args,self.num_classes,self.device)
                        if(self.args.if_freeze_dt_classifier):
                              self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, self.data, idx_train=self.idx_train, idx_trojan=idx_trojan, idx_unlabeled=self.idx_unlabeled, idx_test = self.idx_test)
                              self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                        else:
                              self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, self.data, idx_train=self.idx_train, idx_trojan=idx_trojan, idx_unlabeled=self.idx_unlabeled, idx_test = self.idx_test)
                              self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
            elif(self.args.compare_method == 'GTA'):
                  from Trojans.GTA import Backdoor
                  self.trigger_generator = Backdoor(self.args,self.num_classes,self.device)
                  # self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan)
                  if(self.prompt_type=='Gprompt'):
                        if(self.args.if_freeze_dt_classifier):
                              self.trojan_center = self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan, self.trojan_center)
                              self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                        else:
                              self.trojan_center, self.trojan_accumulated_centers, self.trojan_accumulated_counts = self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan, self.trojan_center)
                              self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                  elif(self.prompt_type == 'GPPT'):
                        from Trojans.GTA_node import Backdoor
                        self.trigger_generator = Backdoor(self.args,self.num_classes,self.device)
                        if(self.args.if_freeze_dt_classifier):
                              self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, self.data, idx_train=self.idx_train, idx_trojan=idx_trojan, idx_unlabeled=self.idx_unlabeled, idx_test = self.idx_test)
                              self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                        else:
                              self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, self.data, idx_train=self.idx_train, idx_trojan=idx_trojan, idx_unlabeled=self.idx_unlabeled, idx_test = self.idx_test)
                              self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                  else:      
                        self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan)
                        self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                        self.trojan_answering = self.trigger_generator.downstream_tasker
            elif(self.args.compare_method == 'UGBA'):
                  from Trojans.UGBA import Backdoor
                  self.trigger_generator = Backdoor(self.args,self.num_classes,self.device)
                  # self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan)
                  if(self.prompt_type=='Gprompt'):
                        # self.trojan_center = self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan, self.trojan_center)
                        # self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                        if(self.args.if_freeze_dt_classifier):
                              self.trojan_center = self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan, self.trojan_center)
                              self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                        else:
                              self.trojan_center, self.trojan_accumulated_centers, self.trojan_accumulated_counts = self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan, self.trojan_center)
                              self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                  elif(self.prompt_type == 'GPPT'):
                        from Trojans.UGBA_node import Backdoor
                        self.trigger_generator = Backdoor(self.args,self.num_classes,self.device)
                        if(self.args.if_freeze_dt_classifier):
                              self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, self.data, idx_train=self.idx_train, idx_trojan=idx_trojan, idx_unlabeled=self.idx_unlabeled, idx_test = self.idx_test)
                              self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                        else:
                              self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, self.data, idx_train=self.idx_train, idx_trojan=idx_trojan, idx_unlabeled=self.idx_unlabeled, idx_test = self.idx_test)
                              self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                  else:      
                        self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan)
                        self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                        self.trojan_answering = self.trigger_generator.downstream_tasker
            elif(self.args.compare_method == 'SBA-Samp'):
                  from Trojans.SBA import RandomBackdoor as Backdoor
                  self.trigger_generator = Backdoor(self.args,self.num_classes,self.device)
                  # self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan)
                  if(self.prompt_type=='Gprompt'):
                        if(self.args.if_freeze_dt_classifier):
                              self.trojan_center = self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan, self.trojan_center)
                              self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                        else:
                              self.trojan_center, self.trojan_accumulated_centers, self.trojan_accumulated_counts = self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan, self.trojan_center)
                              self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                  elif(self.prompt_type == 'GPPT'):
                        from Trojans.SBA_node import Backdoor
                        self.trigger_generator = Backdoor(self.args,self.num_classes,self.device)
                        if(self.args.if_freeze_dt_classifier):
                              self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, self.data, idx_train=self.idx_train, idx_trojan=idx_trojan, idx_unlabeled=self.idx_unlabeled, idx_test = self.idx_test)
                              self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                        else:
                              self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, self.data, idx_train=self.idx_train, idx_trojan=idx_trojan, idx_unlabeled=self.idx_unlabeled, idx_test = self.idx_test)
                              self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                  else:      
                        self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan)
                        self.trojan_prompt = self.trigger_generator.prompt_generator
                        self.trojan_answering = self.trigger_generator.downstream_tasker
      
            elif(self.args.compare_method == 'Ours'):
                  # self.trojan_answering = self.trigger_generator.answering # note that we do not tune the answering, just for clearify
                  if(self.prompt_type=='Gprompt'):
                        from Trojans.backdoor import Backdoor
                        self.trigger_generator = Backdoor(self.args,self.num_classes,self.device)
                        if(self.args.if_freeze_dt_classifier):
                              self.trojan_center = self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan, self.trojan_center)
                              self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                        else:
                              self.trojan_center, self.trojan_accumulated_centers, self.trojan_accumulated_counts = self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan, self.trojan_center)
                              self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                  elif(self.prompt_type == 'GPPT'):
                        from Trojans.backdoor_node import Backdoor
                        self.trigger_generator = Backdoor(self.args,self.num_classes,self.device)
                        if(self.args.if_freeze_dt_classifier):
                              self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, self.data, idx_train=self.idx_train, idx_trojan=idx_trojan, idx_unlabeled=self.idx_unlabeled, idx_test = self.idx_test)
                              self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                        else:
                              self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, self.data, idx_train=self.idx_train, idx_trojan=idx_trojan, idx_unlabeled=self.idx_unlabeled, idx_test = self.idx_test)
                              self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                  else:      
                        from Trojans.backdoor import Backdoor
                        self.trigger_generator = Backdoor(self.args,self.num_classes,self.device)
                        self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan)
                        self.trojan_prompt = deepcopy(self.trigger_generator.prompt_generator)
                        self.trojan_answering = self.trigger_generator.downstream_tasker
            else:
                  NotImplementedError("No implemented backdoor attack methods.")
            # return trigger_generator, trojan_PG, trojan_answering

      def prompt_graph_poisoning_freeze_DT(self):
            # trojan_prompt = self.trojan_prompt
            # trojan_answering = self.trojan_answering
            trojan_prompt = deepcopy(self.prompt)
            trojan_answering = deepcopy(self.answering)
            if(self.prompt_type == 'GPPT'):
                  pass
                  num_poison = int(len(self.idx_train) * self.args.poison_ratio)
                  print("Number of poisoning graphs: {}".format(num_poison))
                  idx_trojan = trojan_utils.obtain_attach_nodes(seed=10,node_idxs=self.idx_unlabeled.cpu().numpy(),size=num_poison)
                  idx_trojan = torch.tensor(idx_trojan).to(self.device)
                  print("Selected trojan graphs in train dataset: {}".format(idx_trojan))
            else:
                  train_graphs_batch = Batch.from_data_list(self.train_graphs)

                  num_poison = int(len(train_graphs_batch) * self.args.poison_ratio)
                  print("Number of poisoning graphs: {}".format(num_poison))
            
                  idx_trojan = trojan_utils.obtain_attach_nodes(seed=10,node_idxs=np.array(range(len(train_graphs_batch))),size=num_poison)
                  print("Selected trojan graphs in train dataset: {}".format(idx_trojan))
            
            if(self.args.compare_method == 'Random'):
                  from Trojans.random_prompt_backdoor import RandomBackdoor
                  if(self.prompt_type=='Gprompt'):
                        self.trigger_generator = RandomBackdoor(self.args, self.num_classes, self.device)
                        self.trojan_center = self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan, self.trojan_center)
                        self.trojan_prompt = self.trigger_generator.prompt_generator
                        self.trojan_center = self.center_ori.clone()
            elif(self.args.compare_method == 'GTA'):
                  from Trojans.GTA import Backdoor
                  self.trigger_generator = Backdoor(self.args,self.num_classes,self.device)
                  # self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan)
                  if(self.prompt_type=='Gprompt'):
                        self.trojan_center = self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan, self.trojan_center)
                        self.trojan_prompt = self.trigger_generator.prompt_generator
                        self.trojan_center = self.center_ori.clone()
                  else:      
                        self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan)
                        self.trojan_prompt = self.trigger_generator.prompt_generator
                        # self.trojan_answering = self.trigger_generator.downstream_tasker
                        self.trojan_answering = deepcopy(self.answering_ori)
            elif(self.args.compare_method == 'UGBA'):
                  from Trojans.UGBA import Backdoor
                  self.trigger_generator = Backdoor(self.args,self.num_classes,self.device)
                  # self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan)
                  if(self.prompt_type=='Gprompt'):
                        self.trojan_center = self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan, self.trojan_center)
                        self.trojan_prompt = self.trigger_generator.prompt_generator
                        self.trojan_center = self.center_ori.clone()
                  else:      
                        self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan)
                        self.trojan_prompt = self.trigger_generator.prompt_generator
                        # self.trojan_answering = self.trigger_generator.downstream_tasker
                        self.trojan_answering = deepcopy(self.answering_ori)
            elif(self.args.compare_method == 'Ours'):
                  # self.trojan_answering = self.trigger_generator.answering # note that we do not tune the answering, just for clearify
                  if(self.prompt_type=='Gprompt'):
                        from Trojans.backdoor import Backdoor
                        self.trigger_generator = Backdoor(self.args,self.num_classes,self.device)
                        self.trojan_center = self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan, self.trojan_center)
                        self.trojan_prompt = self.trigger_generator.prompt_generator
                        self.trojan_center = self.center_ori.clone()
                  elif(self.prompt_type == 'GPPT'):
                        from Trojans.backdoor_node import Backdoor
                        self.trigger_generator = Backdoor(self.args,self.num_classes,self.device)
                        self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, self.data, idx_train=self.idx_train, idx_trojan=idx_trojan, idx_unlabeled=self.idx_unlabeled)
                        self.trojan_prompt = self.trigger_generator.prompt_generator
                  else:      
                        from Trojans.backdoor import Backdoor
                        self.trigger_generator = Backdoor(self.args,self.num_classes,self.device)
                        self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan)
                        self.trojan_prompt = self.trigger_generator.prompt_generator
                        # self.trojan_answering = self.trigger_generator.downstream_tasker
                        self.trojan_answering = deepcopy(self.answering_ori)
            else:
                  NotImplementedError("No implemented backdoor attack methods.")
            # return trigger_generator, trojan_PG, trojan_answering

      def trojan_ASR_test_evaluation(self):
            if self.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus']:
                  split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(self.test_graphs), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
                  idx_clean_test, idx_atk = split_trojan_evaluation['clean_test'], split_trojan_evaluation['attack']

                  test_graphs_batch = Batch.from_data_list(self.test_graphs)
                  test_graphs_batch = test_graphs_batch.cpu()

                  clean_test_data_list = test_graphs_batch[idx_clean_test]
                  attack_test_data_list = self.trigger_generator.transfer_to_trojan_test_dataset(test_graphs_batch,idx_atk, self.prompt_type, self.device)

                  clean_test_loader = DataLoader(clean_test_data_list, batch_size=self.args.trojan_batch_size, shuffle=False)
                  attack_test_loader = DataLoader(attack_test_data_list, batch_size=self.args.trojan_batch_size, shuffle=False)
                              
                  if self.prompt_type == 'All-in-one':
                        test_acc, F1 = AllInOneEva(clean_test_loader, self.trojan_prompt, self.gnn, self.trojan_answering, self.output_dim, self.device)           
                        bkd_asr, F1 = AllInOneEva(attack_test_loader, self.trojan_prompt, self.gnn, self.trojan_answering, self.output_dim, self.device)                                           
                  elif self.prompt_type in ['GPF', 'GPF-plus']:
                        test_acc = GPFEva(clean_test_loader, self.gnn, self.trojan_prompt, self.trojan_answering, self.device)                    
                        bkd_asr = GPFEva(attack_test_loader, self.gnn, self.trojan_prompt, self.trojan_answering, self.device)                                                         
                  elif self.prompt_type =='Gprompt':
                        test_acc = GpromptEva(clean_test_loader, self.gnn, self.trojan_prompt, self.trojan_center, self.device)
                        bkd_asr = GpromptEva(attack_test_loader, self.gnn, self.trojan_prompt, self.trojan_center, self.device)

                  print("===========After Attack=====================")
                  print("clean test accuracy {:.4f} ".format(test_acc))                        
                  print("attack success rate {:.4f} ".format(bkd_asr))        
                  return bkd_asr              
            elif self.prompt_type in ['GPPT']:
                  split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(self.idx_test), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
                  idx_clean_test = self.idx_test[split_trojan_evaluation['clean_test']]
                  idx_atk = self.idx_test[split_trojan_evaluation['attack']]

                  
                  poison_data = self.trigger_generator.get_poisoned(idx_atk)
                  clean_data = deepcopy(self.data_ori)
                  # clean_data = self.trigger_generator.get_clean()
                  print("clean",clean_data.y)
                  if(self.prompt_type == 'GPPT'):
                        bkd_asr = GPPTEva(poison_data, idx_atk, self.gnn, self.trojan_prompt)    
                        # print(bkd_asr)
                        test_acc = GPPTEva(clean_data, idx_clean_test, self.gnn, self.trojan_prompt)  
                        # print(test_acc)
                        # test_acc = self.evaluate_ASR_1by1(clean_data, idx_clean_test)  
                        # print(test_acc)
                        # test_acc = self.evaluate_ASR_overall(clean_data, idx_clean_test)    
                        # print(test_acc)  
                  print("===========After Attack=====================")
                  print("clean test accuracy {:.4f} ".format(test_acc))                        
                  print("attack success rate {:.4f} ".format(bkd_asr))        
            else:
                  NotImplementedError("No implemented trojan evaluation for other prompts")
      def evaluate_ASR_1by1(self, poison_data, idx_atk):
            from torch_geometric.utils  import k_hop_subgraph
            asr = 0

            attack_data_list = []
            for i,idx in enumerate(idx_atk):
                  idx=int(idx)
                  sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask  = k_hop_subgraph(node_idx = [idx], num_hops = 2, edge_index = poison_data.edge_index, relabel_nodes=True) # sub_mapping means the index of [idx] in sub)nodeset
                  ori_node_idx = sub_induct_nodeset[sub_mapping]
                  relabeled_node_idx = sub_mapping
                  sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(self.device)
                  with torch.no_grad():
                        # inject trigger on attack test nodes (idx_atk)'''
                        if(self.args.compare_method in ['Random','SBA-Samp']):
                              induct_x, induct_edge_index,induct_edge_weights = self.trigger_generator.inject_trigger_rand(relabeled_node_idx,poison_data.x[sub_induct_nodeset],sub_induct_edge_index)
                        else:
                              induct_x, induct_edge_index,induct_edge_weights = self.trigger_generator.inject_trigger(relabeled_node_idx,poison_data.x[sub_induct_nodeset],sub_induct_edge_index,sub_induct_edge_weights,self.device)
                        induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
                        
                        # induct_data = Data(x = induct_x, edge_index=induct_edge_index, y = self.args.target_class)
                        induct_data = Data(x = induct_x, edge_index=induct_edge_index, y = poison_data.y[idx])

                        attack_data_list.append(induct_data)
            
            attack_dataloader = DataLoader(attack_data_list, batch_size=256, shuffle=False)
            if(self.prompt_type == 'GPPT'):
                  bkd_asr = GPPTGraphEva(attack_dataloader, self.gnn, self.prompt, self.device)
            else:
                  raise NotImplementedError("No implemented ASR evaluation for other prompts")
            return bkd_asr

      def evaluate_ASR_overall(self, poison_data, idx_atk):
            from torch_geometric.utils  import k_hop_subgraph
            asr = 0
            
            with torch.no_grad():
                  # inject trigger on attack test nodes (idx_atk)'''
                  poison_edge_weights = torch.ones([poison_data.edge_index.shape[1]]).to(self.device)
                  if(self.args.compare_method in ['Random','SBA-Samp']):
                        update_poison_x, update_poison_edge_index,update_poison_edge_weights = self.trigger_generator.inject_trigger_rand(idx_atk,poison_data.x,poison_data.edge_index)
                  else:
                        update_poison_x, update_poison_edge_index,update_poison_edge_weights = self.trigger_generator.inject_trigger(idx_atk,poison_data.x,poison_data.edge_index,poison_edge_weights,self.device)
                  update_poison_x, update_poison_edge_index,update_poison_edge_weights = update_poison_x.clone().detach(), update_poison_edge_index.clone().detach(),update_poison_edge_weights.clone().detach()
                  update_poison_y = poison_data.y.clone()
                  # update_poison_y[idx_atk] = self.args.target_class 
                  update_poison_data = Data(x = update_poison_x, edge_index=update_poison_edge_index, y = update_poison_y)


            if(self.prompt_type == 'GPPT'):
                  # bkd_asr = GPPTGraphEva(attack_dataloader, self.gnn, self.prompt, self.device)
                  bkd_asr = GPPTEva(update_poison_data, idx_atk, self.gnn, self.prompt)  
            else:
                  raise NotImplementedError("No implemented ASR evaluation for other prompts")
            return bkd_asr

      def trojan_finetune_dt_evaluation(self):
            self.trojan_finetune_downstream_classifier()
            self.trojan_ACC_test_evaluation_finetune()
            self.trojan_ASR_test_evaluation_finetune()

      def trojan_finetune_downstream_classifier(self):

            if(self.args.training_split == 'k-shot'):
                  i = 5 # select another set rather than normal training set 
                  idx_train = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/train_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).to(self.device)
                  train_lbls = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/train_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().to(self.device)

                  idx_test = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/test_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).to(self.device)
                  test_lbls = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/test_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().to(self.device)
            
            elif(self.args.training_split == 'classical'):
                  seed = self.args.seed + 6
                  split = trojan_utils.get_split_self(num_samples=self.data.x.shape[0], train_ratio=0.7, test_ratio=0.2, seed=seed)
                  idx_train = split['train']
                  idx_valid = split['valid']
                  idx_test = split['test']

            '''
            finetune downstream classifier
            '''
            # for all-in-one and Gprompt we use k-hop subgraph
            if self.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus', 'All-in-one-meta']:
                  if(self.dataset_name in ['Flickr', 'ogbn-arxiv','Physics']):
                        train_graphs = self.load_induced_graph_random_split_specific(idx_train, 'train_trojan')
                        test_graphs = self.load_induced_graph_random_split_specific(idx_test, 'test_trojan')
                  else:
                        graphs_list = self.load_induced_graph()

                        train_graphs = []
                        test_graphs = []
                        
                        for graph in graphs_list:                              
                              if graph.index in idx_train:
                                    train_graphs.append(graph)
                              elif graph.index in idx_test:
                                    test_graphs.append(graph)

                  train_loader = DataLoader(train_graphs, batch_size=self.args.dt_batch_size, shuffle=True)
                  test_loader = DataLoader(test_graphs, batch_size=self.args.dt_batch_size, shuffle=False)
                  print("prepare induce graph data is finished!")
                  self.test_graphs = test_graphs
                  self.train_graphs = train_graphs
                  

            elif self.prompt_type == 'MultiGprompt':
                  embeds, _ = self.Preprompt.embed(self.features, self.sp_adj, True, None, False)
                  pretrain_embs = embeds[0, idx_train]
                  test_embs = embeds[0, idx_test]

            elif self.prompt_type == 'GPPT':
                  self.idx_train = idx_train.to(self.device)
                  self.idx_test = idx_test.to(self.device)

                  idx_full = torch.tensor(range(self.data.x.shape[0])).to(self.device)

                  idx_train_test = torch.concat((self.idx_train,self.idx_test))
                  train_test_mask = torch.isin(idx_full, idx_train_test)
                  idx_unlabeled = idx_full[~train_test_mask]

                  self.idx_unlabeled = idx_unlabeled
            
            '''
            initialize trojan-finetune optimizer 
            '''
            if self.prompt_type in ['Gprompt', 'GPPT']:
                  trojan_pg_opi = optim.Adam(self.trojan_prompt.parameters(), lr=self.args.prompt_lr, weight_decay=self.args.prompt_weight_decay)
            else:
                  raise NotImplementedError("Unsupported prompt type")

            for epoch in range(1, self.args.epoch_trojan_finetune+1):
                  t0 = time.time()                                                
                  if self.prompt_type =='Gprompt':
                        self.trojan_prompt_finetune = deepcopy(self.trojan_prompt)
                        self.trojan_center_finetune = self.trojan_center.clone()
                        loss, center, self.trojan_prompt_finetune = self.TrojanGpromptFinetune(train_loader, self.trojan_prompt_finetune, trojan_pg_opi, self.trojan_accumulated_centers, self.trojan_accumulated_counts)
                        self.trojan_center_finetune = center.clone()
                  elif(self.prompt_type == 'GPPT'):
                        self.trojan_prompt_finetune = deepcopy(self.trojan_prompt)
                        loss, self.trojan_prompt_finetune = self.TrojanGPPTFinetune(self.data, idx_train, self.trojan_prompt_finetune, trojan_pg_opi)    
                  else:
                        raise NotImplementedError("Unsupported prompt type")
                  print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f}  ".format(epoch, time.time() - t0, loss))

                  
      def trojan_ACC_test_evaluation_finetune(self):
            if self.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus']:
                  split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(self.test_graphs), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
                  idx_clean_test, idx_atk = split_trojan_evaluation['clean_test'], split_trojan_evaluation['attack']

                  test_graphs_batch = Batch.from_data_list(self.test_graphs)

                  clean_test_data_list = test_graphs_batch[idx_clean_test]
                  clean_test_loader = DataLoader(clean_test_data_list, batch_size=10, shuffle=False)
                              
                  if self.prompt_type == 'All-in-one':
                        test_acc, F1 = AllInOneEva(clean_test_loader, self.trojan_prompt, self.gnn, self.trojan_answering_finetune, self.output_dim, self.device)           
                  elif self.prompt_type in ['GPF', 'GPF-plus']:
                        test_acc = GPFEva(clean_test_loader, self.gnn, self.trojan_prompt, self.trojan_answering_finetune, self.device)                    
                  elif self.prompt_type =='Gprompt':
                        test_acc = GpromptEva(clean_test_loader, self.gnn, self.trojan_prompt_finetune, self.trojan_center, self.device)

                  print("===========Before Attack (Finetune)=====================")
                  print("clean test accuracy {:.4f} ".format(test_acc))         
            elif self.prompt_type in ['GPPT']:
                  split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(self.idx_test), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
                  idx_clean_test = self.idx_test[split_trojan_evaluation['clean_test']]
                  idx_atk = self.idx_test[split_trojan_evaluation['attack']]

                  if self.prompt_type == 'GPPT':
                        test_acc = GPPTEva(self.data, idx_clean_test, self.gnn, self.trojan_prompt_finetune)    
                  
                  print("===========Before Attack (Finetune)=====================")
                  print("clean test accuracy {:.4f} ".format(test_acc))   
            else:
                  NotImplementedError("No implemented trojan evaluation for other prompts")
            
      def trojan_ASR_test_evaluation_finetune(self):
            if self.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus']:
                  split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(self.test_graphs), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
                  idx_clean_test, idx_atk = split_trojan_evaluation['clean_test'], split_trojan_evaluation['attack']

                  test_graphs_batch = Batch.from_data_list(self.test_graphs)
                  test_graphs_batch = test_graphs_batch.cpu()

                  clean_test_data_list = test_graphs_batch[idx_clean_test]
                  attack_test_data_list = self.trigger_generator.transfer_to_trojan_test_dataset(test_graphs_batch,idx_atk, self.prompt_type, self.device)

                  clean_test_loader = DataLoader(clean_test_data_list, batch_size=self.args.trojan_batch_size, shuffle=False)
                  attack_test_loader = DataLoader(attack_test_data_list, batch_size=self.args.trojan_batch_size, shuffle=False)
                              
                  if self.prompt_type == 'All-in-one':
                        test_acc, F1 = AllInOneEva(clean_test_loader, self.trojan_prompt, self.gnn, self.trojan_answering_finetune, self.output_dim, self.device)           
                        bkd_asr, F1 = AllInOneEva(attack_test_loader, self.trojan_prompt, self.gnn, self.trojan_answering_finetune, self.output_dim, self.device)                                           
                  elif self.prompt_type in ['GPF', 'GPF-plus']:
                        test_acc = GPFEva(clean_test_loader, self.gnn, self.trojan_prompt, self.trojan_answering_finetune, self.device)                    
                        bkd_asr = GPFEva(attack_test_loader, self.gnn, self.trojan_prompt, self.trojan_answering_finetune, self.device)                                                         
                  elif self.prompt_type =='Gprompt':
                        test_acc = GpromptEva(clean_test_loader, self.gnn, self.trojan_prompt_finetune, self.trojan_center, self.device)
                        bkd_asr = GpromptEva(attack_test_loader, self.gnn, self.trojan_prompt_finetune, self.trojan_center, self.device)

                  print("===========After Attack (Finetune)=====================")
                  print("clean test accuracy {:.4f} ".format(test_acc))                        
                  print("attack success rate {:.4f} ".format(bkd_asr))        
                  return bkd_asr              
            elif self.prompt_type in ['GPPT']:
                  split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(self.idx_test), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
                  idx_clean_test = self.idx_test[split_trojan_evaluation['clean_test']]
                  idx_atk = self.idx_test[split_trojan_evaluation['attack']]
      
                  # poison_data = self.trigger_generator.get_poisoned(idx_atk)
                  # clean_data = self.trigger_generator.get_clean()
                  poison_data = self.trigger_generator.get_poisoned(idx_atk)
                  clean_data = deepcopy(self.data_ori)
                  if(self.prompt_type == 'GPPT'):
                        test_acc = GPPTEva(poison_data, idx_atk, self.gnn, self.trojan_prompt)    
                        bkd_asr = GPPTEva(clean_data, idx_clean_test, self.gnn, self.trojan_prompt)    

                  if(self.prompt_type == 'GPPT'):
                        test_acc = GPPTEva(poison_data, idx_clean_test, self.gnn, self.trojan_prompt_finetune)    
                        # bkd_asr = self.evaluate_ASR_1by1(poison_data, idx_atk)  
                        bkd_asr = self.evaluate_ASR_overall(poison_data, idx_atk)      
                  print("===========After Attack (Finetune)=====================")
                  print("clean test accuracy {:.4f} ".format(test_acc))                        
                  print("attack success rate {:.4f} ".format(bkd_asr))        
            else:
                  NotImplementedError("No implemented trojan evaluation for other prompts")

      def trojan_freeze_dt_evaluation(self):
            self.trojan_ACC_test_evaluation_freeze()
            self.trojan_ASR_test_evaluation_freeze()

      def trojan_ACC_test_evaluation(self):
            if self.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus']:
                  split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(self.test_graphs), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
                  idx_clean_test, idx_atk = split_trojan_evaluation['clean_test'], split_trojan_evaluation['attack']

                  test_graphs_batch = Batch.from_data_list(self.test_graphs)

                  clean_test_data_list = test_graphs_batch[idx_clean_test]
                  clean_test_loader = DataLoader(clean_test_data_list, batch_size=10, shuffle=False)
                              
                  if self.prompt_type == 'All-in-one':
                        test_acc, F1 = AllInOneEva(clean_test_loader, self.trojan_prompt, self.gnn, self.trojan_answering, self.output_dim, self.device)           
                  elif self.prompt_type in ['GPF', 'GPF-plus']:
                        test_acc = GPFEva(clean_test_loader, self.gnn, self.trojan_prompt, self.trojan_answering, self.device)                    
                  elif self.prompt_type =='Gprompt':
                        test_acc = GpromptEva(clean_test_loader, self.gnn, self.trojan_prompt, self.center_ori, self.device)

                  print("===========Before Attack=====================")
                  print("clean test accuracy {:.4f} ".format(test_acc))         
            elif self.prompt_type in ['GPPT']:
                  split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(self.idx_test), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
                  idx_clean_test = self.idx_test[split_trojan_evaluation['clean_test']]
                  idx_atk = self.idx_test[split_trojan_evaluation['attack']]

                  if self.prompt_type == 'GPPT':
                        test_acc = GPPTEva(self.data, idx_clean_test, self.gnn, self.trojan_prompt)    
                  
                  print("===========Before Attack=====================")
                  print("clean test accuracy {:.4f} ".format(test_acc))       
            else:
                  NotImplementedError("No implemented trojan evaluation for other prompts")
      
      def trojan_ACC_test_evaluation_freeze(self):
            if self.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus']:
                  split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(self.test_graphs), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
                  idx_clean_test, idx_atk = split_trojan_evaluation['clean_test'], split_trojan_evaluation['attack']

                  test_graphs_batch = Batch.from_data_list(self.test_graphs)

                  clean_test_data_list = test_graphs_batch[idx_clean_test]
                  clean_test_loader = DataLoader(clean_test_data_list, batch_size=10, shuffle=False)
                              
                  if self.prompt_type == 'All-in-one':
                        test_acc, F1 = AllInOneEva(clean_test_loader, self.trojan_prompt, self.gnn, self.answering_ori, self.output_dim, self.device)           
                  elif self.prompt_type in ['GPF', 'GPF-plus']:
                        test_acc = GPFEva(clean_test_loader, self.gnn, self.trojan_prompt, self.answering_ori, self.device)                    
                  elif self.prompt_type =='Gprompt':
                        test_acc = GpromptEva(clean_test_loader, self.gnn, self.trojan_prompt, self.center_ori, self.device)

                  print("===========Before Attack (Freeze)=====================")
                  print("clean test accuracy {:.4f} ".format(test_acc))         
            elif self.prompt_type in ['GPPT']:
                  split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(self.idx_test), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
                  idx_clean_test = self.idx_test[split_trojan_evaluation['clean_test']]
                  idx_atk = self.idx_test[split_trojan_evaluation['attack']]

                  if self.prompt_type == 'GPPT':
                        test_acc = GPPTEva(self.data, idx_clean_test, self.gnn, self.trojan_prompt)    
                  
                  print("===========Before Attack (Freeze)=====================")
                  print("clean test accuracy {:.4f} ".format(test_acc))   
            else:
                  NotImplementedError("No implemented trojan evaluation for other prompts")
            
      def trojan_ASR_test_evaluation_freeze(self):
            if self.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus']:
                  split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(self.test_graphs), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
                  idx_clean_test, idx_atk = split_trojan_evaluation['clean_test'], split_trojan_evaluation['attack']

                  test_graphs_batch = Batch.from_data_list(self.test_graphs)
                  test_graphs_batch = test_graphs_batch.cpu()

                  clean_test_data_list = test_graphs_batch[idx_clean_test]
                  attack_test_data_list = self.trigger_generator.transfer_to_trojan_test_dataset(test_graphs_batch,idx_atk, self.prompt_type, self.device)

                  clean_test_loader = DataLoader(clean_test_data_list, batch_size=self.args.trojan_batch_size, shuffle=False)
                  attack_test_loader = DataLoader(attack_test_data_list, batch_size=self.args.trojan_batch_size, shuffle=False)
                              
                  if self.prompt_type == 'All-in-one':
                        test_acc, F1 = AllInOneEva(clean_test_loader, self.trojan_prompt, self.gnn, self.answering_ori, self.output_dim, self.device)           
                        bkd_asr, F1 = AllInOneEva(attack_test_loader, self.trojan_prompt, self.gnn, self.answering_ori, self.output_dim, self.device)                                           
                  elif self.prompt_type in ['GPF', 'GPF-plus']:
                        test_acc = GPFEva(clean_test_loader, self.gnn, self.trojan_prompt, self.answering_ori, self.device)                    
                        bkd_asr = GPFEva(attack_test_loader, self.gnn, self.trojan_prompt, self.answering_ori, self.device)                                                         
                  elif self.prompt_type =='Gprompt':
                        test_acc = GpromptEva(clean_test_loader, self.gnn, self.trojan_prompt, self.center_ori, self.device)
                        bkd_asr = GpromptEva(attack_test_loader, self.gnn, self.trojan_prompt, self.center_ori, self.device)

                  print("===========After Attack (Freeze)=====================")
                  print("clean test accuracy {:.4f} ".format(test_acc))                        
                  print("attack success rate {:.4f} ".format(bkd_asr))        
                  return bkd_asr              
            elif self.prompt_type in ['GPPT']:
                  split_trojan_evaluation = trojan_utils.get_split_trojan_evaluation(len(self.idx_test), clean_test_ratio = 0.5, attack_ratio = 0.5, seed = 42, device=self.device)
                  idx_clean_test = self.idx_test[split_trojan_evaluation['clean_test']]
                  idx_atk = self.idx_test[split_trojan_evaluation['attack']]

                  # poison_data = self.trigger_generator.get_poisoned(idx_atk)
                  # clean_data = self.trigger_generator.get_clean()
                  poison_data = self.trigger_generator.get_poisoned(idx_atk)
                  clean_data = deepcopy(self.data_ori)
                  if(self.prompt_type == 'GPPT'):
                        test_acc = GPPTEva(poison_data, idx_atk, self.gnn, self.trojan_prompt)    
                        bkd_asr = GPPTEva(clean_data, idx_clean_test, self.gnn, self.trojan_prompt)    
                        # bkd_asr = self.evaluate_ASR_1by1(poison_data, idx_atk)  
                        # bkd_asr = self.evaluate_ASR_overall(poison_data, idx_atk)      
                  print("===========After Attack (Freeze)=====================")
                  print("clean test accuracy {:.4f} ".format(test_acc))                        
                  print("attack success rate {:.4f} ".format(bkd_asr))        
            else:
                  NotImplementedError("No implemented trojan evaluation for other prompts")



      def visualize_embeddings_graph_level(self):
            benign_dataset = deepcopy(self.train_graphs)

            benign_dataloader = DataLoader(benign_dataset, batch_size= 128, shuffle=False)

            trigger_dataset = []
            # trigger_dataloader = DataLoader(trigger_dataset, batch_size= 128, shuffle=False)

            benign_embd = torch.FloatTensor([]).to(self.device)
            for data in benign_dataset:
                  data = data.to(self.device)
                  data = Batch.from_data_list([data])
                  embed = self.gnn(data.x, data.edge_index, data.batch, prompt = None, prompt_type = None)
                  benign_embd = torch.concat((benign_embd,data.x), dim = 0)

                  idx_attach = trojan_utils.obtain_attach_nodes(seed=self.args.seed,node_idxs=np.array(list(range(data.x.shape[0]))),size=1)
                  trojan_edges = self.trigger_generator.get_trojan_edge(0, idx_attach=idx_attach, trigger_size=self.args.trigger_size).to(self.device)
                  trojan_feat, trojan_weights = self.trigger_generator.trojan(embed,self.args.thrd)
                  trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
                  trojan_weights = trojan_weights.flatten()
                  trojan_weights = torch.cat([trojan_weights,trojan_weights])
                  trojan_feat = trojan_feat.view([-1,data.x.shape[1]])

                  trojan_edges = trojan_edges[:,trojan_weights>0.0]
                  trojan_weights = trojan_weights[trojan_weights>0.0]
                  
                  trigger_data = Data(x = trojan_feat, edge_index=trojan_edges)
                  trigger_dataset.append(trigger_data)

            trigger_dataloader = DataLoader(trigger_dataset, batch_size= 128, shuffle=False)

            trigger_embd = torch.FloatTensor([]).to(self.device)
            for data in trigger_dataset:
                  data = Batch.from_data_list([data])
                  data = data.to(self.device)
                  # embed = self.gnn(data.x, data.edge_index, data.batch, prompt = None, prompt_type = None)
                  embed = data.x
                  trigger_embd = torch.concat((trigger_embd,embed), dim = 0)
            
            full_embd = torch.concat((benign_embd,trigger_embd))

            benign_embd = benign_embd.cpu().detach().numpy()
            trigger_embd = trigger_embd.cpu().detach().numpy()

            full_embd = full_embd.cpu().detach().numpy()


            # Reduce the dimensionality of features to 2D using PCA
            pca = PCA(n_components=2)
            full_embd_2d = pca.fit_transform(full_embd)

            # Split the data into two groups
            trigger_embd_2d = full_embd_2d[benign_embd.shape[0]:]
            benign_embd_2d = full_embd_2d[:benign_embd.shape[0]]

            # last_30 = features_2d[-len(idx_attach)*self.args.trigger_size:]  # Last 30 rows
            # rest = features_2d[:-len(idx_attach)*self.args.trigger_size]    # All other rows
            # last_30 = features_2d[-len(idx_attach)*2:]  # Last 30 rows
            # rest = features_2d[:-len(idx_attach)*2]    # All other rows

            indices = []
            for i in range(3, 33, 3):  # Start from 0, up to 29, step by 3
                  indices.append(-i)


            # Plotting
            plt.figure(figsize=(8, 6))

            # Plot for clean samples
            plt.scatter(benign_embd_2d[:, 0], benign_embd_2d[:, 1], color='cornflowerblue', label='Clean', s=20)
            # Plot for triggers
            plt.scatter(trigger_embd_2d[:, 0], trigger_embd_2d[:, 1], color='red', label='Triggers', s=20)
            # plt.scatter(rest[:, 0], rest[:, 1], color='cornflowerblue', label='Clean')
            # # Plot for triggers
            # plt.scatter(last_30[:, 0], last_30[:, 1], color='red', label='Triggers')

            legend = plt.legend(fontsize=22, loc='upper left')

            legend.legendHandles[0]._sizes = [50]  # Adjust clean sample size
            legend.legendHandles[1]._sizes = [50]

            # Remove x and y ticks
            plt.xticks([])
            plt.yticks([])

            # Save the plot as a PDF
            plt.savefig('./figures/highlighted_embeddings.png')
            plt.savefig('./figures/highlighted_embeddings.pdf', format='pdf', bbox_inches='tight')
      def run(self):
            test_accs = []
            # if self.prompt_type == 'MultiGprompt':    
            for i in range(1, 2):
                  # self.dataset_name ='Cora'
                  if(self.args.training_split == 'k-shot'):
                        idx_train = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/train_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).to(self.device)
                        train_lbls = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/train_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().to(self.device)

                        idx_test = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/test_idx.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).to(self.device)
                        test_lbls = torch.load("./Experiment/sample_data/Node/{}/{}_shot/{}/test_labels.pt".format(self.dataset_name, self.shot_num, i)).type(torch.long).squeeze().to(self.device)
                  
                  elif(self.args.training_split == 'classical'):
                        split = trojan_utils.get_split_self(num_samples=self.data.x.shape[0], train_ratio=0.7, test_ratio=0.2, seed=self.args.seed)
                        idx_train = split['train']
                        idx_valid = split['valid']
                        idx_test = split['test']


                  # for all-in-one and Gprompt we use k-hop subgraph
                  if self.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus', 'All-in-one-meta']:
                        if(self.dataset_name in ['Flickr', 'ogbn-arxiv','Physics']):
                              train_graphs = self.load_induced_graph_random_split_specific(idx_train, 'train_trojan')
                              test_graphs = self.load_induced_graph_random_split_specific(idx_test, 'test_trojan')
                        else:
                              graphs_list = self.load_induced_graph()

                              # split_self = trojan_utils.get_split_self(len(graphs_list), train_ratio = 0.1, test_ratio = 0.85, seed = 42, device=self.device)

                              # idx_train = split_self['train']
                              # idx_test = split_self['test']

                              train_graphs = []
                              test_graphs = []
                              
                              for graph in graphs_list:                              
                                    if graph.index in idx_train:
                                          train_graphs.append(graph)
                                    elif graph.index in idx_test:
                                          test_graphs.append(graph)

                        train_loader = DataLoader(train_graphs, batch_size=self.args.dt_batch_size, shuffle=True)
                        test_loader = DataLoader(test_graphs, batch_size=self.args.dt_batch_size, shuffle=False)
                        print("prepare induce graph data is finished!")
                        self.test_graphs = test_graphs
                        self.train_graphs = train_graphs
                        if(self.prompt_type=='All-in-one-meta'):
                              pass
                        

                  elif self.prompt_type == 'MultiGprompt':
                        embeds, _ = self.Preprompt.embed(self.features, self.sp_adj, True, None, False)
                        pretrain_embs = embeds[0, idx_train]
                        test_embs = embeds[0, idx_test]

                  elif self.prompt_type == 'GPPT':
                        self.idx_train = idx_train.to(self.device)
                        self.idx_test = idx_test.to(self.device)

                        idx_full = torch.tensor(range(self.data.x.shape[0])).to(self.device)

                        idx_train_test = torch.concat((self.idx_train,self.idx_test))
                        train_test_mask = torch.isin(idx_full, idx_train_test)
                        idx_unlabeled = idx_full[~train_test_mask]

                        self.idx_unlabeled = idx_unlabeled
                  elif self.prompt_type == 'All-in-one-meta':
                        pass
                  patience = 10000
                  best = 1e9
                  cnt_wait = 0
                 

                  best_test_acc = 0.0

                  for epoch in range(1, self.epochs):
                        t0 = time.time()
                        if self.prompt_type == 'None':
                              loss = self.train(self.data, idx_train)                             
                        elif self.prompt_type == 'GPPT':
                              loss = self.GPPTtrain(self.data, idx_train)    
                              self.trojan_prompt = deepcopy(self.prompt)            
                        elif self.prompt_type == 'All-in-one':
                              loss = self.AllInOneTrain(train_loader)     
                              self.trojan_prompt = deepcopy(self.prompt)
                              self.trojan_answering = deepcopy(self.answering)                                  
                        elif self.prompt_type in ['GPF', 'GPF-plus']:
                              loss = self.GPFTrain(train_loader)  
                              self.trojan_prompt = deepcopy(self.prompt)
                              self.trojan_answering = deepcopy(self.answering)                                                        
                        elif self.prompt_type =='Gprompt':
                              loss, center = self.GpromptTrain(train_loader)
                              self.trojan_prompt = deepcopy(self.prompt)
                              self.trojan_center = center
                              self.center_ori = center.clone()
                        elif self.prompt_type == 'MultiGprompt':
                              loss = self.MultiGpromptTrain(pretrain_embs, train_lbls, idx_train)


                        if loss < best:
                              best = loss
                              # best_t = epoch
                              cnt_wait = 0
                              # torch.save(model.state_dict(), args.save_name)
                              '''
                              save checkpoint for prompt and answering
                              '''
                              prompt_fp = self.args.prompt_path
                        
                              torch.save(self.prompt, prompt_fp)
                              if(self.prompt_type in ['All-in-one', 'GPF', 'GPF-plus']):
                                    answer_fp = self.args.answer_path
                                    torch.save(self.answering, answer_fp)
                              elif(self.prompt_type in ['Gprompt']):
                                    answer_fp = self.args.answer_path
                                    torch.save(self.center_ori, answer_fp)
                        else:
                              cnt_wait += 1
                              if cnt_wait == patience:
                                    print('-' * 100)
                                    print('Early stopping at '+str(epoch) +' eopch!')
                                    break
                        print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f}  ".format(epoch, time.time() - t0, loss))

                        
                        if self.prompt_type == 'None':
                              test_acc = GNNNodeEva(self.data, idx_test, self.gnn, self.answering)                           
                        elif self.prompt_type == 'GPPT':
                              test_acc = GPPTEva(self.data, idx_test, self.gnn, self.prompt)                
                        elif self.prompt_type == 'All-in-one':
                              test_acc, F1 = AllInOneEva(test_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)                                           
                        elif self.prompt_type in ['GPF', 'GPF-plus']:
                              test_acc = GPFEva(test_loader, self.gnn, self.prompt, self.answering, self.device)                                                         
                        elif self.prompt_type =='Gprompt':
                              test_acc = GpromptEva(test_loader, self.gnn, self.prompt, center, self.device)
                              # if(test_acc > best_test_acc):
                              #       best_test_acc = test_acc
                              #       self.trojan_prompt = deepcopy(self.prompt)
                              #       self.trojan_center = center
                              #       self.center_ori = center.clone()
                              #       print("checkpoints copied.")

                        elif self.prompt_type == 'MultiGprompt':
                              prompt_feature = self.feature_prompt(self.features)
                              test_acc = MultiGpromptEva(test_embs, test_lbls, idx_test, prompt_feature, self.Preprompt, self.DownPrompt, self.sp_adj)

                        print("test accuracy {:.4f} ".format(test_acc))
                  test_accs.append(test_acc)
         

            mean_test_acc = np.mean(test_accs)
            std_test_acc = np.std(test_accs)    
            print(" Final best | test Accuracy {:.4f} | std {:.4f} ".format(mean_test_acc, std_test_acc))         
            