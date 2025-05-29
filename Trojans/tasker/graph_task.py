import torch
from prompt_graph.data import load4graph,load4node
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from .task import BaseTask
from prompt_graph.utils import center_embedding, Gprompt_tuning_loss
from prompt_graph.evaluation import GpromptEva, GNNGraphEva, GPFEva, AllInOneEva
import time

from copy import deepcopy

import Trojans.trojan_utils as trojan_utils
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import TUDataset


class GraphTask(BaseTask):
    def __init__(self, *args, **kwargs):    
        super().__init__(*args, **kwargs)
        self.load_data()
        self.initialize_gnn()
        self.initialize_prompt()
        self.answering =  torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim),
                                            torch.nn.Softmax(dim=1)).to(self.device)
        self.initialize_optimizer()

        self.trojan_prompt = deepcopy(self.prompt)
        self.trojan_answering = deepcopy(self.answering)
        self.prompt_ori = deepcopy(self.prompt)
        self.answering_ori = deepcopy(self.answering)
    

    def load_data(self):
        if self.dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']:
            self.input_dim, self.output_dim, self.train_dataset, self.test_dataset, self.val_dataset, self.dataset = load4graph(self.dataset_name, self.shot_num)
        # if self.dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY']:
        #     self.dataset = TUDataset(root='data/TUDataset', name=self.dataset_name)
        #     self.input_dim = self.dataset.num_features
        #     self.out_dim = self.dataset.num_classes

    def Train(self, train_loader):
        self.gnn.train()
        total_loss = 0.0 
        for batch in train_loader:  
            self.optimizer.zero_grad() 
            batch = batch.to(self.device)
            out = self.gnn(batch.x, batch.edge_index, batch.batch)
            out = self.answering(out)
            loss = self.criterion(out, batch.y)  
            loss.backward()  
            self.optimizer.step()  
            total_loss += loss.item()  
        return total_loss / len(train_loader)  
        
    def AllInOneTrain(self, train_loader):
        #we update answering and prompt alternately.
        
        answer_epoch = 1  # 50
        prompt_epoch = 1  # 50
        
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

    def GPFTrain(self, train_loader):
        self.prompt.train()
        total_loss = 0.0 
        for batch in train_loader:  
            self.optimizer.zero_grad() 
            batch = batch.to(self.device)
            batch.x = self.prompt.add(batch.x)
            out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = self.prompt_type)
            out = self.answering(out)
            loss = self.criterion(out, batch.y)  
            loss.backward()  
            self.optimizer.step()  
            total_loss += loss.item()  
        return total_loss / len(train_loader)  
    
    def GpromptTrain(self, train_loader):
        self.prompt.train()
        total_loss = 0.0 
        for batch in train_loader:  
            self.pg_opi.zero_grad() 
            batch = batch.to(self.device)
            out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = 'Gprompt')
            # out = s𝑡,𝑥 = ReadOut({p𝑡 ⊙ h𝑣 : 𝑣 ∈ 𝑉 (𝑆𝑥)}),
            center = center_embedding(out, batch.y, self.output_dim)
            criterion = Gprompt_tuning_loss()
            loss = criterion(out, center, batch.y)  
            loss.backward()  
            self.pg_opi.step()  
            total_loss += loss.item()  
        return total_loss / len(train_loader)  
        

    def prompt_graph_poisoning(self):
            # trojan_prompt = self.trojan_prompt
            # trojan_answering = self.trojan_answering
            trojan_prompt = deepcopy(self.prompt)
            trojan_answering = deepcopy(self.answering)
                  
            train_graphs_batch = Batch.from_data_list(self.train_graphs)

            num_poison = int(len(train_graphs_batch) * self.args.poison_ratio)
            print("Number of poisoning graphs: {}".format(num_poison))
    
            idx_trojan = trojan_utils.obtain_attach_nodes(seed=10,node_idxs=np.array(range(len(train_graphs_batch))),size=num_poison)
            print("Selected trojan graphs in train dataset: {}".format(idx_trojan))
            
            if(self.args.compare_method == 'Random'):
                  from Trojans.random_prompt_backdoor import RandomBackdoor
                  self.trigger_generator = RandomBackdoor(self.args, self.num_classes, self.device)
                  if(self.prompt_type=='Gprompt'):
                        self.trojan_center = self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan)
                        self.trojan_prompt = self.trigger_generator.prompt_generator
                  else:      
                        self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan)
                        self.trojan_prompt = self.trigger_generator.prompt_generator
                        self.trojan_answering = self.trigger_generator.downstream_tasker
            elif(self.args.compare_method == 'GTA'):
                  from Trojans.GTA import Backdoor
                  self.trigger_generator = Backdoor(self.args,self.num_classes,self.device)
                  # self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan)
                  if(self.prompt_type=='Gprompt'):
                        self.trojan_center = self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan, self.trojan_center)
                        self.trojan_prompt = self.trigger_generator.prompt_generator
                  else:      
                        self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan)
                        self.trojan_prompt = self.trigger_generator.prompt_generator
                        self.trojan_answering = self.trigger_generator.downstream_tasker
            elif(self.args.compare_method == 'UGBA'):
                  from Trojans.UGBA import Backdoor
                  self.trigger_generator = Backdoor(self.args,self.num_classes,self.device)
                  # self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan)
                  if(self.prompt_type=='Gprompt'):
                        self.trojan_center = self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan, self.trojan_center)
                        self.trojan_prompt = self.trigger_generator.prompt_generator
                  else:      
                        self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan)
                        self.trojan_prompt = self.trigger_generator.prompt_generator
                        self.trojan_answering = self.trigger_generator.downstream_tasker
            elif(self.args.compare_method == 'Ours'):
                  # self.trojan_answering = self.trigger_generator.answering # note that we do not tune the answering, just for clearify
                  if(self.prompt_type=='Gprompt'):
                        from Trojans.backdoor import Backdoor
                        self.trigger_generator = Backdoor(self.args,self.num_classes,self.device)
                        self.trojan_center = self.trigger_generator.fit(self.prompt_type, trojan_prompt, self.gnn, trojan_answering, train_graphs_batch, idx_trojan, self.trojan_center)
                        self.trojan_prompt = self.trigger_generator.prompt_generator
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
                        self.trojan_answering = self.trigger_generator.downstream_tasker
            else:
                  NotImplementedError("No implemented backdoor attack methods.")

    def run(self):

        train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=16, shuffle=False)
        val_loader = DataLoader(self.val_dataset, batch_size=16, shuffle=False)
        print("prepare data is finished!")
        best_val_acc = final_test_acc = 0
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            if self.prompt_type == 'None':
                loss = self.Train(train_loader)
                test_acc = GNNGraphEva(test_loader, self.gnn, self.answering, self.device)
                val_acc = GNNGraphEva(val_loader, self.gnn, self.answering, self.device)
            elif self.prompt_type == 'All-in-one':
                loss = self.AllInOneTrain(train_loader)
                test_acc, F1 = AllInOneEva(test_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
                val_acc, F1 = AllInOneEva(val_loader, self.prompt, self.gnn, self.answering, self.output_dim, self.device)
            elif self.prompt_type in ['GPF', 'GPF-plus']:
                loss = self.GPFTrain(train_loader)
                test_acc = GPFEva(test_loader, self.gnn, self.prompt, self.answering, self.device)
                val_acc = GPFEva(val_loader, self.gnn, self.prompt, self.answering, self.device)
            elif self.prompt_type =='Gprompt':
                loss = self.GpromptTrain(train_loader)
                test_acc = GpromptEva(test_loader, self.gnn, self.prompt, self.answering, self.device)
                val_acc = GpromptEva(val_loader, self.gnn, self.prompt, self.answering, self.device)
                    

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
            print("Epoch {:03d}/{:03d}  |  Time(s) {:.4f}| Loss {:.4f} | val Accuracy {:.4f} | test Accuracy {:.4f} ".format(epoch, self.epochs, time.time() - t0, loss, val_acc, test_acc))
            
        print(f'Final Test: {final_test_acc:.4f}')
        
        print("Graph Task completed")

        

        
