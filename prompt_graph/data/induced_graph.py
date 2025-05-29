# from collections import defaultdict
import pickle as pk
from torch_geometric.utils import subgraph, k_hop_subgraph
import torch
import numpy as np
from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.data import Data, Batch
import random
import os
from prompt_graph.utils import mkdir
from random import shuffle
from torch_geometric.utils import subgraph, k_hop_subgraph
from torch_geometric.data import Data
import numpy as np
import pickle

def induced_graphs(data, smallest_size=10, largest_size=30):

    induced_graph_list = []

    for index in range(data.x.size(0)):
        current_label = data.y[index].item()

        current_hop = 2
        subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                            edge_index=data.edge_index, relabel_nodes=True)
        
        while len(subset) < smallest_size and current_hop < 5:
            current_hop += 1
            subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                                edge_index=data.edge_index)
            
        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            pos_nodes = torch.argwhere(data.y == int(current_label)) 
            candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), subset.numpy()))
            candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]
            subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

        if len(subset) > largest_size:
            subset = subset[torch.randperm(subset.shape[0])][0:largest_size - 1]
            subset = torch.unique(torch.cat([torch.LongTensor([index]), torch.flatten(subset)]))

        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)

        x = data.x[subset]

        induced_graph = Data(x=x, edge_index=sub_edge_index, y=current_label)
        induced_graph_list.append(induced_graph)
        # print(index)
    return induced_graph_list



def split_induced_graphs(name, data, smallest_size=10, largest_size=30):

    induced_graph_list = []
    
    for index in range(data.x.size(0)):
        current_label = data.y[index].item()

        current_hop = 2
        subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                            edge_index=data.edge_index, relabel_nodes=True)
        
        # while len(subset) < smallest_size and current_hop < 2:
        #     current_hop += 1
        #     subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
        #                                         edge_index=data.edge_index)
            
        # if len(subset) < smallest_size:
        #     need_node_num = smallest_size - len(subset)
        #     pos_nodes = torch.argwhere(data.y == int(current_label)) 
        #     candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), subset.numpy()))
        #     candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]
        #     subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

        # if len(subset) > largest_size:
        #     subset = subset[torch.randperm(subset.shape[0])][0:largest_size - 1]
        #     subset = torch.unique(torch.cat([torch.LongTensor([index]), torch.flatten(subset)]))

        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)

        x = data.x[subset]

        induced_graph = Data(x=x, edge_index=sub_edge_index, y=current_label, index = index)
        induced_graph_list.append(induced_graph)
        if index%50 == 0:
            print(index)

    dir_path = f'./Experiment/induced_graph/{name}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path) 

    file_path = os.path.join(dir_path, 'induced_graph.pkl')
    with open(file_path, 'wb') as f:
        # Assuming 'data' is what you want to pickle
        pickle.dump(induced_graph_list, f) 

import os
import torch
from torch_geometric.utils import k_hop_subgraph, subgraph
from torch_geometric.data import Data
import pickle
import numpy as np

def split_fixed_neighbor_khop_graphs(name, data, K=2, num_neighbors=2):
    fixed_neighbor_graph_list = []
    
    for index in range(data.x.size(0)):
        current_label = data.y[index].item()

        # Get the k-hop subgraph for the current node
        subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=K, edge_index=data.edge_index, relabel_nodes=True)
        
        # Ensure the subset has at least `num_neighbors` neighbors
        if len(subset) > num_neighbors:
            # Sample `num_neighbors` neighbors from the k-hop subset
            sampled_neighbors = torch.tensor(np.random.choice(subset[1:].numpy(), num_neighbors, replace=False), dtype=torch.long)
            # Include the current node in the subset
            subset = torch.cat([torch.tensor([index], dtype=torch.long), sampled_neighbors])
        elif len(subset) == num_neighbors:
            # If the subset size is exactly `num_neighbors`, no need to sample
            subset = subset
        else:
            # If fewer neighbors than needed, sample with replacement
            sampled_neighbors = torch.tensor(np.random.choice(subset[1:].numpy(), num_neighbors, replace=True), dtype=torch.long)
            subset = torch.cat([torch.tensor([index], dtype=torch.long), sampled_neighbors])

        # Get the subgraph induced by this subset
        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)

        # Create the subgraph Data object
        x = data.x[subset]
        fixed_neighbor_graph = Data(x=x, edge_index=sub_edge_index, y=current_label, index=index)
        fixed_neighbor_graph_list.append(fixed_neighbor_graph)
        
        if index % 50 == 0:
            print(f"Processed {index} nodes")
    
    # Save the induced graphs to a file
    dir_path = f'./Experiment/fixed_neighbor_khop_graph/{name}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path) 

    file_path = os.path.join(dir_path, f'fixed_neighbor_khop_graph_{K}_hop_{num_neighbors}_neighbors.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(fixed_neighbor_graph_list, f) 

    return fixed_neighbor_graph_list


def split_khop_induced_graphs(name, data, K=2):

    induced_graph_list = []
    
    for index in range(data.x.size(0)):
        current_label = data.y[index].item()

        current_hop = K
        subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                            edge_index=data.edge_index, relabel_nodes=True)
        
        # while len(subset) < smallest_size and current_hop < 2:
        #     current_hop += 1
        #     subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
        #                                         edge_index=data.edge_index)
            
        # if len(subset) < smallest_size:
        #     need_node_num = smallest_size - len(subset)
        #     pos_nodes = torch.argwhere(data.y == int(current_label)) 
        #     candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), subset.numpy()))
        #     candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]
        #     subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

        # if len(subset) > largest_size:
        #     subset = subset[torch.randperm(subset.shape[0])][0:largest_size - 1]
        #     subset = torch.unique(torch.cat([torch.LongTensor([index]), torch.flatten(subset)]))

        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)

        x = data.x[subset]

        induced_graph = Data(x=x, edge_index=sub_edge_index, y=current_label, index = index)
        induced_graph_list.append(induced_graph)
        if index%50 == 0:
            print(index)
    
    dir_path = f'./Experiment/induced_graph/{name}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path) 

    file_path = os.path.join(dir_path, 'induced_graph_{}_hop.pkl'.format(K))
    with open(file_path, 'wb') as f:
        # Assuming 'data' is what you want to pickle
        pickle.dump(induced_graph_list, f) 

    return induced_graph_list

def split_khop_induced_graphs_specific(name, data, idx_list, flag, K=2):

    induced_graph_list = []
    
    for index in idx_list:
        current_label = data.y[index].item()

        current_hop = K
        subset, _, _, _ = k_hop_subgraph(node_idx=[index], num_hops=current_hop,
                                            edge_index=data.edge_index, relabel_nodes=True)
        
        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)

        x = data.x[subset]

        induced_graph = Data(x=x.cpu(), edge_index=sub_edge_index.cpu(), y=current_label, index = index.cpu())
        induced_graph_list.append(induced_graph)
        if index%50 == 0:
            print(index)
        # Explicitly delete variables and free memory
        del subset, sub_edge_index, x, induced_graph, index
        torch.cuda.empty_cache()
    
    dir_path = f'./Experiment/induced_graph/{name}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path) 

    file_path = os.path.join(dir_path, 'induced_graph_{}_hop_{}.pkl'.format(K, flag))
    with open(file_path, 'wb') as f:
        # Assuming 'data' is what you want to pickle
        pickle.dump(induced_graph_list, f) 

    return induced_graph_list
