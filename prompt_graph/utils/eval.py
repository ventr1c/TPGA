import numpy as np
import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

def accuracy(output, labels):
    """Return accuracy of output compared to labels.
    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels
    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def get_split_self(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8, seed: int = 42, device=None):
    """Return indices for train, test, and valid splits."""
    assert train_ratio + test_ratio <= 1
    rs = np.random.RandomState(seed)
    perm = rs.permutation(num_samples)
    indices = torch.tensor(perm).to(device)
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    # indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'test': indices[train_size: test_size + train_size],
        'valid': indices[test_size + train_size:]
    }