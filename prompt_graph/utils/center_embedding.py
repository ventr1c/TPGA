import torch

# ours
def center_embedding(input, index, label_num):
    device=input.device
    '''
    It initializes a tensor c of zeros with dimensions (label_num, input.size(1)). This tensor will store the summed embeddings for each class.
    '''
    c = torch.zeros(label_num, input.size(1)).to(device)
    '''
    It uses scatter_add_ to accumulate the embeddings for each class based on the provided index.
    '''
    c = c.scatter_add_(dim=0, index=index.unsqueeze(1).expand(-1, input.size(1)), src=input)
    '''
    It calculates the count of samples for each class using torch.bincount and stores it in class_counts.
    '''
    class_counts = torch.bincount(index, minlength=label_num).unsqueeze(1).to(dtype=input.dtype, device=device)

    # Take the average embeddings for each class
    # If directly divided the variable 'c', maybe encountering zero values in 'class_counts', such as the class_counts=[[0.],[4.]]
    # So we need to judge every value in 'class_counts' one by one, and seperately divided them.
    # output_c = c/class_counts
    for i in range(label_num):
        if(class_counts[i].item()==0):
            continue
        else:
            c[i] /= class_counts[i]

    return c, class_counts

def distance2center(input,center):
    n = input.size(0)
    k = center.size(0)
    input_power = torch.sum(input * input, dim=1, keepdim=True).expand(n, k)
    center_power = torch.sum(center * center, dim=1).expand(n, k)

    distance = input_power + center_power - 2 * torch.mm(input, center.transpose(0, 1))
    return distance
