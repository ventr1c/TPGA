import torch

def swd(p1, p2, device, n_repeat_projection = 128, proj_per_repeat = 4):
    p1, p2 = p1.to(device), p2.to(device)
    distances = []
    for j in range(n_repeat_projection):
        # random
        rand = torch.randn(p1.size(1), proj_per_repeat).to(device)  # (slice_size**2*ch)
        rand = rand / torch.std(rand, dim=0, keepdim=True)  # noramlize
        # projection
        proj1 = torch.matmul(p1, rand)
        proj2 = torch.matmul(p2, rand)
        proj1, _ = torch.sort(proj1, dim=0)
        proj2, _ = torch.sort(proj2, dim=0)
        d = torch.abs(proj1 - proj2)
        distances.append(torch.mean(d))
    return torch.mean(torch.stack(distances))

if __name__ == '__main__':
    p1 = torch.randn(200, 128)
    p2 = torch.randn(100, 128)

    swdist = swd(p1,p2, 'cuda')
    print(swdist)