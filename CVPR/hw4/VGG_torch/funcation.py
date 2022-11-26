import torch


def get_acc(out, label):
    out_index = torch.argmax(out, dim=1)
    right = (out_index==label)
    return torch.sum(right)


if __name__ == '__main__':
    out = torch.Tensor([[12, -31, 13], [11, 32, 112]])
    label = torch.Tensor([2, 1])
    print(get_acc(out, label))