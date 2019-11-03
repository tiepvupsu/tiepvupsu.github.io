import numpy as np
import torch
from torch import nn


def conv1d(a, w, b=0, stride=1, pad=0):
    """
    compute 1d convolutional (with bias)
    """
    w_old = a.shape[0]
    f = w.shape[0]
    a_pad = np.pad(a, pad_width=pad, mode='constant', constant_values=0)
    w_new = int((w_old - f + 2*pad)/stride) + 1
    a_res = np.zeros((w_new))
    for i in range(w_new):
        start = i*stride
        end = start + f
        a_res[i] = np.sum(a_pad[start:end]*w) + b
    return a_res


def test_conv1d():
    a = np.random.rand(10)
    b = np.random.rand(1)
    w = np.random.rand(3)
    numpy_res = conv1d(a, w, b)

    conv1 = nn.Conv1d(1, 1, 3, stride=1)
    conv1.weight.data = torch.tensor(w[np.newaxis, np.newaxis, :])
    conv1.bias.data = torch.tensor(b)
    input = torch.tensor(a[np.newaxis, np.newaxis, :])
    output = conv1(input)
    print(output.cpu().detach().numpy())
    print(numpy_res)
    print(output)


if __name__ == "__main__":
    conv1 = nn.Conv1d(1, 1, 3, stride=1)
    conv1.weight.data = torch.tensor([[[1., 1., 1.]]])
    conv1.bias.data = torch.tensor([1.])
    input = torch.tensor([[[1., 1., 1.]]])
    output = conv1(input)
    print(output)
