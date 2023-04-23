import torch


def rand_img(*size):
    distr = torch.rand(size)

    return distr.clone()


def img_to_weighted_sample(img):
    if type(img) != torch.Tensor:
        img = torch.tensor(img)
    N = img.size()[0]
    M = img.size()[1]
    x_grid = torch.linspace(0, N - 1, steps=N)
    y_grid = torch.linspace(0, M - 1, steps=M)
    grid = torch.cartesian_prod(x_grid, y_grid).double()

    return grid, img.reshape(N * M).double()
