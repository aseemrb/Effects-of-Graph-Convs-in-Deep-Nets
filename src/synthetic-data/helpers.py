import numpy as np
import torch
import torch.nn as nn
import torch.linalg as linalg
import torch_geometric.utils as utils
import torch_geometric.nn as gnn
from torch.distributions import Bernoulli, MultivariateNormal
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum

class XCSBM:
    def __init__(self, n_points, n_features, dist_sd_ratio, sig_sq=None, p=1., q=0.):
        self.n_points = n_points
        self.n_features = n_features
        self.K = dist_sd_ratio
        self.sig_sq = 1./n_features if sig_sq is None else sig_sq
        self.p = p
        self.q = q
        # Fix a random pair of orthogonal means.
        u = torch.rand(n_features) # First mean.
        v = torch.rand(n_features) # Second mean.
        v[-1] = -(u[:-1] @ v[:-1]) / u[-1] # Make v orthogonal to u.
        u /= linalg.norm(u)
        v /= linalg.norm(v)
        self.normed_uv = torch.stack((u, v))
        
    # The XGMM synthetic data model definition.
    def xgmm(self):
        # Make K = norm(u-v) = sqrt(2)*norm(u).
        K_ = self.K/np.sqrt(2.0)
        u = K_*self.normed_uv[0]
        v = K_*self.normed_uv[1]
        X = torch.zeros((self.n_points, self.n_features))
        y = torch.zeros(self.n_points, dtype=torch.long)
        # Decide class and cluster based on two independent Bernoullis.
        eps = Bernoulli(torch.tensor([0.5]))
        eta = Bernoulli(torch.tensor([0.5]))
        for i in range(self.n_points):
            y[i] = eps.sample()
            cluster = eta.sample()
            # Mean is -mu, mu, -nu or nu based on eps_i and eta_i.
            mean = (2*cluster - 1)*((1-y[i])*u + y[i]*v)
            if self.sig_sq > 0:
                cov = torch.eye(self.n_features) * self.sig_sq
                distr = MultivariateNormal(mean, cov)
                X[i] = distr.sample()
            else:
                X[i] = mean
        return Data(x=X, y=y, edge_index=None)

    # Generate a dataset from the XCSBM synthetic data model.
    def generate_data(self):
        data = self.xgmm()
        # The inbuilt function stochastic_blockmodel_graph does not support
        # random permutations of the nodes, hence, design it manually.
        # Use with_replacement=True to include self-loops.
        probs = torch.tensor([[self.p, self.q], [self.q, self.p]], dtype=torch.float)
        row, col = torch.combinations(torch.arange(self.n_points), r=2, with_replacement=True).t()
        mask = torch.bernoulli(probs[data.y[row], data.y[col]]).to(torch.bool)
        data.edge_index = torch.stack([row[mask], col[mask]], dim=0)
        data.edge_index = utils.to_undirected(data.edge_index, num_nodes=self.n_points)
        return data

# MLP with ReLU activations and sigmoidal output.
class MLP(torch.nn.Module):
    def __init__(self, n_layers, n_features, channels=None):
        super().__init__()
        self.n_layers = n_layers
        self.activations = [nn.ReLU()]*n_layers
        self.activations[-1] = nn.Sigmoid()
        # Set default number of channels for every layer if not specified.
        # channels[0] stores input dimensions for each layer.
        # channels[1] stores output dimensions for each layer.
        if channels is None:
            in_channels = [4]*n_layers
            in_channels[0] = n_features
            out_channels = [4]*n_layers
            out_channels[-1] = 1
            channels = [in_channels, out_channels]
        
        assert channels[0][0] == n_features, "Input dimension of the first layer must match dimension of data."
        assert channels[1][-1] == 1, "Output dimension of the last layer must be 1."
        self.module_list = []
        for i in range(n_layers):
            self.module_list.append(nn.Sequential(
                nn.Linear(channels[0][i], channels[1][i], bias=False),
                self.activations[i]
            ))
        self.module_list = nn.ModuleList(self.module_list)

    def forward(self, data):
        x = data.x
        for module in self.module_list:
            x = module(x)
        out = torch.squeeze(x, dim=1)
        return out

# GCN with ReLU activations and sigmoidal output.
class GCN(torch.nn.Module):
    def __init__(self, n_layers, n_features, convolutions, channels=None):
        super().__init__()
        self.n_layers = n_layers
        self.num_nodes = None
        self.norm = None
        self.convs = convolutions
        self.activations = [nn.ReLU()]*n_layers
        self.activations[-1] = nn.Sigmoid()
        
        # Set default number of channels for every layer if not specified.
        # channels[0] stores input dimensions for each layer.
        # channels[1] stores output dimensions for each layer.
        if channels is None:
            in_channels = [4]*n_layers
            in_channels[0] = n_features
            out_channels = [4]*n_layers
            out_channels[-1] = 1
            channels = [in_channels, out_channels]
        
        assert channels[0][0] == n_features, "Input dimension of the first layer must match dimension of data."
        assert channels[1][-1] == 1, "Output dimension of the last layer must be 1."
        assert len(convolutions) == n_layers, "Length of 'convolutions' must be the same as the number of layers."
        self.module_list = []
        for i in range(n_layers):
            self.module_list.append(nn.Sequential(
                nn.Linear(channels[0][i], channels[1][i], bias=False),
                self.activations[i]
            ))
        self.module_list = nn.ModuleList(self.module_list)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.num_nodes is None:
            self.num_nodes = utils.num_nodes.maybe_num_nodes(edge_index)
            # Compute vector of deg^{-1}.
            edge_weight = torch.ones((edge_index.size(1), ), dtype=None, device=edge_index.device)
            row, col = edge_index
            deg = scatter_add(edge_weight, row, dim=0, dim_size=self.num_nodes)
            deg_inv = deg.pow_(-1)
            deg_inv[deg_inv == float('inf')] = 0
            self.norm = deg_inv.view(-1,1) # Normalize node features by deg^{-1}.
        for (i, module) in enumerate(self.module_list):
            x = self.convolve(x, edge_index, self.convs[i])
            x = module(x)
        out = torch.squeeze(x, dim=1)
        return out
    
    def convolve(self, x, edge_index, k):
        row, col = edge_index
        x_ = x
        for i in range(k):
            x_ = scatter_add(x_[row], col, dim=0, dim_size=self.num_nodes)
            x_ = self.norm*x_
        return x_

<<<<<<< HEAD
def train_model(model, data, loss_fn=nn.BCELoss(), lr=0.001, epochs=200, eps=1e-5, logs=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    epoch = 0
    print_freq = epochs//10
    prev_loss = 0
    while epoch < epochs:
        optimizer.zero_grad(set_to_none=True)
        out = model(data)
        loss = loss_fn(out[data.train_mask], data.y_[data.train_mask].float())
        loss.backward()
        optimizer.step()
        if logs is not None and epoch % print_freq == 0:
            print(logs + ' Loss: ' + str(round(loss.item(), 5)) + '\t\t', end='\r')
        if np.abs(prev_loss - loss.item()) <= eps and loss.item() < eps:
            break
        epoch += 1
        prev_loss = loss.item()

# Set parameters to be the ansatz.
def set_params(net, data_model, n_layers, device, R=1):
    params = [param for param in net.parameters()]
    u = R*data_model.normed_uv[0]
    v = R*data_model.normed_uv[1]
=======
def train_model(model, data, loss_fn=nn.BCELoss(), epochs=200, min_loss=1e-3, idx=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    model.train()
    epoch = 0
    print_freq = epochs//10
    while epoch < epochs:
        optimizer.zero_grad(set_to_none=True)
        out = model(data)
        loss = loss_fn(out, data.y.float())
        loss.backward()
        optimizer.step()
        if epoch % print_freq == 0 and idx is not None:
            print('\rIdx:', idx, ', Loss:', loss.item(), end='')
        if loss.item() < min_loss:
            break
        epoch += 1

# Set parameters to be the ansatz.
def set_params(net, data_model, n_layers, device):
    params = [param for param in net.parameters()]
    u = data_model.normed_uv[0]
    v = data_model.normed_uv[1]
>>>>>>> 102ffc02b2c24fe0de1705bc340d472557cb4528
    params[0].data = torch.stack([u, -u, v, -v], dim=0).to(device)
    if n_layers == 2:
        params[1].data = torch.tensor([[-1., -1., 1., 1.]]).to(device)
    elif n_layers == 3:
        params[1].data = torch.tensor([[-1., -1., 1., 1.],[1., 1., -1., -1.]]).to(device)
        params[2].data = torch.tensor([[1., -1.]]).to(device)

# Compute prediction accuracy.
def accuracy(y_hat, y):
    assert y_hat.size(0) == y.size(0)
    assert y.size(0) > 0
    pred = (y_hat>=0.5).long()
    incorrect = torch.sum(torch.abs(pred-y))
    return 1. - (incorrect / y.size(0))