import torch
import torch_geometric as tg
import torch.nn.functional as F
import torch_scatter as tc
import torch_scatter
from enum import Enum
from functools import partial

class AggregationType(Enum):
    ADD = 'add'
    MEAN = 'mean'
    MAX = 'max'
    MIN = 'min'
    MUL = 'mul'


class messagePassing(torch.nn.Module):
    def __init__(self, agg: AggregationType = AggregationType.ADD, **kwargs):
        super(messagePassing, self).__init__()
        self.agg = partial(torch_scatter.scatter, reduce = agg)
    
    def forward(self, x, edge_index, edge_attr = None, self_loop = False):
        # first gather the values from the source nodes
        temp = torch.index_select(x, 0, edge_index[0])
        return self.agg(temp, edge_index[1], dim = 0, out = x) if self_loop else self.agg(temp, edge_index[1], dim = 0) 

class GCNconv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, add_self_loops = True,):
        super(GCNconv, self).__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.mp = messagePassing("add")
        self.self_loops = add_self_loops
        
    def forward(self, x, edge_index):
        degrees = torch.sqrt(tg.utils.degree(edge_index[0]) +1 )
        x = x / degrees.view(-1, 1)
        x = self.mp(x, edge_index, self_loop = self.self_loops)
        x = x / degrees.view(-1, 1)
        x = self.lin(x)
        return x
    
class MLPReadout(torch.nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ torch.nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(torch.nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = torch.nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers, add_self_loops = True, reduce = "mean", n_iter = None, residual = True):
        super(GCN, self).__init__()
        self.embedding_layer = torch.nn.Embedding(1, in_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers-1):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(GCNconv(in_channels, hidden_channels, add_self_loops))
        self.convs.append(GCNconv(hidden_channels, out_channels, add_self_loops))

        # for now implementing sum reduction
        if reduce == "sum":
            self.reduce = partial(tc.scatter_add, dim=0)
        elif reduce == "mean":
            self.reduce = partial(tc.scatter_mean, dim=0)
        elif reduce == "max":
            self.reduce = partial(tc.scatter_max, dim=0)
        elif reduce == "set2set":
            self.reduce = set2setReadout(out_channels, n_iter)
        else:
            raise ValueError("Invalid value for reduce")
        self.mlp = MLPReadout(out_channels, 1) if reduce != "set2set" else MLPReadout(out_channels*2, 1)
        self.residual = residual

    def forward(self, x, edge_index, ptr=None):
        x = self.embedding_layer(x.view(-1))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index)) + x if self.residual else F.relu(conv(x, edge_index))
        x = self.reduce(x, ptr)
        x = self.mlp(x)
        return x
    
class GATconv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads = 1, add_self_loops = True, f_additive = True):
        super(GATconv, self).__init__()
        assert out_channels % heads == 0
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.att_src = torch.nn.Parameter(torch.empty(1, heads, out_channels//heads))
        self.att_dst = torch.nn.Parameter(torch.empty(1, heads, out_channels//heads))
        self.heads = heads
        self.self_loops = add_self_loops
        self.reset_parameters()
        self.f_additive = f_additive
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.att_src)
        torch.nn.init.xavier_uniform_(self.att_dst)

        
    def forward(self, x, edge_index):
        if self.self_loops:
            edge_index, _ = tg.utils.add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x).view(x.size(0), self.heads, -1)
        e_ij = self.prepareEdgeWeights(x, edge_index)
        y = self.propagate(x, edge_index, e_ij)
        return y.view(x.size(0), -1)
        
    def prepareEdgeWeights(self,x ,edge_index):
        alpha_dst = (x * self.att_dst).sum(dim=-1)
        alpha_src = (x * self.att_src).sum(dim=-1)
        alpha = torch.index_select(alpha_src, 0, edge_index[0]) + torch.index_select(alpha_dst, 0, edge_index[1])
        alpha = F.leaky_relu(alpha, 0.2)
        e_ij = tg.utils.softmax(alpha, edge_index[1])
        return e_ij
    def propagate(self, x, edge_index, e_ij):
        temp = torch.index_select(x, 0, edge_index[0])
        edge_weights = e_ij.unsqueeze(-1) + 1  if self.f_additive else e_ij.unsqueeze(-1) 
        temp = temp *  edge_weights
        return tc.scatter_add(temp, edge_index[1], dim=0)
    
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers, heads, add_self_loops = True, reduce = "mean", n_iter = None,residual = True, f_additive = True):
        super(GAT, self).__init__()
        self.embedding_layer = torch.nn.Embedding(1, in_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers-1):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(GATconv(in_channels, hidden_channels, heads, add_self_loops, f_additive))
        self.convs.append(GATconv(hidden_channels, out_channels, heads, add_self_loops, f_additive))
        # for now implementing sum reduction
        if reduce == "sum":
            self.reduce = partial(tc.scatter_add, dim=0)
        elif reduce == "mean":
            self.reduce = partial(tc.scatter_mean, dim=0)
        elif reduce == "max":
            self.reduce = partial(tc.scatter_max, dim=0)
        elif reduce == "set2set":
            self.reduce = set2setReadout(out_channels, n_iter)
        else:
            raise ValueError("Invalid value for reduce")

        self.mlp = MLPReadout(out_channels,1, 1) if reduce != "set2set" else MLPReadout(out_channels*2, 1)
        self.residual = residual    
        

    def forward(self, x, edge_index, ptr=None):
        x = self.embedding_layer(x.view(-1))
        for conv in self.convs:
            x = F.elu(conv(x, edge_index)) + x if self.residual else F.elu(conv(x, edge_index))
        x = self.reduce(x, ptr)
        x = F.relu(x)
        x = self.mlp(x)
        return x
    
class set2setReadout(torch.nn.Module):
    def __init__(self, input_dim, n_iters, n_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim * 2
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.LSTM_layers = torch.nn.LSTM(self.output_dim, input_dim, num_layers=n_layers)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.LSTM_layers.reset_parameters()

        
    def forward(self, x, ptr):
        batch_size = ptr.max().item() + 1
        q = x.new_zeros((batch_size, self.output_dim)).to(x.device)
        h = (x.new_zeros((self.n_layers, batch_size, self.input_dim)).to(x.device), x.new_zeros((self.n_layers, batch_size, self.input_dim)).to(x.device))
        for _ in range(self.n_iters):
            q, h = self.LSTM_layers(q.unsqueeze(0), h)
            q = q.view(batch_size, self.input_dim)
            e = (x * q[ptr]).sum(dim=-1, keepdim=True)
            alpha = tg.utils.softmax(e, ptr, dim=-2)
            r = tc.scatter_add(alpha * x, ptr, dim=-2)
            q = torch.cat([q, r], dim=-1)
        return q
