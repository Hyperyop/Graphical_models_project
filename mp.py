import torch 
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