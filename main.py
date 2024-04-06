import torch
import torch_geometric as tg
import torch.nn.functional as F
import torch_scatter as tc
from functools import partial
dataset = tg.datasets.ZINC('./data/ZINC', subset=True)
A = next(iter(dataset))
x = A.x
edge_index = A.edge_index
edge_attr = A.edge_attr

from mp import messagePassing 
layer = messagePassing("add")
edge_index.shape
x.shape
from torch_geometric.data import DataLoader

# Define the batch size
batch_size = 10

# Create loaders for train, validation, and test sets
train_loader = DataLoader(dataset[:8000], batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset[8000:9000], batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset[9000:], batch_size=batch_size, shuffle=False)

test = next(iter(train_loader))
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
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers, add_self_loops = True, reduce = "mean"):
        super(GCN, self).__init__()
        self.embedding_layer = torch.nn.Embedding(28, in_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers-1):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(GCNconv(in_channels, hidden_channels, add_self_loops))
        self.convs.append(GCNconv(hidden_channels, out_channels, add_self_loops))

        # for now implementing sum reduction
        match reduce:
            case "sum":
                self.reduce = partial(tc.scatter_add, dim = 0)
            case "mean":
                self.reduce = partial(tc.scatter_mean, dim = 0)
            case "max":
                self.reduce = partial(tc.scatter_max, dim = 0)
            case _:
                raise ValueError("Invalid value for reduce")
        self.mlp = MLPReadout(out_channels, 1)

    def forward(self, x, edge_index, ptr=None):
        x = self.embedding_layer(x.view(-1))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.reduce(x, ptr)
        x = self.mlp(x)
        return x

x = test.x
edge_index = test.edge_index
ptr = test.batch
torch.concat((x,x), dim = 1).shape
test_conv = GCNconv(2,10, add_self_loops = True)
a = test_conv(torch.concat((x,x), dim=1), edge_index)
a.shape
x.shape
model = GCN(28, 1, 64, 2)
test_model = model(x, edge_index, ptr)
test_model.shape
seed=41; epochs=1000; batch_size=5; init_lr=5e-5; lr_reduce_factor=0.5; lr_schedule_patience=25; min_lr = 1e-6; weight_decay=0
L=4; hidden_dim=145; out_dim=hidden_dim; dropout=0.0; readout='mean'

def train_epoch(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    for iter, batch in enumerate(data_loader):
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        # edge_attr = batch.edge_attr.to(device)
        targets = batch.y.to(device)
        ptr = batch.batch.to(device)
        optimizer.zero_grad()
        batch_scores = model.forward(x, edge_index, ptr)
        loss = F.l1_loss(batch_scores, targets.view(-1, 1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        nb_data += targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    
    return epoch_loss, epoch_train_mae, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(in_channels=hidden_dim, hidden_channels=hidden_dim, out_channels=out_dim, num_layers=L, reduce=readout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_reduce_factor, patience=lr_schedule_patience, min_lr=min_lr,verbose=True)
epoch_loss, epoch_train_mae, optimizer = train_epoch(model, optimizer, device, train_loader, 0)
from tqdm import tqdm
# for epoch in tqdm(range(epochs)):
#     epoch_loss, epoch_train_mae, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
#     print(f'Epoch {epoch}, Loss: {epoch_loss}, MAE: {epoch_train_mae}')
#     scheduler.step(epoch_loss)
#     if optimizer.param_groups[0]['lr'] <= min_lr:
#         print('Early stopping')
#         break
import torch.profiler

def train_with_profiler(model, optimizer, device, data_loader, epochs):
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        with_stack=True
    ) as prof:
        for idx, batch in enumerate(data_loader):
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_attr.to(device)
            targets = batch.y.to(device)
            ptr = batch.batch.to(device)
            optimizer.zero_grad()
            batch_scores = model.forward(x, edge_index, ptr)
            loss = F.l1_loss(batch_scores, targets.view(-1, 1))
            loss.backward()
            optimizer.step()
            prof.step()
            if idx >= 1 + 1 + 3:
                break
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(in_channels=hidden_dim, hidden_channels=hidden_dim, out_channels=out_dim, num_layers=L, reduce=readout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_reduce_factor, patience=lr_schedule_patience, min_lr=min_lr,verbose=True)

train_with_profiler(model, optimizer, device, train_loader, 4)