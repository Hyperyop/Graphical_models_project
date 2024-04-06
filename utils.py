import torch.nn.functional as F
import torch
import time
from tqdm import tqdm
from torchmetrics.classification import BinaryAccuracy

# metric = lambda x,y: F.l1_loss(x,y).detach().item()

def train_epoch(model, loss_fn, optimizer,scheduler, device, data_loader, epoch, metric):
    model.train()
    epoch_loss = 0
    epoch_train_metric = 0
    nb_data = 0
    for iter, batch in enumerate(data_loader):
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        # edge_attr = batch.edge_attr.to(device)
        targets = batch.y.to(device)
        ptr = batch.batch.to(device)
        optimizer.zero_grad()
        batch_scores = model.forward(x, edge_index, ptr)
        loss = loss_fn(batch_scores, targets.view(-1, 1).float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_metric += metric(F.sigmoid(batch_scores), targets.view(-1, 1)).detach().item()
        nb_data += targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_metric /= (iter + 1)
    
    return epoch_loss, epoch_train_metric, optimizer
def evaluate_epoch(model, loss_fn, device, data_loader, metric):
    model.eval()
    epoch_loss = 0
    epoch_metric = 0
    nb_data = 0
    with torch.no_grad():
        for iter, batch in enumerate(data_loader):
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            # edge_attr = batch.edge_attr.to(device)
            targets = batch.y.to(device)
            ptr = batch.batch.to(device)
            batch_scores = model.forward(x, edge_index, ptr)
            loss = loss_fn(batch_scores, targets.view(-1, 1).float())
            epoch_loss += loss.detach().item()
            epoch_metric += metric(F.sigmoid(batch_scores), targets.view(-1, 1)).item()
            nb_data += targets.size(0)
        epoch_loss /= (iter + 1)
        epoch_metric /= (iter + 1)
    return epoch_loss, epoch_metric
def evaluate_model(model, loss_fn, device, data_loader):
    epoch_loss, epoch_metric = evaluate_epoch(model, loss_fn, device, data_loader)
    print(f'Loss: {epoch_loss}, metric: {epoch_metric}')
    return epoch_loss, epoch_metric

from collections import OrderedDict
def train(model, loss_fn, optimizer, scheduler, device, train_loader, val_loader, test_loader, epochs, min_lr):
    start = time.time()
    logs = { 'train_loss': [], 'val_loss': [], 'test_loss': [], 'train_metric': [], 'val_metric': [], 'test_metric': [], "time_per_epoch": []}
    metric =  BinaryAccuracy().to(device)
    with tqdm(range(epochs)) as pbar:
        for epoch in pbar:
            pbar.set_description(f'Epoch {epoch}')
            epoch_loss, epoch_train_metric, optimizer = train_epoch(model, loss_fn, optimizer, scheduler, device, train_loader, epoch, metric = metric)
            val_loss, val_metric = evaluate_epoch(model, loss_fn, device, val_loader, metric = metric)
            test_loss, test_metric = evaluate_epoch(model, loss_fn, device, test_loader, metric = metric)
            postfix_dict = OrderedDict(
                time = time.time() - start,
                lr=optimizer.param_groups[0]['lr'],
                train_loss=epoch_loss,
                val_loss=val_loss,
                test_loss=test_loss,
                train_metric=epoch_train_metric,
                val_metric=val_metric,
                test_metric=test_metric
            )
            pbar.set_postfix(postfix_dict)
            scheduler.step(val_loss)
            logs['train_loss'].append(epoch_loss)
            logs['val_loss'].append(val_loss)
            logs['test_loss'].append(test_loss)
            logs['train_metric'].append(epoch_train_metric)
            logs['val_metric'].append(val_metric)
            logs['test_metric'].append(test_metric)
            logs['time_per_epoch'].append(time.time() - start)
            if optimizer.param_groups[0]['lr'] <= min_lr:
                print('Early stopping')
                break
    return logs