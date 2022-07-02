import config
import torch


''' Training function '''
def train(model, criterion, metric, loader, optimizer, scaler, device = config.DEVICE, verbose = True):
    
    model.train()
    totalLoss = 0
    actual, predicted  = [], []
    
    for batchNo, (x, y) in enumerate(loader):

        optimizer.zero_grad(set_to_none = True)
        x = x.to(device)
        y = y.to(device)

        with torch.cuda.amp.autocast():
            yhat = model(x)
            loss = criterion(yhat, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        totalLoss += loss.item() * x.shape[0]
        actual.append(y)
        predicted.append(yhat.detach())

        if verbose and batchNo % 50 == 0:
            print(f'\t Training Batch {batchNo:3d} loss: {loss.item():.3f}')
    
    return totalLoss, metric(actual, predicted)


''' Validation function '''
def validate(model, criterion, metric, loader, device = config.DEVICE, verbose = True):
    
    model.eval()
    totalLoss = 0
    actual, predicted  = [], []
    
    with torch.no_grad():
        for batchNo, (x, y) in enumerate(loader):

            x    = x.to(device)
            y    = y.to(device)
            yhat = model(x)
            loss = criterion(yhat, y) 
            totalLoss += loss.item() * x.shape[0]
            actual.append(y)
            predicted.append(yhat.detach())

            if verbose and batchNo % 50 == 0:
                print(f'\t Validation Batch {batchNo:3d} loss: {loss.item():.3f}')
    
    return totalLoss, metric(actual, predicted)

