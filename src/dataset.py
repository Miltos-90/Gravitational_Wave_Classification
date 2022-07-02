import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config
import utils
from signal_generator import SignalGenerator


''' Makes a dataset and the corresponding dataloader'''
def makeDataloader(df, 
                   transform  = None, 
                   batchsize  = config.BATCHSIZE, 
                   workers    = config.WORKERS,
                   pin_memory = True,
                   shuffle    = True):
    
    dset = G2dataset(fileList   = df['id'].values,
                     targetList = df['target'].values,
                     transform  = transform)
    
    loader = DataLoader(
                dataset     = dset,
                batch_size  = batchsize,
                num_workers = workers,
                pin_memory  = pin_memory,
                shuffle     = shuffle)
    
    return loader


''' Dataset class'''
class G2dataset(Dataset):
    
    def __init__(self, fileList, targetList, 
                 transform  = None,
                 srRatio    = config.SR_RATIO,
                 classRatio = config.CLASS_RATIO,
                 window     = config.WINDOW,
                 noiseFile  = config.NOISE_FILE):
    
        self.files     = fileList
        self.labels    = targetList
        self.window    = window
        self.transform = transform
        self.srRatio   = srRatio
        self.psdNoise  = np.loadtxt(noiseFile, delimiter = ',')
        self.signalGen = SignalGenerator(classRatio)
    
    
    def __len__(self):
        return len(self.labels)

    
    ''' Fetch sample '''
    def __getitem__(self, idx):

        x, y = self.drawSample(idx)
        x = self.preprocess(x)
        
        if self.transform is not None: 
            x = self.transform(x)
        
        y = torch.tensor([y]).float()
        return x, y
    
    
    ''' Randomly draws a synthetic or real sample '''
    def drawSample(self, idx):
        
        if self.drawReal(): 
            x = utils.load(self.files[idx])
            y = self.labels[idx]
        else: 
            x, y = self.signalGen.generate()
        
        return x, y
    
    
    ''' Boolean indicating if a real or synthetic sample should be returned ''' 
    def drawReal(self): return np.random.rand() >= self.srRatio
    
    
    ''' Preprocessing pipeline '''
    def preprocess(self, x):
        
        x = x * self.window
        x = utils.whiten(x, self.psdNoise)
        x = utils.bandpass(x)
        x = x[:, 256:-256]
        x = utils.downsample(x)
        x = torch.tensor(x).float()
        
        return x


