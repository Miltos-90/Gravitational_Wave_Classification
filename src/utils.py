import numpy as np
import os
from scipy import signal
from scipy.fft import fft, ifft
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from itertools import chain
from sklearn.metrics import roc_auc_score
import torch
import os
import config



''' Computes AUROC from targets and model predictions'''
def computeAUROC(actual, predicted):
    
    yTrue = torch.tensor( list(chain(*actual)),    dtype = float).to('cpu').numpy().squeeze()
    yPred = torch.tensor( list(chain(*predicted)), dtype = float).sigmoid().to('cpu').numpy().squeeze()

    return roc_auc_score(yTrue, yPred)


''' Stratified split in a train / test set '''
def dataSplit(df, seed, testSize):
    
    lt_splitter  = StratifiedShuffleSplit(n_splits     = 1, 
                                          test_size    = testSize, 
                                          random_state = seed)

    l_idx, t_idx = list(lt_splitter.split(df['id'], df['target']))[0]
    train_df     = df.iloc[l_idx]
    test_df      = df.iloc[t_idx]

    return train_df, test_df


''' Makes a path from the ID of a sample .npy file '''
def _makePath(imgId, mainDir = config.RAW_DATA_DIR):
    
    pathStr = f"{mainDir}{imgId[0]}/{imgId[1]}/{imgId[2]}/{imgId}.npy"
    return pathStr


''' Loads an .npy file'''
def load(sampleID): 
    return np.load(_makePath(sampleID))


''' Whitens strain data given the psd and sample rate, also applying a phase
    shift and time shift.
'''
def whiten(data, psdNoise,
            fs = config.SAMPLE_FREQUENCY,
            phaseShift = 0, timeShift = 0):
    
    spec      = fft(data)
    noSamples = data.shape[-1]
    dc_len    = psdNoise.shape[-1]
    whitened  = np.real(ifft(spec[:, :dc_len] / psdNoise[1:, :], n = noSamples))
    whitened *= np.sqrt(noSamples / 2)
    whitened /= np.max(np.abs(whitened), axis = -1).reshape(3, 1)
    
    return whitened


''' Bandpasses strain data using a butterworth filter. '''
def bandpass(data, 
              fs    = config.SAMPLE_FREQUENCY, 
              order = config.BANDPASS_ORDER, 
              fband = config.BANDPASS_FREQS):
    
    # Butterworth filter
    fLow   = fband[0] * 2. / fs
    fHigh  = fband[1] * 2. / fs
    bb, ab = signal.butter(order, [fLow, fHigh], btype = 'band')
    
    # Apply filter and normalise
    normalization = np.sqrt(fHigh - fLow)
    strainBP      = signal.filtfilt(bb, ab, data) / normalization
    
    return strainBP


''' Downsamples signals from one data point'''
def downsample(data, fMax = config.BANDPASS_FREQS[1] * 2.56):
    
    secs  = float(data.shape[1]) / config.SAMPLE_FREQUENCY  # Number of seconds in signal X
    samps = int(secs * fMax)                                # Number of samples to downsample
    data  = signal.resample(data, num = samps, axis = 1)    # Resample

    return data


''' Estimates spectral noise by averaging FFT levels of all noise samples'''
def spectralNoiseEstimate(df):

    psd = 0
    for sampleID in tqdm(df["id"]):
        data   = utils.load(sampleID)
        f, Pxx = signal.periodogram(data, window = config.WINDOW,
                                    fs = config.SAMPLE_FREQUENCY) 
        psd += Pxx

    psd = psd ** 0.5
    psd /= dfNoise.shape[0]

    return np.vstack([f, psd]) # First row: frequencies, second, third, fourth: FFT levels for each observatory


''' Make a generic pytorch checkpoint '''
def makeCheckpoint(epoch, model, optimizer, loss, bestLoss, 
                   scheduler = None, 
                   scaler    = None, 
                   filePath  = config.CHECKPOINT, 
                   verbose   = True):
    
    # Check if improvement was found
    if loss < bestLoss:
        
        bestLoss = loss
        if verbose: 
            print('Validation loss improved. Making checkpoint...')
            
        if scheduler is not None: 
            schedulerDict = scheduler.state_dict()
        else:                     
            schedulerDict = None
            
        if scaler is not None:    
            scalerDict = scaler.state_dict()
        else:                     
            scalerDict = None
            
        torch.save({'epoch'                : epoch,
                    'model_state_dict'     : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'scheduler_state_dict' : schedulerDict,
                    'scaler_state_dict'    : scalerDict,
                    'best_validation_loss' : bestLoss,
                    }, filePath)
    
    return bestLoss


''' Load a genetic pytorch checkpoint '''
def loadCheckpoint(model, optimizer, 
                   scheduler   = None, 
                   scaler      = None, 
                   filePath    = config.CHECKPOINT, 
                   mapLocation = config.DEVICE, 
                   verbose     = True):
    
    if os.path.exists(filePath):

        if verbose: print('Loading checkpoint...')
        
        checkpoint = torch.load(filePath, map_location = mapLocation)
        
        epoch       = checkpoint['epoch'] + 1 # Get epoch to be run now
        bestValLoss = checkpoint['best_validation_loss']
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None:    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    else:
        if verbose: print('No checkpoint found.')
        epoch       = 0
        bestValLoss = np.inf
    
    return model, optimizer, scheduler, scaler, epoch, bestValLoss