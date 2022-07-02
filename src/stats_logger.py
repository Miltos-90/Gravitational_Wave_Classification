import os
import csv

''' Holds, updates and saves/loads training statistics'''
class StatsLogger(object):
    
    ''' Initialisation '''
    def __init__(self, filename : str  = None): # filename: path to .csv file to load / save
        
        self.filename = filename
        
        # if filename exists load it 
        if self.filename is not None and os.path.exists(self.filename):
            self.load()
        
        else: # Initialise
            self.initialise()
    
    ''' Updates statistics '''
    def __call__(self,
                 trainLoss   : float = None, 
                 valLoss     : float = None, 
                 trainMetric : float = None, 
                 valMetric   : float = None,
                 verbose     : bool  = True):
        
        self.epoch += 1
        if trainLoss is not None:   self.trainLoss.append(trainLoss)
        if valLoss is not None:     self.valLoss.append(valLoss)
        if trainMetric is not None: self.trainMetric.append(trainMetric)
        if valMetric is not None:   self.valMetric.append(valMetric)
                
        if verbose:
            print(f'Epoch: {self.epoch:3d} | Train loss: {trainLoss:.3f} | Train metric: {trainMetric:.3f} | Val loss: {valLoss:.3f} | Val metric: {valMetric:.3f}')

        return
    
    ''' Saves statistics to a .csv file'''
    def save(self):
        
        headers = ['trainLoss', 'valLoss', 'trainMetric', 'valMetric']
        rows    = zip(self.trainLoss, self.valLoss, self.trainMetric, self.valMetric)

        with open(self.filename, 'a') as outcsv:
            writer = csv.writer(outcsv, delimiter = ',', quotechar = '|', quoting = csv.QUOTE_MINIMAL, lineterminator = '\n')
            writer.writerow(headers)

            for row in rows:
                writer.writerow(row)
                
        return 
    
    ''' Loads statistics from a .csv file '''
    def load(self):
        
        self.initialise()
        with open(self.filename, 'r') as rFile:

            csv_reader = csv.reader(rFile)
            for row in csv_reader:

                try: # Will fail for headers
                    trainLoss, valLoss, trainMetric, valMetric = [float(elem) for elem in row]
                except:
                    pass
                else:
                    self.trainLoss.append(trainLoss)
                    self.valLoss.append(valLoss)
                    self.trainMetric.append(trainMetric)
                    self.valMetric.append(valMetric)
                
        self.epoch = len(self.trainLoss)
        
        return 
    
    ''' Initialise properties if file is not found '''
    def initialise(self):
        
        self.trainLoss   = []
        self.valLoss     = []
        self.trainMetric = []
        self.valMetric   = []
        self.epoch       = 0
        
        return