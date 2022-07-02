import config

from scipy.interpolate import interp1d
import pycbc.noise.reproduceable as pycbc_noise
from pycbc.types.frequencyseries import FrequencySeries
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
import numpy as np
import abc


''' Abstract class for the WaveformGenerator, NoiseGenerator 
    and SignalGenerator classes
'''
class AbstractGenerator(metaclass = abc.ABCMeta):
    
    @abc.abstractmethod
    def generate():
        return


''' Randomly generates a positive/negative signal according to 
    a probability ratio.
'''
class SignalGenerator(AbstractGenerator):
   
    def __init__(self, classRatio = config.CLASS_RATIO): # Positive to negative ratio
        
        self.waveGen  = WaveformGenerator()
        self.noiseGen = NoiseGenerator()
        self.p        = classRatio
        return
        
    
    ''' Synthetic signal generator'''
    def generate(self):
        
        signal = self.noiseGen.generate()
        label  = 0 # Signal contains only noise up to this point
        
        if np.random.rand() >= self.p: # Add collision signal
            signal += self.waveGen.generate()
            label = 1
            
        return signal, label



''' Generator of synthetic black hole collisions '''
class WaveformGenerator(AbstractGenerator):
    
    def __init__(self, 
                 approxList   = None,                                         # List of approximants to be used (instead of the default below)
                 signalLength = config.SAMPLE_FREQUENCY * config.SAMPLE_TIME, # Required signal length
                 timeStep     = np.round((1.0 / config.SAMPLE_FREQUENCY), 4)  # Sampling timestep for the signal
                ):
        
        if approxList is None:
            self.approxList = [
                'SEOBNRv3_pert', 'SEOBNRv3_opt', 'SEOBNRv3_opt_rk4', 
                'SpinTaylorT1',  'SpinTaylorT5', 'SEOBNRv2',     'SEOBNRv3', 
                'SEOBNRv2T',     'SEOBNRv4T',    'IMRPhenomB',   
                'SEOBNRv2_opt',  'SpinTaylorT4', 'SEOBNRv4P',
                'IMRPhenomPv3',  'IMRPhenomXP',  'TEOBResumS',   'IMRPhenomXAS',
                'IMRPhenomT',    'IMRPhenomTHM', 'IMRPhenomTP',  'IMRPhenomTPHM']
        else:
            self.approxList = approxList
        
        self.detectors    = [Detector('H1'), Detector('L1'), Detector('V1')]
        self.noDetectors  = len(self.detectors)
        self.timeStep     = timeStep
        self.signalLength = signalLength
        
        return
    
    
    ''' Generates a waveform from a randomly sampled parameter space.
        Wraps the _makeWaveform function that some times fails due to 
        inappropriate (randomly chosen) combination of parameters
    '''
    def generate(self):
        
        flag = True
        
        while flag:
            try:
                sig = self._makeWaveform()
                flag = False
            except:
                pass
    
        return sig
     
    
    ''' Generates a waveform from a randomly sampled parameter space '''
    def _makeWaveform(self):
        
        # Signal duration: 2 seconds exactly
        startTime = np.random.randint(0, 40)
        endTime   = startTime + 120
        
        hplus, hcross   = get_td_waveform(
            start_time  = startTime,
            delta_t     = self.timeStep,
            approximant = np.random.choice(self.approxList),
            mass1       = np.random.randint(10, 20),
            mass2       = np.random.randint(10, 20),
            spin1z      = np.random.uniform(0.2, 0.9),
            spin2z      = np.random.uniform(0.2, 0.9),
            inclination = np.random.uniform(0, 2 * np.pi),
            coa_phase   = np.random.uniform(0, 2 * np.pi),
            f_lower     = np.random.randint(25, 40),
            f_higher    = np.random.randint(200, 1000),
            distance    = np.random.randint(1e1, 1e5))

        hplus.start_time += endTime
        hcross.start_time += endTime

        hplus.resize(self.signalLength)
        hcross.resize(self.signalLength)
        
        # Randomly sampled parameters
        declination     = np.random.uniform(- np.pi / 2, np.pi / 2)
        right_ascension = np.random.uniform(0, 2 * np.pi)
        polarization    = np.random.uniform(0, 2 * np.pi)
        
        # Project waveform to all detectors
        signals = np.empty(shape = (self.noDetectors, self.signalLength), dtype = float)
        
        for idx, detector in enumerate(self.detectors):
            sig = detector.project_wave(hplus, hcross, right_ascension, declination, polarization)
            sig = self.padToSize(sig.data)
            signals[idx, :] = sig
        
        return signals
    
    
    ''' Pads a signal to the appropriate size'''
    def padToSize(self, sig):
    
        actualSize = sig.shape[0]
        extraSize  = self.signalLength - actualSize
        padLeft    = - extraSize // 2
        padRight   = extraSize + padLeft
    
        return sig[padLeft:padRight]


''' Generator of synthetic noise signals '''
class NoiseGenerator(AbstractGenerator):
    
    def __init__(self, 
                 designCurveFile = config.NOISE_FILE,         # Configuration (.txt) file containing the design curves for each detector 
                 sampleFrequency = config.SAMPLE_FREQUENCY,   # Sampling Frequency [sec] for the signal
                 signalDuration  = config.SAMPLE_TIME,        # Total duration [sec] of the signal
                 scaleFactors    = config.NOISE_SCALE,        # Scaling factor to generate noise of the right amplitude (~1e-21))
                 deltaF          = 1.0):                      # Equidistant steps for the frequency [Hz] interpolation of a given PSD
        
        self.scaleFactors    = scaleFactors
        self.frequencyStep   = deltaF
        self.sampleFrequency = sampleFrequency
        self.signalDuration  = signalDuration 
        self._makeDesignCurves(designCurveFile)
        
        return
    
    
    ''' Loads and interpolates the design curves of the detectors '''
    def _makeDesignCurves(self, designCurveFile):

        designCurves = np.loadtxt(designCurveFile, delimiter = ',')     # Load file with the design curves
        curvInterObj = interp1d(designCurves[0,:], designCurves[1:,:])  # Interpolant for the estimated noise PSD
        
        fmin = designCurves[0,:].min()
        fmax = designCurves[0,:].max()
        fNew = np.arange(fmin, fmax, self.frequencyStep)        # Make equidistant frequencies and corresponding PSD levels
        cInt = curvInterObj(fNew) ** 2.0                        # Interpolate on the new frequencies
        
        # Make design curve for each detector
        self.designCurves = []
        for detector in range(cInt.shape[0]):
            
            fSeries = FrequencySeries(initial_array = cInt[detector, :], delta_f = self.frequencyStep)
            self.designCurves.append(fSeries)
        
        return
        
    
    ''' Generates a waveform from a randomly sampled parameter space '''
    def generate(self):
        
        # Empty array to hold resulting signals
        sh = (len(self.designCurves), self.signalDuration * self.sampleFrequency)
        noiseSignal = np.empty(shape = sh, dtype = float)
        
        for detectorID in range(len(self.designCurves)):

            # Generate 2.0 seconds of noise at the current sampling frequency
            ts = pycbc_noise.colored_noise(
                    psd         = self.designCurves[detectorID],
                    end_time    = self.signalDuration,
                    sample_rate = self.sampleFrequency,
                    seed        = np.random.randint(0, 1e9),
                    start_time  = 0)
            
            # Get the noise signal from the timeSeries
            noiseSignal[detectorID, :] = ts.data * self.scaleFactors[detectorID]
    
        return noiseSignal
