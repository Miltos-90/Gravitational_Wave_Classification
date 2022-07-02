from scipy.signal import tukey

# Directories
RAW_DATA_DIR = './data/train/'
NOISE_FILE   = './data/psdNoiseEstimate.txt'
CHECKPOINT   = './checkpoint.pt'
STATS_FILE   = './training_stats.csv'

# Data-related parameters
SAMPLE_FREQUENCY = 2048 # From description
SAMPLE_TIME      = 2    # From description

# Preprocessing parameters
BANDPASS_FREQS   = [30.0, 300.0]
BANDPASS_ORDER   = 4
TUKEY_ALPHA      = 0.2

# Train/test split
TEST_RATIO       = 0.15
VAL_RATIO        = 0.15

# Training parameters
DEVICE       = 'cuda'
NOISE_SCALE  = [3.5e2, 3.5e2, 3.5e2] # Scaling factors for the amplitude of the synthetic signals
SR_RATIO     = 0.4                   # Synthetic to real sample ratio for training
CLASS_RATIO  = 0.5                   # Synthetic signal generator class balance
BATCHSIZE    = 128
WORKERS      = 16
LEARN_RATE   = 5e-4
WEIGHT_DECAY = 1e-5
EPOCHS       = 26

# Window signal for FFT
WINDOW = tukey(SAMPLE_TIME * SAMPLE_FREQUENCY, alpha = TUKEY_ALPHA)