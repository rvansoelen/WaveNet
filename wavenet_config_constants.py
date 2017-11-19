# Logging
LogRoot = './log'


# Audio inputs
AudioDirectory = './data/wavenet_corpus'
BatchSize = 2 # files to process at once (threading)

# Training
CheckpointsToKeep = 5
LogDir = 'Train'
TrainIterations =  1e5
CheckpointFreq = 50


FilterWidth = 2
InitialFilterWidth = 32
SampleRate = 16000
LayerDialationFactors = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                         1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                         1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                         1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                         1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
ResidualFilters = 32
DialationFilters = 32
SkipFilters = 512
AmplitudeFilters = 256 #8bit
