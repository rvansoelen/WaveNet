#This file extracts the data of importance from the YouTube M8 dataset
import random, threading, librosa, os, fnmatch
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import wavenet_config_constants
import numpy as np


def getAudioData(directory, sampleRate):
    # find all audio files in director
    audioFiles = []
    for root, dir, file in os.walk(directory):
        for filename in fnmatch.filter(file, '*.wav'):
            audioFiles.append(os.path.join(root, filename))

    if not audioFiles:
        raise ValueError('No audio files found in {}'.format(directory))

    print('Loading {} audio file(s)'.format(len(audioFiles)))

    # load audio files as floating point time series
    rawAudio = []
    for filePath in audioFiles:
        audio, sr = librosa.load(filePath, sr=sampleRate, mono=True)
        audio = audio.reshape(-1,1)
        rawAudio.append(audio)
    return rawAudio


class ExtractData(object):
    def __init__(self):
        self.audioDirectory = wavenet_config_constants.AudioDirectory
        self.sampleRate = wavenet_config_constants.SampleRate
        self.threads = []
        queuesize = 32
        maxDataVal = tf.placeholder(dtype=tf.float32, shape= [])

        self.extractedAudio = getAudioData(self.audioDirectory, self.sampleRate)


    def getExtractedAudio(self):
        return self.extractedAudio


    def loadDataToSession(self, session):
        trainAudio = tf.constant(self.extractedAudio)
        trainData = Dataset.from_tensor_slices(trainAudio)

        audioIterator = Iterator.from_structure(trainData.output_types, trainData.output_shapes)
        nextEle = audioIterator.get_next()
        trainInitOp = audioIterator.make_initializer(trainData)
        session.run(trainInitOp)
        while True:
            try:
                elem = session.run(nextEle)
                print(elem)
            except tf.errors.OutOfRangeError:
                print("End of training dataset.")
                break
