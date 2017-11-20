#This file extracts the data of importance from the YouTube M8 dataset
import random, threading, librosa, os, fnmatch
import tensorflow as tf
import wavenet_config_constants
import model
import numpy as np


def collectAudioFiles(directory, sampleRate):
    audioFiles = []
    for root, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(root, '*.wav'):
            audioFiles.append(os.path.join(root, filename))

    if not audioFiles:
        raise ValueError('No audio files found in {}'.format(directory))

    print('Found {} audio files'.format(len(audioFiles)))
    for filePath in audioFiles:
        audio, sr = librosa.load(filePath, sr=sampleRate, mono=True)
        audio = audio.reshape(-1,1)
        yield audio, filePath


class ExtractData(object):
    def __init__(self, coordinator):
        self.audioDirectory = wavenet_config_constants.AudioDirectory
        self.coordinator = coordinator
        self.sampleRate = wavenet_config_constants.SampleRate
        self.receptiveField = model.calculateReceptiveField(wavenet_config_constants.FilterWidth, wavenet_config_constants.DialationFilters, wavenet_config_constants.InitialFilterWidth)
        self.threads = []
        queuesize = 32
        self.placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue =tf.PaddingFIFOQueue(queuesize, ['float32'], shapes=[(None, 1)])
        self.enqueue = self.queue.enqueue([self.placeholder])

        self.extractedAudio = collectAudioFiles(self.audioDirectory, self.sampleRate)



    def dequeue(self, num):
        return self.queue.dequeue_many(num)

    def createThread(self, session):
        finished = False
        while not finished:
            audioIterator = self.extractedAudio
            for audio, filePath in audioIterator:
                if self.coordinator.should_stop():
                    finished = True
                    break
                rawAudio = np.pad(audio, [[self.receptiveField, 0], [0,0]], 'constant')
                session.run(self.enqueue, feed_dict={self.placeholder: rawAudio})


    def startThreads(self, session):
        thread = threading.Thread(target=self.createThread, args=(session,))
        thread.daemon = True
        thread.start()
        self.threads.append(thread)
        return self.threads


#open the dataset files

#keep only videos with the specified labels (piano, for example)

#get the audio

#possibly downsample audio

#save it as a new tensorflow record file