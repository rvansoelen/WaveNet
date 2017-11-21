#This is where the model is trained
from __future__ import print_function

import time

import tensorflow as tf

import model
import wavenet_config_constants
from codeTooSimilarToGitHub import extractData


#train model
def main():
    # access and preprocess audio data
    audioExtractor = extractData()
    audioBatch = audioExtractor.getExtractedAudio()

    # create network
    trainNet = model.WaveNet()
    loss = trainNet.loss(audioExtractor)
    optimizer = tf.train.AdamOptimizer(learning_rate=wavenet_config_constants.LearningRate, epsilon=1e-4)
    optimal = optimizer.minimize(loss, var_list=tf.trainable_variables())

    # session
    session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    startSession = tf.global_variables_initializer()
    session.run(startSession)

    checkpointHandler = checkpointHandler.CheckpointHandler()
    #TODO add option to restore from previous checkpoint
    audioExtractor.startThreads(session)

    # step through and train network
    iteration = None
    lastSavedIteration = None
    try:
        for iteration in range(wavenet_config_constants.TrainIterations):
            startTime = time.time()
            fetch = [summary, loss, optimal]
            summary, lossValue, opt = session.run(fetch)
            # store metadata every 100th checkpoint
            if iteration % wavenet_config_constants.CheckpointFreq == 0:
                print('Storing training metadata')
                checkpointHandler.logData(summary, iteration)

            runTime = time.time() - startTime
            print('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'.format(iteration, lossValue, runTime))
            # save checkpoint at specified interval
            if iteration % wavenet_config_constants.CheckpointFreq == 0:
                checkpointHandler.saveModel(session, wavenet_config_constants.LogTrainDir, iteration)
                lastSavedIteration = iteration
    except KeyboardInterrupt:
        print()
    finally:
        if iteration > lastSavedIteration:
            checkpointHandler.saveModel(session, wavenet_config_constants.LogTrainDir, iteration)


if __name__ == '__main__':
    main()