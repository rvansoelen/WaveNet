#This is where the model is trained
from __future__ import print_function

import os
import time
from datetime import datetime

import tensorflow as tf
from tensorflow.python.client import timeline

import model
import wavenet_config_constants
from codeTooSimilarToGitHub import extractData

BATCH_SIZE = 1
CHECKPOINT_EVERY = 50
NUM_STEPS = int(1e5)
LEARNING_RATE = 1e-3
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = 0.3
EPSILON = 0.001
MOMENTUM = 0.9
METADATA = False



#train model
def main():
    # coordinator
    coordinator = tf.train.Coordinator()

    # access and preprocess audio data
    with tf.name_scope('Create Inputs'):
        #TODO add silence threshold
        audioExtractor = extractData(coordinator)
        audioBatch = audioExtractor.dequeue(wavenet_config_constants.BatchSize)

    # create network
    trainNet = model.WaveNet()
    loss = trainNet.loss(audioBatch)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,epsilon=1e-4)
    trainable = tf.trainable_variables()
    optimal = optimizer.minimize(loss, var_list=trainable)

    # log
    logdir = os.path.join(wavenet_config_constants.LogRoot, wavenet_config_constants.LogDir, STARTED_DATESTRING)
    fileWritter = tf.summary.FileWriter(logdir)
    fileWritter.add_graph(tf.get_default_graph())
    metaData = tf.RunMetadata()
    results = tf.summary.merge_all()

    # session
    session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    startSession = tf.global_variables_initializer()
    session.run(startSession)

    checkpointSaver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=wavenet_config_constants.CheckpointsToKeep)
    #TODO add option to restore from previous checkpoint
    threads = tf.train.start_queue_runners(sess=session, coord=coordinator)
    audioExtractor.startThreads(session)

    # step through and train network
    iteration = None
    lastSavedIteration = None
    try:
        for iteration in range(wavenet_config_constants.TrainIterations):
            startTime = time.time()
            fetch = [summary, loss, optimal]
            # store metadata every 100th checkpoint
            if iteration % wavenet_config_constants.CheckpointFreq == 0:
                print('Storing training metadata')
                runOpts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # run session with specified loss and optimizer
                summary, lossValue, opt = session.run(fetch, options=runOpts, run_metadata=metaData)
                # log
                fileWritter.add_summary(summary, iteration)
                fileWritter.add_run_metadata(metaData, 'step_{:04d}'.format(iteration))
                timeLine = timeline.Timeline(metaData.step_stats)
                with open(os.path.join(logdir, 'timeline.trace')) as file:
                    file.write(timeLine.generate_chrome_trace_format(show_memory=True))
            # simplify run if not logging
            else:
                summary, lossValue, opt = session.run(fetch)
                fileWritter.add_summary(summary, iteration)
            runTime = time.time() - startTime
            print('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'.format(iteration, lossValue, runTime))
            # save checkpoint at specified interval
            if iteration % wavenet_config_constants.CheckpointFreq == 0:
                saveModel(checkpointSaver, session, logdir, iteration)
                lastSavedIteration = iteration
    except KeyboardInterrupt:
        print()
    finally:
        if iteration > lastSavedIteration:
            saveModel(checkpointSaver, session, logdir, iteration)
        coordinator.request_stop()
        coordinator.join(threads)



# periodically save model and test audio generation
def saveModel(modelCheckpoint, session, logDirectory, step):
    saveFile = logDirectory + 'model.ckpt'
    print('Saving checkpoint to ...'.format(logDirectory),end='')

    if not os.path.exists(logDirectory):
        os.makedirs(logDirectory)

    modelCheckpoint.save(session, saveFile, global_step=step)
    print('Done')

def loadModel(modelCheckpoint, session, logDirectory):
    print('Loading checkpoint from {} ...'.format(logDirectory),end='')
    checkpoint = tf.train.get_checkpoint_state(logDirectory)
    if checkpoint:
        print('Discovered checkpoint: {}'.format(checkpoint.model_checkpoint_path))
        global_step = int(checkpoint.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print('Global step of checkpoint: {}'.format(global_step))
        modelCheckpoint.restore(session, checkpoint.model_checkpoint_path)
        print('Done')
        return global_step
    else:
        return None



if __name__ == '__main__':
    main()