import tensorflow as tf
import wavenet_config_constants
import os
from datetime import datetime

STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())


class CheckpointHandler:
    def __init__(self):
        self.checkpointSaver = tf.train.Saver(var_list=tf.trainable_variables(),
                                         max_to_keep=wavenet_config_constants.CheckpointsToKeep)
        # log
        logdir = os.path.join(wavenet_config_constants.LogRoot, wavenet_config_constants.LogDir, STARTED_DATESTRING)
        self.fileWritter = tf.summary.FileWriter(logdir)
        self.fileWritter.add_graph(tf.get_default_graph())
        # metaData = tf.RunMetadata()
        # results = tf.summary.merge_all()


    # periodically save model and test audio generation
    def saveModel(self, session, saveDirectory, step):
        saveFile = saveDirectory + 'model.ckpt'
        print('Saving checkpoint to ...'.format(saveDirectory),end='')

        if not os.path.exists(saveDirectory):
            os.makedirs(saveDirectory)

        self.checkpointSaver.save(session, saveFile, global_step=step)
        print('Done')

    def loadModel(self, session, saveDirectory):
        print('Loading checkpoint from {} ...'.format(saveDirectory),end='')
        checkpoint = tf.train.get_checkpoint_state(saveDirectory)
        if checkpoint:
            print('Discovered checkpoint: {}'.format(checkpoint.model_checkpoint_path))
            global_step = int(checkpoint.model_checkpoint_path.split('/')[-1].split('-')[-1])
            print('Global step of checkpoint: {}'.format(global_step))
            self.checkpointSaver.restore(session, checkpoint.model_checkpoint_path)
            print('Done')
            return global_step
        else:
            return None


    def logData(self, summary, step):
        self.fileWritter.add_summary(summary, step)
        # TODO do we need this?
        # self.fileWritter.add_run_metadata(self.metaData, 'step_{:04d}'.format(step))
        # timeLine = timeline.Timeline(metaData.step_stats)
        # with open(os.path.join(logdir, 'timeline.trace')) as file:
        #     file.write(timeLine.generate_chrome_trace_format(show_memory=True))