#This is where the wavenet model will be defined
import tensorflow as tf

class WaveNet:
	'''Implements the WaveNet network for generative audio.
    
    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        net = WaveNetModel(batch_size, dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)
        loss = net.loss(input_batch)
    '''

	#initialize class
	def __init__(self):
		#TODO: Fill these variables
		self.dilations = None


	def createCausalLayer(self):


	def createDilatedLayer(self, input, name='Dilated Layer'):
		with tf.variable_scope(name):
			w = tf.createVariable('weights')
			b = tf.createVariable('bias')
			conv = tf.conv1d(input, w, stride=1, padding='SAME')
			conv = tf.add(conv, b)
			actv = tf.tanh(conv)*tf.sigmoid(conv)
			skip = actv
			residual = tf.add(actv, input)
			return skip, residual

	#create and return model
	def createModel(self, input):
		#TODO: try not using currentLayer variable to make it more clear where each layer is defined
		currentLayer = input

		#preprocessing layer

		#causal layer
		currentLayer = self.createCausalLayer(currentLayer) #"causalLayer" named scope is defined here?
		
		#dilated convolutional stack 
		outputs = []
		with tf.variable_scope('Dilated Stack'):
			for layerNumber, dilation in enumerate(self.dilations):
				with tf.named_scope('Dilated Layer '+str(layerNumber)):
					#represents skip(direct connection to last layer) and residual(normal) connections
					skipOutput, currentLayer = self.createDilatedLayer(currentLayer)
					outputs.append(skipOutput)

		#post processing layer
		with tf.variable_scope('Post-Processing'):
			w1 = createVariable('postWeights1', shapeW1)
			b1 = createVariable('postBias1', shapeB1)
			w2 = createVariable('postWeights1', shapeW2)
			b2 = createVariable('postBias1', shapeB2)

			total = tf.nn.relu(sum(outputs))
			conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
     		conv1 = tf.nn.relu(tf.add(conv1, b1))

     		conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
            conv2 = tf.add(conv2, b2)
        return conv2

    def wrapLossOptimizer(self, model):
		with tf.named_scope('Loss'):
			#loss layer
			target = tf.placeholder(tf.float)
			loss = tf.softmax_cross_entropy_with_logits(logits=model, labels=target)
		return loss

