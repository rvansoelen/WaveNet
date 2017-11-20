#This is where the wavenet model will be defined
import tensorflow as tf
import wavenet_config_constants

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
		self.audio_files = wavenet_config_constants.BatchSize
		self.layer_dialation_factors = wavenet_config_constants.LayerDialationFactors
		self.filter_width = wavenet_config_constants.FilterWidth
		self.residual_filters = wavenet_config_constants.ResidualFilters
		self.dialation_filters = wavenet_config_constants.DialationFilters
		self.skip_filters = wavenet_config_constants.SkipFilters
		self.amplitude_filters = wavenet_config_constants.AmplitudeFilters
		self.field = (self.filter_width - 1) * sum(self.layer_dialation_factors) + self.filter_width
		# convolution filter variable initializer
		self.conv_init = tf.contrib.layers.xavier_initializer_conv2d()
		self.inputDType = tf.uint8
		self.oneHotDetpth = 256 #8bit
		self.computationalDType = tf.float32

		layer_dictionary = dict()
		with tf.variable_scope('Wavenet'):
			# setup causal layer variables
			with tf.variable_scope('Causal Layer'):
				init_filters = self.amplitude_vals
				init_filter_width = self.filter_width
				layer_dictionary['Causal Layer']['Filter'] = tf.Variable(self.conv_init(init_filter_width,init_filters,self.residual_filters), 'filter')

			# setup dilated convolutional stack  variables
			layer_dictionary['Dialated Stack'] = list()
			with tf.variable_scope('Dialated Stack'):
				for layerNumber, dilation in enumerate(self.layer_dialation_factors):
					with tf.variable_scope('layer{}'.format(layerNumber)):
						currentLayer = dict()
						initializers = [self.filter_width, self.residual_filters, self.dialation_filters]
						currentLayer['Filter'] = tf.Variable(self.conv_init(shape=initializers), name='Filter')
						currentLayer['Gate'] = tf.Variable(self.conv_init(shape=initializers), name='Gate')
						currentLayer['Dense'] = tf.Variable(self.conv_init(shape=initializers), name='Dense')
						currentLayer['Skip'] = tf.Variable(self.conv_init(shape=initializers), name='Skip')
						layer_dictionary['Dialated Stack'].append(currentLayer)

			# setup post-procssing layer variables
			with tf.variable_scope('Post-Processing'):
				currentLayer = dict()
				currentLayer['Post-Process1'] = tf.Variable(self.conv_init(shape=[1, self.skip_filters, self.skip_filters]), name='Post-Process1')
				currentLayer['Post-Process2'] = tf.Variable(self.conv_init(shape=[1, self.skip_filters, self.amplitude_vals]), name='Post-Process2')
				layer_dictionary['Post-Processing'] = currentLayer


	def createCausalConv(self, input, dilation, name, numFilters=1):
		with tf.variable_scope(name):
			numInChannels = tf.shape(input)[1]
			w = tf.get_variable('weights', shape=(2, numInChannels, numFilters)) #shape: (filterWidth, numInputChannels, numOutputChannels)
			downSample = input[:, :, ::dilation]
			if tf.shape(downSample)[2]%2 != 0:
				downSample = tf.pad(downSample, [[0, 0], [0, 0], [1, 0]])
			conv = tf.conv1d(downSample, w, stride=1, data_format='NCHW')
			conv = tf.transpose(conv, perm=[0, 2, 1]) #THIS MAY NOT BE NEEDED. NEED TO CHECK THIS
			b = tf.get_variable('bias', shape=tf.shape(conv))
			convWithBias = tf.add(conv, b)
			return convWithBias

	def createDilatedLayer(self, input, name):
		with tf.variable_scope(name):
			filterConv = self.createCausalConv(input, dilation, name='Filter Causal Convolution')
			gateConv = self.createCausalConv(input, dilation, name='Gate Causal Convolution')
			actv = tf.tanh(filterConv)*tf.sigmoid(gateConv)
			skip = actv
			residual = tf.add(actv, input)
			return skip, residual

	#create and return model
	def getGenerativeModel(self):
		#TODO: try not using currentLayer variable to make it more clear where each layer is defined
		#input shape: (batchSize, , numChannels, receptiveField)
		input = tf.placeholder(self.inputDType, shape=(-1, 1, self.receptiveField), name='Raw Input') 

		#preprocessing layer (one hot encoding)
		oneHot = tf.one_hot(input, self.oneHotDepth, dtype=self.computationalDType, name='One Hot Encoding')
		#oneHot has the shape (batchSize, 1, receptiveField, oneHotDepth)

		#causal layer (not sure if this is needed)
		#currentLayer = self.createCausalConv(currentLayer) #"causalLayer" named scope is defined here?
		
		#dilated convolutional stack 
		outputs = []
		currentLayer = oneHot
		with tf.variable_scope('Dilated Stack'):
			for layerNumber, dilation in enumerate(self.layer_dialation_factors):
				with tf.named_scope('Dilated Layer '+str(layerNumber)):
					#represents skip(direct connection to last layer) and residual(normal) connections
					skipOutput, currentLayer = self.createDilatedLayer(currentLayer, dilation, name='Dilated Layer '+str(layerNumber))
					outputs.append(skipOutput)

		#post processing layer
		with tf.variable_scope('Post-Processing'):
			w1 = tf.get_variable('postWeights1', shapeW1)
			b1 = tf.get_variable('postBias1', shapeB1)
			w2 = tf.get_variable('postWeights1', shapeW2)
			b2 = tf.get_variable('postBias1', shapeB2)

			total = tf.nn.relu(sum(outputs))
			conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
			conv1 = tf.nn.relu(tf.add(conv1, b1))

     		conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
            conv2 = tf.add(conv2, b2)
        return conv2

    def getTrainingModel(self, input):
    	model = self.getGenerativeModel()
		with tf.named_scope('Loss'):
			#loss layer
			target = tf.placeholder(tf.float)
			loss = tf.softmax_cross_entropy_with_logits(logits=model, labels=target)

		return loss

