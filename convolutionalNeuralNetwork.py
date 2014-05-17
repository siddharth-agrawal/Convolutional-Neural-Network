# This piece of software is bound by The MIT License (MIT)
# Copyright (c) 2014 Siddharth Agrawal
# Code written by : Siddharth Agrawal
# Email ID : siddharth.950@gmail.com

import numpy
import math
import time
import scipy.io
import scipy.signal
import scipy.optimize
import matplotlib.pyplot

###########################################################################################
" The Convolutional Neural Network class """

class ConvolutionalNeuralNetwork(object):

    #######################################################################################
    """ Initialization of the network """

    def __init__(self, W1, b1, zca_white, mean_patch, patch_dim, pool_dim):
    
        """ Store the weights, taking into account preprocessing done """
    
        self.W = numpy.dot(W1, zca_white)
        self.b = b1 - numpy.dot(self.W, mean_patch)
        
        """ Variables associated with the network """
        
        self.patch_dim = patch_dim
        self.pool_dim  = pool_dim

    #######################################################################################
    """ Returns elementwise sigmoid output of input array """
    
    def sigmoid(self, x):
    
        return (1 / (1 + numpy.exp(-x)))
        
    #######################################################################################
    """ Returns the convolved features of the input images """
    
    def convolve(self, input_images, num_features):
    
        """ Extract useful values """
    
        image_dim      = input_images.shape[0]
        image_channels = input_images.shape[2]
        num_images     = input_images.shape[3]
        
        """ Assign memory for the convolved features """
        
        conv_dim           = image_dim - self.patch_dim + 1
        convolved_features = numpy.zeros((num_features, num_images, conv_dim, conv_dim));
        
        for image_num in range(num_images):
        
            for feature_num in range(num_features):
            
                """ Initialize convolved image as array of zeros """
            
                convolved_image = numpy.zeros((conv_dim, conv_dim))
                
                for channel in range(image_channels):
                
                    """ Extract feature corresponding to the indices """
                
                    limit0  = self.patch_dim * self.patch_dim * channel
                    limit1  = limit0 + self.patch_dim * self.patch_dim
                    feature = self.W[feature_num, limit0 : limit1].reshape(self.patch_dim, self.patch_dim)
                    
                    """ Image to be convolved """
                    
                    image = input_images[:, :, channel, image_num]
                    
                    """ Convolve image with the feature and add to existing matrix """

                    convolved_image = convolved_image + scipy.signal.convolve2d(image, feature, 'valid');
                
                """ Take sigmoid transform and store """
                    
                convolved_image = self.sigmoid(convolved_image + self.b[feature_num, 0])
                convolved_features[feature_num, image_num, :, :] = convolved_image
                
        return convolved_features
        
    #######################################################################################
    """ Pools the given convolved features """
    
    def pool(self, convolved_features):
    
        """ Extract useful values """
    
        num_features = convolved_features.shape[0]
        num_images   = convolved_features.shape[1]
        conv_dim     = convolved_features.shape[2]
        res_dim      = conv_dim / self.pool_dim
        
        """ Initialize pooled features as array of zeros """
        
        pooled_features = numpy.zeros((num_features, num_images, res_dim, res_dim))
        
        for image_num in range(num_images):
        
            for feature_num in range(num_features):
            
                for pool_row in range(res_dim):
                
                    row_start = pool_row * self.pool_dim
                    row_end   = row_start + self.pool_dim
                    
                    for pool_col in range(res_dim):
                    
                        col_start = pool_col * self.pool_dim
                        col_end   = col_start + self.pool_dim
                        
                        """ Extract image patch and calculate mean pool """
                        
                        patch = convolved_features[feature_num, image_num, row_start : row_end,
                                                   col_start : col_end]
                        pooled_features[feature_num, image_num, pool_row, pool_col] = numpy.mean(patch)
                        
        return pooled_features
        
###########################################################################################
""" The Softmax Regression class """

class SoftmaxRegression(object):

    #######################################################################################
    """ Initialization of Regressor object """

    def __init__(self, input_size, num_classes, lamda):
    
        """ Initialize parameters of the Regressor object """
    
        self.input_size  = input_size  # input vector size
        self.num_classes = num_classes # number of classes
        self.lamda       = lamda       # weight decay parameter
        
        """ Randomly initialize the class weights """
        
        rand = numpy.random.RandomState(int(time.time()))
        
        self.theta = 0.005 * numpy.asarray(rand.normal(size = (num_classes*input_size, 1)))
    
    #######################################################################################
    """ Returns the groundtruth matrix for a set of labels """
        
    def getGroundTruth(self, labels):
    
        """ Prepare data needed to construct groundtruth matrix """
    
        labels = numpy.array(labels).flatten()
        data   = numpy.ones(len(labels))
        indptr = numpy.arange(len(labels)+1)
        
        """ Compute the groundtruth matrix and return """
        
        ground_truth = scipy.sparse.csr_matrix((data, labels, indptr))
        ground_truth = numpy.transpose(ground_truth.todense())
        
        return ground_truth
        
    #######################################################################################
    """ Returns the cost and gradient of 'theta' at a particular 'theta' """
        
    def softmaxCost(self, theta, input, labels):
    
        """ Compute the groundtruth matrix """
    
        ground_truth = self.getGroundTruth(labels)
        
        """ Reshape 'theta' for ease of computation """
        
        theta = theta.reshape(self.num_classes, self.input_size)
        
        """ Compute the class probabilities for each example """
        
        theta_x       = numpy.dot(theta, input)
        hypothesis    = numpy.exp(theta_x)      
        probabilities = hypothesis / numpy.sum(hypothesis, axis = 0)
        
        """ Compute the traditional cost term """
        
        cost_examples    = numpy.multiply(ground_truth, numpy.log(probabilities))
        traditional_cost = -(numpy.sum(cost_examples) / input.shape[1])
        
        """ Compute the weight decay term """
        
        theta_squared = numpy.multiply(theta, theta)
        weight_decay  = 0.5 * self.lamda * numpy.sum(theta_squared)
        
        """ Add both terms to get the cost """
        
        cost = traditional_cost + weight_decay
        
        """ Compute and unroll 'theta' gradient """
        
        theta_grad = -numpy.dot(ground_truth - probabilities, numpy.transpose(input))
        theta_grad = theta_grad / input.shape[1] + self.lamda * theta
        theta_grad = numpy.array(theta_grad)
        theta_grad = theta_grad.flatten()
        
        return [cost, theta_grad]
    
    #######################################################################################
    """ Returns predicted classes for a set of inputs """
            
    def softmaxPredict(self, theta, input):
    
        """ Reshape 'theta' for ease of computation """
    
        theta = theta.reshape(self.num_classes, self.input_size)
        
        """ Compute the class probabilities for each example """
        
        theta_x       = numpy.dot(theta, input)
        hypothesis    = numpy.exp(theta_x)      
        probabilities = hypothesis / numpy.sum(hypothesis, axis = 0)
        
        """ Give the predictions based on probability values """
        
        predictions = numpy.zeros((input.shape[1], 1))
        predictions[:, 0] = numpy.argmax(probabilities, axis = 0)
        
        return predictions

###########################################################################################
""" Loads the training images and labels """
    
def loadTrainingDataset():

    """ Loads the images and labels as numpy arrays
        The dataset is originally read as a dictionary """

    train_data   = scipy.io.loadmat('stlTrainSubset.mat')
    train_images = numpy.array(train_data['trainImages'])
    train_labels = numpy.array(train_data['trainLabels'])
    
    return [train_images, train_labels]
    
###########################################################################################
""" Loads the test images and labels """
    
def loadTestDataset():

    """ Loads the images and labels as numpy arrays
        The dataset is originally read as a dictionary """

    test_data   = scipy.io.loadmat('stlTestSubset.mat')
    test_images = numpy.array(test_data['testImages'])
    test_labels = numpy.array(test_data['testLabels'])
    
    return [test_images, test_labels]

###########################################################################################
""" Visualizes the obtained optimal W1 values as images """

def visualizeW1(opt_W1, vis_patch_side, hid_patch_side):

    """ Add the weights as a matrix of images """
    
    figure, axes = matplotlib.pyplot.subplots(nrows = hid_patch_side,
                                              ncols = hid_patch_side)
    
    """ Rescale the values from [-1, 1] to [0, 1] """
    
    opt_W1 = (opt_W1 + 1) / 2
    
    """ Define useful values """
    
    index  = 0
    limit0 = 0
    limit1 = limit0 + vis_patch_side * vis_patch_side
    limit2 = limit1 + vis_patch_side * vis_patch_side
    limit3 = limit2 + vis_patch_side * vis_patch_side
                                              
    for axis in axes.flat:
    
        """ Initialize image as array of zeros """
    
        img = numpy.zeros((vis_patch_side, vis_patch_side, 3))
        
        """ Divide the rows of parameter values into image channels """
        
        img[:, :, 0] = opt_W1[index, limit0 : limit1].reshape(vis_patch_side, vis_patch_side)
        img[:, :, 1] = opt_W1[index, limit1 : limit2].reshape(vis_patch_side, vis_patch_side)
        img[:, :, 2] = opt_W1[index, limit2 : limit3].reshape(vis_patch_side, vis_patch_side)
        
        """ Plot the image on the figure """
        
        image = axis.imshow(img, interpolation = 'nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1
        
    """ Show the obtained plot """  
        
    matplotlib.pyplot.show()
    
###########################################################################################
""" Returns pooled features for the provided data from a trained network """

def getPooledFeatures(network, images, num_features, res_dim, step_size):
    
    num_images = images.shape[3]
    
    """ Initialize pooled features as array of zeros """
    
    pooled_features_data = numpy.zeros((num_features, num_images, res_dim, res_dim))
    
    for step in range(num_images / step_size):
    
        """ Limits to access batch of images """
        
        limit0 = step_size * step
        limit1 = step_size * (step+1)
        
        image_batch = images[:, :, :, limit0 : limit1]
        
        """ Calculate pooled features for the image batch """
    
        convolved_features = network.convolve(image_batch, num_features)
        pooled_features    = network.pool(convolved_features)
        
        pooled_features_data[:, limit0 : limit1, :, :] = pooled_features
        
        """ Avoid memory overflow """
        
        del(image_batch)
        del(convolved_features)
        del(pooled_features)
    
    """ Reshape data for training / testing """
    
    input_size = pooled_features_data.size / num_images
    pooled_features_data = numpy.transpose(pooled_features_data, (0, 2, 3, 1))
    pooled_features_data = pooled_features_data.reshape(input_size, num_images)
    
    return pooled_features_data
    
###########################################################################################
""" Loads data, trains the Convolutional Neural Network model and predicts classes for test data """

def executeConvolutionalNeuralNetwork():

    """ Initialize parameters for the Convolutional Neural Network model """

    image_dim       = 64     # dimension of the input images
    image_channels  = 3      # number of channels in the image patches
    vis_patch_side  = 8      # side length of sampled image patches
    hid_patch_side  = 20     # side length of representative image patches
    pool_dim        = 19     # dimension of patches taken while pooling
    
    visible_size = vis_patch_side * vis_patch_side * image_channels # number of input units
    hidden_size  = hid_patch_side * hid_patch_side                  # number of hidden units
    res_dim      = (image_dim - vis_patch_side + 1) / pool_dim      # dimension of pooled features

    """ Load parameters learned in the SparseAutoencoderLinear exercise """

    opt_param  = numpy.load('opt_param.npy')
    zca_white  = numpy.load('zca_white.npy')
    mean_patch = numpy.load('mean_patch.npy')
    
    """ Limits to access 'W1' and 'b1' """
    
    limit0 = 0
    limit1 = hidden_size * visible_size
    limit2 = 2 * hidden_size * visible_size
    limit3 = 2 * hidden_size * visible_size + hidden_size
    
    """ Extract 'W1' and 'b1' from the learned parameters """
    
    opt_W1 = opt_param[limit0 : limit1].reshape(hidden_size, visible_size)
    opt_b1 = opt_param[limit2 : limit3].reshape(hidden_size, 1) 
    
    """ Visualize the learned optimal W1 weights """
    
    visualizeW1(numpy.dot(opt_W1, zca_white), vis_patch_side, hid_patch_side)
    
    """ Initialize Convolutional Neural Network model """
    
    network = ConvolutionalNeuralNetwork(opt_W1, opt_b1, zca_white, mean_patch, vis_patch_side, pool_dim)
    
    """ Step size for the pooling process
        Pooling done iteratively to avoid memory overflow """
    
    step_size = 50
    
    """ Load training and test data
        Labels are mapped from [1, 2, 3, 4] to [0, 1, 2, 3] """
    
    train_images, train_labels = loadTrainingDataset()
    test_images, test_labels   = loadTestDataset()
    train_labels = train_labels - 1
    test_labels  = test_labels - 1
    
    """ Get pooled features for training and test data """
    
    softmax_train_data = getPooledFeatures(network, train_images, hidden_size, res_dim, step_size)
    softmax_test_data  = getPooledFeatures(network, test_images, hidden_size, res_dim, step_size)
    
    """ Initialize parameters of the Regressor """
    
    input_size     = hidden_size * res_dim * res_dim  # input vector size
    num_classes    = 4                                # number of classes
    lamda          = 0.0001                           # weight decay parameter
    max_iterations = 200                              # number of optimization iterations
    
    """ Initialize Softmax Regressor with the above parameters """
    
    regressor = SoftmaxRegression(input_size, num_classes, lamda)
    
    """ Run the L-BFGS algorithm to get the optimal parameter values """
    
    opt_solution  = scipy.optimize.minimize(regressor.softmaxCost, regressor.theta, 
                                            args = (softmax_train_data, train_labels,), method = 'L-BFGS-B', 
                                            jac = True, options = {'maxiter': max_iterations})                                        
    opt_theta     = opt_solution.x
    
    """ Obtain predictions from the trained model """
    
    predictions = regressor.softmaxPredict(opt_theta, softmax_test_data)
    
    """ Print accuracy of the trained model """
    
    correct = test_labels[:, 0] == predictions[:, 0]
    print """Accuracy :""", numpy.mean(correct)

executeConvolutionalNeuralNetwork()
