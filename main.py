import layerdefinition as layer
import cv2
import numpy as np
from scipy.signal import convolve2d
np.random.seed(1)

image = cv2.imread("avacado.jpg", 0)
number_outputs = 2
y = np.array([[1],[0]])

arr = cv2.resize(image, (100, 100), interpolation = cv2.INTER_CUBIC)

learning_rate = 0.1
#error = 0.0
no_filters = 1
delta_conv = np.zeros([no_filters, 100,100], dtype = float)
delta_fc  = np.zeros([no_filters, 100,100], dtype = float)
loss_conv = np.zeros([no_filters, 100,100], dtype = float)
weight_1 = np.random.randn(no_filters, 3, 3)
bias_1 = np.random.randn(no_filters)

#weight_2 = np.array([np.random.randn(no_filters*50*50)])
weight_2 = np.random.randn(no_filters*50*50, 2)
#print(weight_2.shape)
bias_2 = np.random.randn(number_outputs, 1)

for epoch in range(1):
    #convolution
    convolved_image = np.empty([no_filters, 100,100], dtype = float)
    weights_summation = np.empty([no_filters,1], dtype = float)
    for iteration in range(no_filters):
        convolved_image[iteration] = np.array(layer.conv2D(arr,\
                                                        weight_1[iteration],\
                                                        bias_1[iteration]))
        weights_summation[iteration] = np.array(layer.sum_of_weights)
    	
    
    #activation - sigmoid
    activated_values_convolution = layer.activation(convolved_image)
    
    #max pooling
    pooled_image = np.empty([no_filters, 50,50], dtype = float)
    for i in range(no_filters):
        pooled_image[iteration] = layer.pool(activated_values_convolution[i])
    
    #fully connected
    fc_output = layer.fullyconnected(pooled_image, weight_2, bias_2)
    
    #activation - sigmoid
    activated_values_fc = layer.activation(fc_output)
    
    #backprop parameters
    loss_fc = layer.loss(y,activated_values_fc)
    d_fc_temp = np.multiply(-1*loss_fc, \
                           layer.activation(pooled_image.reshape(50,50), True))    
    delta_fc = np.dot(pooled_image.reshape(50,50), d_fc_temp)
    
    #reshaping the weights for convolving - backpropagation
    weight_2_r = weight_2.reshape(2,50,50)
    
    delta_conv_temp = (convolve2d(d_fc_temp, weight_2_r[0]) +\
                       convolve2d(d_fc_temp, weight_2_r[1]))
            
    #padding the matrix with zeros to get 100x100 shape                           
    delta_conv_temp = np.append(delta_conv_temp, np.zeros([1,99]), axis = 0)
    delta_conv_temp = np.append(delta_conv_temp, np.zeros([100,1]), axis = 1)
               
    d_conv_temp =  delta_conv_temp * \
                   layer.activation(convolved_image, True)[0]
        
    delta_conv = np.dot(arr, d_conv_temp)   
    delta_conv_sum = np.sum(delta_conv)             
    
    #updating the weights    
    weight_2_r[0] = weight_2_r[0] - learning_rate * delta_fc
    weight_2_r[1] = weight_2_r[1] - learning_rate * delta_fc
    weight_2 = weight_2_r.reshape(2500,2)
    
    weight_1 = weight_1 - learning_rate * delta_conv_sum

    #results
    print ("error value:",loss_fc)
