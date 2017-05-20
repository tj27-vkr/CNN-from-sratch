import layerdefinition as layer
import cv2
import numpy as np

np.random.seed(10)

image = cv2.imread("avacado.jpg", 0)
number_outputs = 2
y = np.array([[1],[0]])

arr = cv2.resize(image, (100, 100), interpolation = cv2.INTER_CUBIC)

learning_rate = 0.1
#error = 0.0
no_filters = 6
delta_conv = np.zeros([no_filters, 100,100], dtype = float)
delta_fc  = np.zeros([no_filters, 100,100], dtype = float)
loss_conv = np.zeros([no_filters, 100,100], dtype = float)
weight_1 = np.random.randn(no_filters, 3, 3)
bias_1 = np.random.randn(no_filters)

weight_2 = np.array([np.random.randn(no_filters*50*50)])
bias_2 = np.random.randn(number_outputs, 1)

#convolution
convolved_image = np.empty([no_filters, 100,100], dtype = float)
weights_summation = np.empty([no_filters,1], dtype = float)
for iteration in range(no_filters):
    convolved_image[iteration] = np.array(layer.conv2D(arr, weight_1[iteration], bias_1[iteration]))
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
delta_fc = loss_fc * layer.activation(activated_values_fc, True)

# print (delta_fc)
# print (weight_2)
# print ((delta_fc*weight_2).shape)
# delta_conv = layer.activation(convolved_image, True) * delta_fc * weight_2

'''
for i in range(no_filters):
	loss_conv[i] = weights_summation[i] * delta_conv[i]
#print (loss_conv[0].shape)
#delta_conv = loss_conv.T.dot(layer.activation(activated_values_convolution, True))
for i in range(no_filters):
    delta_conv[i] = loss_conv[i] * layer.activation(activated_values_convolution, True)[0]
'''
	
#updating the weights
#weight_1 = weight_1 - learning_rate * delta_conv
weight_2 = weight_2 - learning_rate * delta_fc

#results
#print ("BackPropogation delta",delta_conv, delta_fc)
#print ("final output:",activated_values_fc)
print ("error value:",loss_fc)
