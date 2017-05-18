import layerdefinition as layer
import cv2
import numpy as np

np.random.seed(10)

image = cv2.imread("test_input.png", 0)
number_outputs = 2
y = np.array([[1],[0]])

arr = cv2.resize(image, (100, 100), interpolation = cv2.INTER_CUBIC)

#convolution
no_filters = 6
weights_1 = []
biases_1 = []
convolved_image = np.empty([no_filters, 100,100], dtype = float)

for iteration in range(no_filters):
    weight_1 = np.random.randn(3, 3)
    bias_1 = np.random.randn()
    convolved_image[iteration] = np.array(layer.conv2D(arr, weight_1, bias_1))
    weights_1.append(weight_1)
    biases_1.append(bias_1)


#activation - sigmoid
activated_values = layer.activation(convolved_image)

#max pooling
pooled_image = np.empty([no_filters, 50,50], dtype = float)
for iteration in range(no_filters):
    pooled_image[iteration] = layer.pool(activated_values[iteration])

#fully connected
weight_2 = np.random.randn(pooled_image[0]*pooled_image[0]*pooled_image[0])
bias_2 = np.random.randn(number_outputs, 1)
fc_output = layer.fullyconnected(pooled_image, weight_2, bias_2)

#results
print "error value:",layer.loss(y,fc_output)