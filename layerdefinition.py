import cv2
import numpy as np

sum_of_weights = 0

def pool(input_data, spatial_extent = 2):
    
    if len(input_data.shape) < 3:
        im_h, im_w = input_data.shape
    else:
        raise Exception("Input to the pooling layer must be a 2D image")
    #print input_data.shape
    r_h = im_h // spatial_extent
    r_w = im_w // spatial_extent
    
    pooled_data = input_data.reshape(r_h, spatial_extent, r_w, spatial_extent)\
                            .max(axis = (1,3))
    return pooled_data
    
def fullyconnected(input_data, weights, biases):
    if len(input_data.shape) < 3:
        im_h, im_w = input_data.shape
        im_z = 1
    else:
        im_h, im_w, im_z = input_data.shape
    
    interim = input_data.reshape(im_h * im_w * im_z, 1)
    output = np.dot(weights.T, interim) + biases
    
    return output
      

def conv2D(image,\
           kernel = (np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]])),\
           bias = 0):
    
    image = image.astype(float)
    #kernel_sum = kernel.sum()
    if len(image.shape) == 3:
        image_height, image_width, image_depth = image.shape
    else:
        image_height, image_width = image.shape
        image_depth = 1
    kernel_height, kernel_width = int(kernel.shape[0]), int(kernel.shape[1])
    
    filtered = np.zeros_like(image)
    w_sum = 0
    for z in range(image_depth):
        for x in range(image_height):
            for y in range(image_width):
                
                weighted_pixel_sum = 0
                
                for kx in range(- int(kernel_height/2), 
                                int(kernel_height/2 + 1)):
                    for ky in range(- int(kernel_width/2), 
                                    int(kernel_width/2 + 1)):
                                 
                        pixel = 0
                        pixel_y = y - ky
                        pixel_x = x - kx
                        if ((pixel_y >= 0) and (pixel_y < image_width) and 
                            (pixel_x >= 0) and (pixel_x < image_height)) :
                            if len(image.shape) == 3:
                                pixel = float(image[pixel_x, pixel_y, z])
                            else:
                                pixel = float(image[pixel_x, pixel_y])
                        #updating the weights as we go
                        weight = kernel[kx + int(kernel_height/2), 
                                        ky + int(kernel_width/2)] \
                                 #- float(learning_rate) \
                                 #* float(delta_conv[x, y])
                                        
                        
                        weighted_pixel_sum += float(pixel) * float(weight) \
                                              + float(bias)
                        w_sum += float(weight)
                if len(image.shape) == 3:
                    filtered[x,y,z] = weighted_pixel_sum
                else:
                    filtered[x,y] = weighted_pixel_sum
    
    global sum_of_weights
    sum_of_weights = w_sum
    
    return filtered
  
def activation(x, derivation = False, method = 'sigmoid'):
    if method == 'sigmoid':
        if derivation:
            return activation(x) * (1 - activation(x))
        else:
            return 1.0/(1.0 + np.exp(-x))
    else:
        raise Exception("Unknown activation method")

def loss(expected, actual):
    return 0.5*np.sum(expected - actual)**2

def get_weights_sum():
    return sum_of_weights
    
if __name__ == '__main__':
        
    image = cv2.imread("test_input.png")
    kernel = (np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]]))
              
    number_outputs = 2
    #input_data = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    input_data = cv2.resize(image, (100, 100), interpolation = cv2.INTER_CUBIC)
    
    convolved_image = conv2D(input_data, kernel)   
    cv2.imwrite("test1.png",convolved_image)
    
    convolved_image = cv2.imread("test1.png", 0)
    convolved_image = pool(convolved_image)
    #cv2.imwrite("test2.png",convolved_image)
    
    if len(convolved_image.shape) < 3:
        w = np.random.randn(convolved_image.shape[0] \
                            * convolved_image.shape[1])
    else:
        w = np.random.randn(convolved_image.shape[0] \
                            * convolved_image.shape[1] \
                            * convolved_image.shape[2])
    
    b = np.random.randn(number_outputs, 1)
    output = fullyconnected(convolved_image, w, b)
    
    print (output)
