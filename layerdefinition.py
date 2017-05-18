import cv2
import numpy as np

def pool(input_data, spatial_extent = 2):
    
    if len(input_data.shape) < 3:
        im_h, im_w = input_data.shape
    else:
        raise Exception("Input to the pooling layer must be a 2D image")
    #print input_data.shape
    r_h = im_h / spatial_extent
    r_w = im_w / spatial_extent
    
    pooled_data = input_data.reshape(r_h, spatial_extent, r_w, spatial_extent)\
                            .max(axis = (1,3))
    return pooled_data
    

def conv2D(image, kernel = (np.array([[1, 1, 1],
                                     [1, 1, 1],
                                     [1, 1, 1]])), bias = 0):
    
    image = image.astype(float) / 255.0
    kernel_sum = kernel.sum()
    
    if len(image.shape) == 3:
        image_height, image_width, image_depth = image.shape
    else:
        image_height, image_width = image.shape
        image_depth = 1
    kernel_height, kernel_width = int(kernel.shape[0]), int(kernel.shape[1])
    
    filtered = np.zeros_like(image)
  
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
                        weight = kernel[kx + int(kernel_height/2), 
                                        ky + int(kernel_width/2)]
                        
                        weighted_pixel_sum += float(pixel) * float(weight)
               
                if len(image.shape) == 3:
                    filtered[x, y, z] = float(weighted_pixel_sum) \
                                        / float(kernel_sum) \
                                        + float(bias)
                else:
                    filtered[x, y] = float(weighted_pixel_sum) \
                                     / float(kernel_sum) \
                                     + float(bias)
            
    return filtered*255.0
    
                    
if __name__ == '__main__':
    
    image = cv2.imread("test_input.png")
    kernel = (np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]]))
              
    #input_data = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    input_data = cv2.resize(image, (100, 100), interpolation = cv2.INTER_CUBIC)
    
    convolved_image = conv2D(input_data, kernel)   
    cv2.imwrite("test1.png",convolved_image)
    convolved_image = cv2.imread("test1.png", 0)
    convolved_image = pool(convolved_image)
    cv2.imwrite("test2.png",convolved_image)
    
    for i in range (convolved_image.shape[0]):
        for j in range (convolved_image.shape[1]):
            if convolved_image[i][j] < 0:
                print "negative:",i,j
