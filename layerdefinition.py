import cv2
import numpy as np

def conv2D(image, kernel = (np.array([[1, 1, 1],
                                     [1, 1, 1],
                                     [1, 1, 1]])), bias = 0):
    
    image = image.astype(float) / 255.0
    kernel_sum = kernel.sum()
   
    image_height, image_width, image_depth = image.shape
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
                            pixel = float(image[pixel_x, pixel_y, z])
                        weight = kernel[kx + int(kernel_height/2), 
                                        ky + int(kernel_width/2)]
                        
                        weighted_pixel_sum += float(pixel) * float(weight)
               
                filtered[x, y, z] = float(weighted_pixel_sum) \
                                    / float(kernel_sum) \
                                    + float(bias)
            
    return filtered*255.0
    
                    
if __name__ == '__main__':
    
    image = cv2.imread("test_elon.jpg")
    kernel = (np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]]))
                      
    convolved_image = conv2D(image, kernel)
    
    for i in range(5):
        convolved_image = conv2D(convolved_image, kernel)
        
    cv2.imwrite("test.png",convolved_image)