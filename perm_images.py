from tqdm import tqdm
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

image_size = 224

def produce_permutated_image_multiple(path, start_x= 0 ,start_y = 0):
    '''
    Single single image prediction takes way more time than batch prediction.
    It is better idea to permute images and process them in batch configuration

    '''
    img_path = path  # Replace with the path to your image
    img = image.load_img(img_path, target_size=(224, 224, 3))
    x = image.img_to_array(img)
    # Define the coordinates of the top-left corner of the region you want to modify
    region_size = 5
    # Set the pixel values in the specified region to 0
    print(' Started producing all permuted images ')
    All_permuted_images = np.empty((0, 224, 224, 3))
    for start_x in tqdm(range(0,image_size)):
        for start_y in range(0,image_size):
            Image_to_permute = np.copy(x)
            Image_to_permute[start_x:start_x+region_size, start_y:start_y+region_size, :] = 0
            filename = f'./permutted_images/permuted_image_{start_x}_{start_y}.png'
            plt.imsave(filename, Image_to_permute / 255.0)

produce_permutated_image_multiple(path='./sample_image.jpg')