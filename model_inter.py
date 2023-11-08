import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import matplotlib.pyplot as plt
# Load the image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


class HeatMap:
    def __init__(self, shape=(224, 224)):
        self.shape = shape
        self.heatmap = np.zeros(shape)

    def add_value(self, x, y, value):
        if 0 <= x < self.shape[0] and 0 <= y < self.shape[1]:
            self.heatmap[x, y] += value

    def visualize(self):
        print(self.heatmap)
        plt.imshow(self.heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()

    def save_to_file(self, filename):
        plt.imsave(filename, self.heatmap, cmap='hot')
        print(f"Heatmap saved to {filename}")



def get_image_per(path):
    img_path = path  # Replace with the path to your image
    img = image.load_img(img_path, target_size=(224, 224, 3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# Load the pre-trained VGG-16 model
model = VGG16(weights='imagenet')
# Load and preprocess the sample image
img_path = 'sample_image.jpg'  # Replace with the path to your image
image_size = 224
img = image.load_img(img_path, target_size=(image_size, image_size))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
# Make predictions using the VGG-16 model
preds = model.predict(x)
# Get the index of the predicted class
predicted_class_index = np.argmax(preds)
decoded_preds = decode_predictions(preds, top=1)[0][0]
_ , class_detection , base_probability = decoded_preds



heat_map_classifier = HeatMap()
All_permuted_images = np.empty((0, 224, 224, 3))


for start_x in tqdm(range(0,224)):
    for start_y in range(0,224):
        image_path = f'./permutted_images/permuted_image_{start_x}_{start_y}.png'
        image_permuted = get_image_per(image_path)
        preds = model.predict(image_permuted) 
        new_probability = preds[0][predicted_class_index]
        heat_map_classifier.add_value(start_x,start_y,abs(base_probability-new_probability))


heat_map_classifier.save_to_file('heat_map.png')
heat_map_classifier.visualize()