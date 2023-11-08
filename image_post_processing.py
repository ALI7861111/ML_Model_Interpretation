import cv2
import numpy as np

# Load your heatmap and original image
heatmap = cv2.imread('heat_map.png', cv2.IMREAD_GRAYSCALE)  # Make sure your heatmap image is grayscale
original_image = cv2.imread('sample_image.jpg')

# Resize the heatmap to match the original image dimensions
heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

# Normalize the values in the heatmap
heatmap_normalized = cv2.normalize(heatmap_resized, None, 0, 2, cv2.NORM_MINMAX)

# Adjust the intensity of the heatmap effect (you can experiment with this value)
intensity = 0.6  # You can change this value to control the effect

# Create a color heatmap by applying a colormap (e.g., Jet) to the normalized heatmap
colormap = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)

# Blend the heatmap with the original image
result = cv2.addWeighted(original_image, 1, colormap, intensity, 0)

# Save or display the resulting image
cv2.imwrite('output_image.jpg', result)


