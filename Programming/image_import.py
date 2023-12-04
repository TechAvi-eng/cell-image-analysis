import os
from PIL import Image

# Set the path to the folder containing the image
folder_path = 'png images'

# Set the name of the image file
image_name = '1_00001.png'

# Obtaining the image path
image_1 = os.path.join(folder_path, image_name)
img = Image.open(image_1)

# Display original image
img.show()