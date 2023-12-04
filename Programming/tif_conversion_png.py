import os
from PIL import Image # Python Imaging Library Pillow

# Set the directory containing the .tif files
directory = "/Users/nikhildhulashia/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Third Year/Individual Project/Programming/RPE_dataset/Subwindows"

# Create a new directory for the .png files
png_directory = os.path.join("/Users/nikhildhulashia/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Third Year/Individual Project/Programming/RPE_dataset", "Subwindows_png")

if not os.path.exists(png_directory): # If the directory does not exist
    os.makedirs(png_directory) # Create the directory

print("Created directory: Subwindows_png")

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".tif"):
        print(filename)
        # Open the .tif file and convert it to .png
        with Image.open(os.path.join(directory, filename)) as im: # Save the .png file in the new directory
            im.save(os.path.join(png_directory, os.path.splitext(filename)[0] + ".png"))
        