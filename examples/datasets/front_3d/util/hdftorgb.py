import os
import h5py
import numpy as np
import argparse
from PIL import Image



def save_normals_as_image(hdf5_file_path: str, output_folder: str, image_name: str):
    # Open the HDF5 file
    with h5py.File(hdf5_file_path, 'r') as f:
        # Read the normals data
        normals_data = f['normals'][()]
    
    # Convert the range of the normals from [-1, 1] to [0, 255]
    normals_data = ((normals_data + 1) / 2 * 255).astype(np.uint8)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the normals data as an image
    image_path = os.path.join(output_folder, image_name)
    Image.fromarray(normals_data).save(image_path)
    print(f"Normals image saved to: {image_path}")



def save_numpy_as_image(normals: np.ndarray, output_folder: str, image_name: str):
    # Check that the normals array has the correct shape
    if normals.ndim != 3 or normals.shape[2] != 3:
        raise ValueError("Normals array must have shape (H, W, 3)")
    
    # Convert the range of the normals from [-1, 1] to [0, 255]
    normals_rgb = ((normals + 1) / 2 * 255).astype(np.uint8)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the normals data as an RGB image
    image_path = os.path.join(output_folder, image_name)
    Image.fromarray(normals_rgb).save(image_path)
    print(f"Normals RGB image saved to: {image_path}")



# # Iterate over each .hdf5 file in the data folder
# for hdf5_file in os.listdir(args.output_dir):
#     if hdf5_file.endswith(".hdf5"):
#         hdf5_file_path = os.path.join(args.output_dir, hdf5_file)
#         image_name = os.path.splitext(hdf5_file)[0] + '_normals.png'  # Construct image name from HDF5 file name
#         save_normals_as_image(hdf5_file_path, args.output_dir, image_name)