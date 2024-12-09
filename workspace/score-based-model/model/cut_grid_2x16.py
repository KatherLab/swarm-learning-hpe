import os

from PIL import Image

def cut_out_images(input_image_path, output_folder):
    # Open the input image
    img = Image.open(input_image_path)
    img_width, img_height = img.size

    # Assuming the grid has 16 rows and 2 columns with 2-pixel spaces
    num_rows = 16
    num_columns = 2
    image_width = (img_width - (num_columns + 1) * 2) // num_columns
    image_height = (img_height - (num_rows + 1) * 2) // num_rows

    # Loop through the grid and cut out images
    for row in range(num_rows):
        for col in range(num_columns):
            left = col * (image_width + 2) + 2
            upper = row * (image_height + 2) + 2
            right = left + image_width
            lower = upper + image_height
            box = (left, upper, right, lower)

            # Crop the image and save it
            cropped_img = img.crop(box)
            grid_image_stemname = os.path.splitext(grid_image_path)[-2].split('/')[-1]
            print(f'Saving image {grid_image_stemname}_{row}_{col}.jpg')
            # Save the cropped image
            image_save_path = os.path.join(save_dir, f'{grid_image_stemname}_{row}_{col}.jpg')
            cropped_img.save(image_save_path)

# Usage example
grid_image_dir = '/mnt/swarm_beta/sbm_evaluation/conditional_grid'  # Path to your grid image
# iterate over all grid images in the directory
for grid_image_name in os.listdir(grid_image_dir):
    grid_image_path = os.path.join(grid_image_dir, grid_image_name)
    save_dir = '/mnt/swarm_beta/sbm_evaluation/conditional_cut'  # Directory to save the cropped images

    cut_out_images(grid_image_path, save_dir)