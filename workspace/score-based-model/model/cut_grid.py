import os
from PIL import Image

def cut_images_from_grid(grid_image_path, save_dir, image_size=224, gap_size=2):
    # Open the grid image
    grid_image = Image.open(grid_image_path)
    grid_width, grid_height = grid_image.size

    # Calculate the number of images per row and column
    num_images_per_row = 4
    num_images_per_col = 4

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Iterate over each image in the grid
    for row in range(num_images_per_col):
        for col in range(num_images_per_row):
            # Calculate the position of the top-left corner of the image
            left = col * (image_size + gap_size)
            upper = row * (image_size + gap_size)
            right = left + image_size
            lower = upper + image_size

            # Crop the image from the grid
            cropped_image = grid_image.crop((left, upper, right, lower))

            grid_image_stemname = os.path.splitext(grid_image_path)[-2].split('/')[-1]
            print(f'Saving image {grid_image_stemname}_{row}_{col}.jpg')
            # Save the cropped image
            image_save_path = os.path.join(save_dir, f'{grid_image_stemname}_{row}_{col}.jpg')
            cropped_image.save(image_save_path)

    print(f'All images have been saved to {save_dir}')

if __name__ == "__main__":
    grid_image_dir = '/mnt/swarm_beta/sbm_evaluation/uncon_sample_grid'  # Path to your grid image
    # iterate over all grid images in the directory
    for grid_image_name in os.listdir(grid_image_dir):
        grid_image_path = os.path.join(grid_image_dir, grid_image_name)
        save_dir = '/mnt/swarm_beta/sbm_evaluation/uncon_sample_cut/'  # Directory to save the cropped images

        cut_images_from_grid(grid_image_path, save_dir)

