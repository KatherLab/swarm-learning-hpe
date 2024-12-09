import os
import torch
from pytorch_fid import fid_score
from PIL import Image
import torchvision.transforms as transforms

# Define transforms to resize images to 299x299
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# Function to resize and save images
def resize_and_save_images(input_dir, output_dir, transform):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img = Image.open(os.path.join(input_dir, filename)).convert('RGB')
            img = transform(img)
            img = transforms.ToPILImage()(img)
            img.save(os.path.join(output_dir, filename))


# Paths to your datasets
original_images_path = '/mnt/swarm_beta/sbm_evaluation/pathology_data/minirandomsamples_uncon'
resized_original_images_path = '/mnt/swarm_beta/sbm_evaluation/pathology_data/minirandomsamples_unconre'
generated_images_parentdir = '/mnt/swarm_beta/sbm_evaluation/uncon_sample_large_cut/'
resized_generated_images_parentdir = '/mnt/swarm_beta/sbm_evaluation/uncon_sample_large_cutre/'

# Resize original images
resize_and_save_images(original_images_path, resized_original_images_path, transform)

for subfolder in os.listdir(generated_images_parentdir):
    # Skip .txt files
    if subfolder.endswith('.txt'):
        continue

    generated_images_dir = os.path.join(generated_images_parentdir, subfolder)
    resized_generated_images_dir = os.path.join(resized_generated_images_parentdir, subfolder)

    # Resize generated images
    resize_and_save_images(generated_images_dir, resized_generated_images_dir, transform)

    print(f'Evaluating images in {resized_generated_images_dir}')

    fid_value = fid_score.calculate_fid_given_paths(
        [resized_original_images_path, resized_generated_images_dir],
        batch_size=16,
        device='cuda',
        dims=2048
    )
    print(f'FID: {fid_value}')

    # Write to txt file with FID and folder name
    with open('/mnt/swarm_beta/sbm_evaluation/uncon_sample_cut/sorted/evaluation_result.txt', 'a') as f:
        f.write(f'FID: {fid_value} for {subfolder}\n')
