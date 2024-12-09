import os
import torch
from torchvision import transforms
from PIL import Image
from pytorch_fid import fid_score
from torchmetrics.functional import structural_similarity_index_measure, peak_signal_noise_ratio
import random
# Set random seed for reproducibility
random.seed(0)
# Evaluate FID
def calculate_fid(real_images_dir, generated_images_dir):
    fid_value = fid_score.calculate_fid_given_paths(
        [real_images_dir, generated_images_dir], batch_size=50,
        device='cuda', dims=2048, num_workers=16)
    return fid_value

# Evaluate SSIM and PSNR
def evaluate_ssim_psnr(real_image, generated_image):
    ssim_value = structural_similarity_index_measure(real_image, generated_image)
    psnr_value = peak_signal_noise_ratio(real_image, generated_image)
    return ssim_value, psnr_value

# Load images from a directory
def load_images_in_batches(image_dir, transform, device, batch_size=50):
    image_files = os.listdir(image_dir)
    for i in range(0, len(image_files), batch_size):
        images = []
        for img_file in image_files[i:i + batch_size]:
            img_path = os.path.join(image_dir, img_file)
            img = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format
            img = transform(img).unsqueeze(0)
            images.append(img)
        images = torch.cat(images).to(device)
        yield images

# Main script
def main(real_images_dir, generated_images_parentdir, device='cuda', batch_size=50):
    # if there are subfolders of real_images_dir, itertate over them
    for subfolder in os.listdir(generated_images_parentdir):
        generated_images_dir = os.path.join(generated_images_parentdir, subfolder)
        print(f'Evaluating images in {generated_images_dir}')



        # Load and evaluate images in batches

        fid_value = calculate_fid(real_images_dir, generated_images_dir)
        print(f'FID: {fid_value}')
        if fid_value < 50:
            print("The generated images are very similar to the real images.")
        elif fid_value < 100:
            print("The generated images are similar to the real images.")
        else:
            print("The generated images are not similar to the real images.")
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

        real_images_batches = load_images_in_batches(real_images_dir, transform, device, batch_size)
        generated_images_batches = load_images_in_batches(generated_images_dir, transform, device, batch_size)

        ssim_values = []
        psnr_values = []

        for real_images, generated_images in zip(real_images_batches, generated_images_batches):
            # Ensure batch sizes match
            min_batch_size = min(real_images.size(0), generated_images.size(0))
            real_images = real_images[:min_batch_size]
            generated_images = generated_images[:min_batch_size]

            ssim_value, psnr_value = evaluate_ssim_psnr(real_images, generated_images)
            ssim_values.append(ssim_value.item())
            psnr_values.append(psnr_value.item())

        # Compute average SSIM and PSNR
        avg_ssim = sum(ssim_values) / len(ssim_values)
        print(f'Average SSIM: {avg_ssim}')
        if avg_ssim > 0.9:
            print("The generated images are very similar to the real images.")
        elif avg_ssim > 0.8:
            print("The generated images are similar to the real images.")
        elif avg_ssim <= 0.8:
            print("The generated images are not similar to the real images.")

        avg_psnr = sum(psnr_values) / len(psnr_values)
        print(f'Average PSNR: {avg_psnr}')
        if avg_psnr > 30:
            print("The generated images are very similar to the real images.")
        elif avg_psnr > 20:
            print("The generated images are similar to the real images.")
        elif avg_psnr <=20:
            print("The generated images are not similar to the real images.")

        # save the results to a file with values and generated_images_dir name
        with open(f'/mnt/swarm_beta/sbm_evaluation/uncon_sample_cut/sorted/evaluation_result.txt', 'a') as f:
            #f.write(f'FID: {fid_value}\n')
            f.write(f'Average SSIM: {avg_ssim}\n')
            f.write(f'Average PSNR: {avg_psnr}\n')
            f.write(f'Generated images directory: {generated_images_dir}\n')


if __name__ == "__main__":
    #real_images_dir_KIRC = '/mnt/swarm_beta/sbm_evaluation/pathology_data/KIRC'
    #real_images_dir_KIRP = '/mnt/swarm_beta/sbm_evaluation/pathology_data/KIRP'
    real_images_dir = '/mnt/swarm_beta/sbm_evaluation/temp/image_gen_10000'

    generated_images_dir = '/mnt/swarm_beta/sbm_evaluation/temp'
    device = 'cuda'  # or 'cpu' if you prefer

    main(real_images_dir, generated_images_dir, device)
