import os
from PIL import Image
import numpy as np
from torchvision import models, transforms
import torch
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import cosine_similarity


# Load a pre-trained InceptionV3 model for FID computation
def load_inception_model():
    model = models.inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # Remove the classification layer
    model.eval()
    return model


# Function to preprocess images for InceptionV3
def preprocess_image(image, transform):
    return transform(image).unsqueeze(0)


# Extract features for a folder of images
def extract_features(folder, model, transform):
    features = []
    for img_name in os.listdir(folder):
        if img_name.lower().endswith('.jpg'):
            img_path = os.path.join(folder, img_name)
            with Image.open(img_path) as img:
                img_tensor = preprocess_image(img, transform)
                with torch.no_grad():
                    feature = model(img_tensor)
                features.append(feature.squeeze().numpy())
    return np.array(features)


# Calculate FID score
def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


# Compute duplicity using cosine similarity
def compute_duplicity(features1, features2):
    similarities = cosine_similarity(features1, features2)
    return (similarities > 0.95).sum() / features1.shape[0]


# Main function
def compare_folders(folder1, folder2):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model = load_inception_model()

    # Extract features
    features1 = extract_features(folder1, model, transform)
    features2 = extract_features(folder2, model, transform)

    # Calculate statistics
    mu1, sigma1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = features2.mean(axis=0), np.cov(features2, rowvar=False)

    # Calculate FID
    fid_score = calculate_fid(mu1, sigma1, mu2, sigma2)

    # Calculate duplicity
    duplicity_score = compute_duplicity(features1, features2)

    return fid_score, duplicity_score


if __name__ == "__main__":
    folder1 = "/mnt/dlhd1/sbm/KIRC_survival/data-and-scratch/data/KIRC_5YSS_Alive"  # Original images
    folder2 = "/mnt/dlhd1/sbm/KIRC_survival/data-and-scratch/data/KIRC_5YSS_Deceased"  # Generated images

    fid, duplicity = compare_folders(folder1, folder2)
    print(f"FID Score: {fid}")
    print(f"Duplicity: {duplicity * 100:.2f}%")
