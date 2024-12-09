import os
import numpy as np
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import tensorflow as tf
from pathlib import Path


# Function to load images and preprocess them for InceptionV3
def load_and_preprocess_images(image_paths):
    images = []
    for image_path in image_paths:
        img = keras_image.load_img(image_path, target_size=(299, 299))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        images.append(img_array)
    return np.vstack(images)


# Function to extract features from images using InceptionV3
def extract_features(images, model):
    return model.predict(images)


# Function to calculate the Fréchet Inception Distance (FID)
def calculate_fid(features1, features2):
    # Calculate the mean and covariance of the features
    mu1, sigma1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = features2.mean(axis=0), np.cov(features2, rowvar=False)

    # Calculate the difference in means
    diff = mu1 - mu2
    # Calculate the trace of the covariance matrices
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)

    # If the covariance matrix has numerical issues, try to fix it
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate the FID score
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


# Function to gather all .jpg files under two directories
def gather_images(folder1, folder2):
    folder1_images = list(Path(folder1).rglob('*.jpg'))
    folder2_images = list(Path(folder2).rglob('*.jpg'))
    return folder1_images, folder2_images


def main(folder1, folder2):
    # Gather all .jpg images from both folders
    folder1_images, folder2_images = gather_images(folder1, folder2)

    # Load the pre-trained InceptionV3 model
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    # Preprocess images and extract features
    folder1_images_data = load_and_preprocess_images(folder1_images)
    folder2_images_data = load_and_preprocess_images(folder2_images)

    folder1_features = extract_features(folder1_images_data, inception_model)
    folder2_features = extract_features(folder2_images_data, inception_model)

    # Calculate and print the FID score
    fid_score = calculate_fid(folder1_features, folder2_features)
    print(f'FID score between the two folders: {fid_score}')


if __name__ == "__main__":
    # Specify your folder paths here
    folder1 = 'path_to_folder_1'
    folder2 = 'path_to_folder_2'

    main(folder1, folder2)
