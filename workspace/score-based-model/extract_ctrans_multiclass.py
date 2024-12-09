import os
import torch
import numpy as np
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import hashlib
import json
import h5py
from swin_transformer import swin_tiny_patch4_window7_224, ConvStem

# Define the CTransPath model and helper functions

class FeatureExtractorCTP:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def init_feat_extractor(self, device: str):
        digest = self.get_digest(self.checkpoint_path)
        assert digest == '7c998680060c8743551a412583fac689db43cec07053b72dfec6dcd810113539'

        self.model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
        self.model.head = nn.Identity()

        ctranspath = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(ctranspath['model'], strict=True)

        if torch.cuda.is_available():
            self.model = self.model.to(device)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        model_name = 'xiyuewang-ctranspath-7c998680'
        print("CTransPath model successfully initialised...\n")
        return model_name

    @staticmethod
    def get_digest(file: str):
        sha256 = hashlib.sha256()
        with open(file, 'rb') as f:
            while True:
                data = f.read(1 << 16)
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()

# Initialize the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint_path = '/home/swarm/anaconda3/envs/stamp/lib/python3.10/site-packages/stamp/resources/ctranspath.pth'
feature_extractor = FeatureExtractorCTP(checkpoint_path=checkpoint_path)
model_name = feature_extractor.init_feat_extractor(device=device)
model = feature_extractor.model.to(device)
model.eval()

# Data transforms
data_transform = feature_extractor.transform

# Paths to the data directories
'''
datatype = 'KIRC'
foldertype = 'test'
base_dir = f'/mnt/swarm_beta/sbm_evaluation/pathology_data/{foldertype}'  # Replace with your actual base directory path
image_dir = os.path.join(base_dir, datatype)  # Directory containing the image tiles
feature_dir = os.path.join(f'/mnt/swarm_beta/sbm_evaluation/pathology_data/{foldertype}_ctrans', datatype)
'''

image_dir = f'/mnt/swarm_alpha/KIRC_4grading/PathologyImages2ndGeneration/step200000_cut'
feature_dir = f'/mnt/swarm_alpha/KIRC_4grading/PathologyImages2ndGeneration/step200000_cut_ctrans/'
os.makedirs(feature_dir, exist_ok=True)

# Load dataset
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = list(Path(image_folder).rglob('*.jpg'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path.stem

dataset = ImageDataset(image_dir, transform=data_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=16)

# Feature extraction function
def extract_features(model, dataloader, feature_dir):
    model.eval()
    with torch.no_grad():
        for inputs, img_names in tqdm(dataloader, desc="Extracting features"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            features = outputs.detach().cpu().numpy()
            for i, feature in enumerate(features):
                feature_path = os.path.join(feature_dir, f"{img_names[i]}.npy")
                np.save(feature_path, feature)

    # Extract features
    extract_features(model, dataloader, feature_dir)

    print("Feature extraction completed.")
