# SWAG consortium
Please switch to branch 'dev_swag_diffusion' for the latest updates.

## Connection using tailscale
### Install tailscale
1. Go to [tailscale.com](https://tailscale.com) and sign up for an account.
2. Install the tailscale client on your machine.
3. Run `tailscale up` to connect to the SWAG network. You will be prompted to log in to your tailscale account. Please ask the SWAG administrator [Fabian](fabian.laqua@online.de) for the authentication key.
4. Run `tailscale status` to check your connection.

## Apply for one of the diffusion model

### Latent Diffusion Model (LDM) 
Workspace here: [swag-latent-diffusion](workspace%2Fswag-latent-diffusion)

This workspace encompasses the implementation of a Latent Diffusion Model (LDM) primarily based on the MONAI framework. It includes code for training an autoencoder, the LDM itself, and for synthesizing samples from the trained LDM model. Below you'll find detailed instructions on setting up your environment, obtaining the dataset, and the necessary commands to train and generate samples from the models.

#### Environment Setup
Before proceeding, ensure that you have the required libraries installed. A `environment.yml` file is provided for convenience. Use this file to create an environment with all dependencies.

#### Dataset
The dataset employed for training can be found at: https://www.kaggle.com/datasets/nih-chest-xrays/data. Additionally, you have the flexibility to create your own data loader and integrate it into the dataset folder provided in this repository.

#### Training the Autoencoder
Start by training the autoencoder using the following command:
```bash
python 2d_autoencoder.py --dataset NIHXRay --data_dir='/data/NIHXRay/' --n_epochs 400 --batch_size=12 --training_samples=10000 --ckpt_dir='/exp_ae/' --downsample 2 --base_lr 0.00005 --disc_lr 0.0001 --perceptual_weight 0.002 --adv_weight 0.005 --kl_weight 0.00000001 --num_channels '(64,128,128,128)' --perceptual_network 'squeeze'
```

**Arguments:**
- `--dataset`: The name of the dataset.
- `--data_dir`: Directory where the data is located.
- `--n_epochs`: Number of training epochs.
- `--ckpt_dir`: Directory to save checkpoints.
- `--downsample`: Factor for downsampling images, set to 2 for 512x512 resolution.

#### Training the Latent Diffusion Model (LDM)
After training the autoencoder, proceed to train the LDM:

```bash
python 2d_ldm.py --dataset NIHXRay --data_dir='/data/NIHXRay' --n_epochs 1000 --batch_size=150 --training_samples=10000 --ckpt_dir='/exp_ldm/' --downsample 2 --ae_ckpt='/exp_ae/model_best_ae' --generate_samples 16 --beta_end 0.0205 --save_model_interval 50 --latent_scaling 'custom' --custom_scale 0.3
```


**Arguments:**
- `--ae_ckpt`: Path to the trained autoencoder model.
- `--generate_samples`: Number of samples to generate during training.
- `--save_model_interval`: How often to save the model.

#### Sampling from Trained LDM
To generate samples from the trained LDM, use the following command. This example demonstrates sampling for epochs 0 to 1000 with an interval of 100 epochs:

```bash
python 2d_ldm_sample.py --dataset NIHXRay --data_dir='/data/NIHXRay' --n_epochs 1000 --ckpt_dir='/exp_ldm/' --downsample 2 --details 'latent scaling 0.3' --ae_ckpt='/ckpt_ae/model_best_ae' --generate_samples 10000 --results_dir '/sampled_exp_ldm/' --latent_scaling 'custom' --custom_scale 0.3 --multi_gpu --batch_size 256 --save_model_interval 100 --epoch_start 0
```

**Arguments:**
- `--results_dir`: Directory to store generated samples.
- `--epoch_start`: Epoch to start sampling from.
- `--save_model_interval 100`: Frequency of loading the model for data generation (in epochs).

Follow these instructions carefully to successfully train and utilize the Latent Diffusion Model for your projects.


### Score Based Model
Workspace here: [score-based-model](workspace%2Fscore-based-model)

This workspace outlines the process for preparing and utilizing a sample X-ray dataset derived from the publicly available vinDr dataset for score-based modeling. 

**Dataset Acquisition:**

1. Obtain the sample dataset from the vinDr dataset available at [Kaggle](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/data). For simplicity, download the unconditional dataset from [this link](https://gigamove.rwth-aachen.de/en/download/d7744ce97498c742eb1c5dc0c76195e3).
2. Alternatively, you can generate your own dataset from DICOM files using the provided script in the SWAG repository. Visit the [Score-Based Model](https://github.com/SWAG-SwarmLearningforGenerativeModels/Score-Based-Model/blob/dev/dataPreprocessing/vinDrToPng.py) page and follow the instructions to convert and organize the images according to the specified structure.

**Dataset Specification:**

To set up your dataset correctly for training, adhere to the following structure and recommendations:

- **Structure:**
  1. The dataset should be organized within a single folder.
  2. This folder must contain one or more subfolders, each representing a class.
  3. For an unconditional dataset, there should be exactly one folder.
  4. For a conditional dataset, each subfolder must correspond to exactly one class.
  5. The subfolders contain `.png` images.
  6. All images should be square (width == height) and either grayscale (1 channel) or RGB (3 channels), using 8 bits per channel.

- **Recommendations:**
  1. Ideally, images should be 1024x1024 pixels in size.
  2. If scaling is necessary, use bilinear interpolation (e.g., `img.resize((1024,1024), Image.BILINEAR)` for a Python PIL image).

Note: The images do not necessarily need to be the same size as the data loader will adjust them to the required training size.

**Additional Resources:**

- A conditional training dataset with three classes: 'No_finding', 'Cardiomegaly', and 'Pleural_effusion' is available for download [here](https://gigamove.rwth-aachen.de/en/download/bb0704987eaecf78d54d162d671ca794). 
- Accompanying the datasets, a `.yml` file is provided for configuration. You may use this as is or modify it according to your needs.

**User specific path settings:**

- In [swop_profile.yaml](workspace%2Fscore-based-model%2Fswop%2Fswop_profile.yaml) file, set the `privatedata` to the path of the parent folder of your dataset on your local machine. 
Under the data-and-scratch folder there should be a dataset directory: data-and-scratch/data, and a folder for storing the logs and results: data-and-scratch/scratch.

## Rerun the Training

1. Terminate all the SL containers:
   ```
   ./workspace/swarm_learning_scripts/stop-swarm
   ```

2. Ensure no `user-env-pyt1.13-sbm-swop` containers are still running:
   ```
   docker ps -a
   ```
   If a `user-env-pyt1.13-sbm-swop` container is still running, stop it using:
   ```
   docker stop <container_id>
   ```

3. Replace the path in `/opt/hpe/swarm-learning-hpe/workspace/score-based-model/swop/swop_profile.yaml` at:
   ```
   privatedata:
     src: "/mnt/dlhd1/sbm/data-and-scratch/"
   ```
   to your local data dir, since it has got overwritten by `git pull`.

4. Run the environment setup script with your specific parameters:
   ```
   sh workspace/automate_scripts/sl_env_setup/replacement.sh -n 2 -s 100.125.38.128 -w score-based-model -e 100 -d <your_institute_name>
   ```

5. Move the `Cardiomegaly`, `No_finding`, `Pleural_effusion` data dirs from `data-and-scratch/data/` to `data-and-scratch/data/xray_data/` folder.