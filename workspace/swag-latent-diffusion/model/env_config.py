import os
from datetime import datetime

def load_environment_variables():
    """Load environment variables and return them as a dictionary."""
    return {
    'data_dir': os.getenv('DATA_DIR', '/mnt/dlhd0/swarmlearning_data/archive/'),
    'dataset': os.getenv('DATASET', 'NIHXRay'),
    'n_epochs': int(os.getenv('N_EPOCHS', 400)),
    'batch_size': int(os.getenv('BATCH_SIZE', 12)),
    'training_samples': int(os.getenv('TRAINING_SAMPLES', 10000)),
    'val_interval': int(os.getenv('VAL_INTERVAL', 1)),  # Assuming a default value as it wasn't in the env vars list
    'num_channels_ae': eval(os.getenv('NUM_CHANNELS_AE', '(64, 128, 128, 128)')),  # Assuming the tuple is stored as a string in the environment variable
    'ae_ckpt': os.getenv('AE_CKPT', ''),
    'num_channels': eval(os.getenv('NUM_CHANNELS', '(256, 512, 768)')),
    'num_head_channels': eval(os.getenv('NUM_HEAD_CHANNELS', '(0, 512, 768)')),
    'base_lr': float(os.getenv('BASE_LR', 0.00005)),
    'disc_lr': float(os.getenv('DISC_LR', 0.0001)),  # Assuming a default as it wasn't in the original env vars
    'perceptual_weight': float(os.getenv('PERCEPTUAL_WEIGHT', 0.002)),
    'adv_weight': float(os.getenv('ADV_WEIGHT', 0.0005)),
    'save_model_interval': int(os.getenv('SAVE_MODEL_INTERVAL', 50)),  # Assuming a default value as it wasn't in the env vars list
    'save_ckpt_interval': int(os.getenv('SAVE_CKPT_INTERVAL', 5)),  # Assuming a default value as it wasn't in the env vars list
    'ckpt_dir': os.getenv('CKPT_DIR', 'data-and-scratch/scratch/exp_ae/'),
    'downsample': int(os.getenv('DOWNSAMPLE', 2)),
    'generate_samples': int(os.getenv('GENERATE_SAMPLES', 3)),  # Assuming a default value as it wasn't in the env vars list
    'latent_scaling': os.getenv('LATENT_SCALING', 'custom'),
    'custom_scale': float(os.getenv('CUSTOM_SCALE', 0.3)),
    'load_checkpoint': os.getenv('LOAD_CHECKPOINT', 'False').lower() == 'true',
    'multi_gpu': os.getenv('MULTI_GPU', 'False').lower() == 'true',
    'augmentation': os.getenv('AUGMENTATION', 'False').lower() == 'true',
    'weighted_sampling': os.getenv('WEIGHTED_SAMPLING', 'False').lower() == 'true',
    'subject_wise': os.getenv('SUBJECT_WISE', 'False').lower() == 'true',
    'num_channels_ae': (64, 128, 128, 128),
    'num_train_timesteps': 1000,
    'val_interval': 1,
    'save_ckpt_interval': 5,
    'load_checkpoint:': os.getenv('load_checkpoint:', 'False').lower() == 'true',
    # Additional vars not directly mapped to argparse but might be needed
    'scratch_dir': os.getenv('SCRATCH_DIR', 'data-and-scratch/scratch'),
    'local_compare_flag': os.getenv('LOCAL_COMPARE_FLAG', 'False').lower() == 'true',
    'use_adaptive_sync': os.getenv('USE_ADAPTIVE_SYNC', 'False').lower() == 'true',
    'sync_frequency': int(os.getenv('SYNC_FREQUENCY', 1024)),
    'ae_ckpt': os.getenv('AE_CKPT', 'data-and-scratch/scratch/exp_ae/model_best_ae'),
    'generate_samples': int(os.getenv('GENERATE_SAMPLES', 16)),
    'beta_end': float(os.getenv('BETA_END', 0.0205)),
    'save_model_interval': int(os.getenv('SAVE_MODEL_INTERVAL', 50)),
    'min_peers': int(os.getenv('MIN_PEERS', 2)),
    'latent_scaling': os.getenv('LATENT_SCALING', 'custom'),
    'custom_scale': float(os.getenv('CUSTOM_SCALE', 0.3)),
    'load_checkpoint': os.getenv('load_checkpoint', 'False').lower() == 'true',
    'max_epochs': int(os.getenv('MAX_EPOCHS', 1000)),
    }

def load_prediction_modules(prediction_flag):
    """Dynamically load prediction modules based on the prediction flag."""
    if prediction_flag == 'ext':
        from predict_ext import predict
        from predict_last_ext import predict_last
    elif prediction_flag == 'internal':
        from predict import predict
        from predict_last import predict_last
    else:
        raise Exception("Invalid prediction flag specified")
    return predict, predict_last

def prepare_dataset(task_data_name, data_dir):
    """Prepare the dataset based on task data name."""
    print('task_data_name: ', task_data_name)
    print("Current Directory ", os.getcwd())

    # Check if data_dir contains only DUKE_ext
    available_dirs = next(os.walk(data_dir))[1]  # List directories directly under data_dir
    if 'DUKE_ext' in available_dirs:
        print("Only DUKE_ext directory found under data_dir. Setting task_data_name to DUKE_ext.")
        task_data_name = "DUKE_ext"

    dataset_class = None
    if task_data_name == "multi_ext":
        from data.datasets import DUKE_Dataset3D_collab as dataset_class
    elif task_data_name == "DUKE_ext":
        from data.datasets import DUKE_Dataset3D as dataset_class

    if dataset_class:
        return dataset_class(flip=True, path_root=os.path.join(data_dir, task_data_name, 'train_val')), task_data_name
    else:
        raise ValueError("Invalid task data name specified")

def generate_run_directory(scratch_dir, task_data_name, model_name, local_compare_flag):
    """Generate the directory path for the current run."""
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    mode = 'local_compare' if local_compare_flag else 'swarm_learning'
    return os.path.join(scratch_dir, f"{current_time}_{task_data_name}_{model_name}_{mode}")

def cal_weightage(train_size):
    estimated_full_dataset_size = 808 # exact training size of Duke 80% dataset, which is the largest across multiple nodes
    weightage = int(100 * train_size / estimated_full_dataset_size)
    if weightage > 100:
        weightage = 100
    return weightage

def cal_max_epochs(preset_max_epochs, weightage):
    return int(preset_max_epochs / (weightage / 100))