import os
from datetime import datetime

def load_environment_variables():
    """Load environment variables and return them as a dictionary."""
    return {
        'task_data_name': os.getenv('DATA_FOLDER', 'multi_ext'),
        'scratch_dir': os.getenv('SCRATCH_DIR', '/platform/scratch'),
        'data_dir': os.getenv('DATA_DIR', '/platform/data/'),
        'max_epochs': int(os.getenv('MAX_EPOCHS', 100)),
        'min_peers': int(os.getenv('MIN_PEERS', 2)),
        'max_peers': int(os.getenv('MAX_PEERS', 7)),
        'local_compare_flag': os.getenv('LOCAL_COMPARE_FLAG', 'False').lower() == 'true',
        'use_adaptive_sync': os.getenv('USE_ADAPTIVE_SYNC', 'False').lower() == 'true',
        'sync_frequency': int(os.getenv('SYNC_FREQUENCY', 1024)),
        'model_name': os.getenv('MODEL_NAME', 'ResNet101'),
        'prediction_flag': os.getenv('PREDICT_FLAG', 'ext')
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
    dataset_class = None
    if task_data_name == "multi_ext":
        from data.datasets import DUKE_Dataset3D_collab as dataset_class
    elif task_data_name == "DUKE_ext":
        from data.datasets import DUKE_Dataset3D as dataset_class
    if dataset_class:
        return dataset_class(flip=True, path_root=os.path.join(data_dir, task_data_name, 'train_val'))
    else:
        raise ValueError("Invalid task data name specified")

def generate_run_directory(scratch_dir, task_data_name, model_name, local_compare_flag):
    """Generate the directory path for the current run."""
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    mode = 'local_compare' if local_compare_flag else 'swarm_learning'
    return os.path.join(scratch_dir, f"{current_time}_{task_data_name}_{model_name}_{mode}")

def cal_weightage(train_size):
    estimated_full_dataset_size = 1000
    weightage = int(100 * train_size / estimated_full_dataset_size)
    if weightage > 100:
        weightage = 100
    return weightage

def cal_max_epochs(preset_max_epochs, weightage):
    return int(preset_max_epochs / (weightage / 100))