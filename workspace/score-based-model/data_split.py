import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
base_dir = '/mnt/swarm_alpha/KIRC_survival_data'
output_dir = '/mnt/swarm_alpha/KIRC_survival_data_train_val_test/'

# Define the split sizes
train_size = 0.7
val_size = 0.15
test_size = 0.15

# Create the directories if they don't exist
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(output_dir, split)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    for grade in ['KIRC_5YSS_Alive', 'KIRC_5YSS_Deceased']:
        grade_dir = os.path.join(split_dir, grade)
        if not os.path.exists(grade_dir):
            os.makedirs(grade_dir)

# Function to split and copy files
def split_and_copy_files(grade_dir, grade, train_size, val_size, test_size):
    files = os.listdir(grade_dir)
    train_files, test_files = train_test_split(files, test_size=(1 - train_size))
    val_files, test_files = train_test_split(test_files, test_size=test_size/(test_size + val_size))

    # Copy files to respective directories
    for file in train_files:
        shutil.copy(os.path.join(grade_dir, file), os.path.join(output_dir, 'train', grade, file))
    for file in val_files:
        shutil.copy(os.path.join(grade_dir, file), os.path.join(output_dir, 'val', grade, file))
    for file in test_files:
        shutil.copy(os.path.join(grade_dir, file), os.path.join(output_dir, 'test', grade, file))

# Perform the split for each grade
for grade in ['KIRC_5YSS_Alive', 'KIRC_5YSS_Deceased']:
    grade_dir = os.path.join(base_dir, grade)
    split_and_copy_files(grade_dir, grade, train_size, val_size, test_size)

print("Data has been successfully split into train, val, and test sets.")
