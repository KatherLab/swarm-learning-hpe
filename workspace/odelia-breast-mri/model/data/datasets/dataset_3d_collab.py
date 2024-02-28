from pathlib import Path
import pandas as pd
from data.datasets import SimpleDataset3D
from pathlib import Path


class DUKE_Dataset3D_collab(SimpleDataset3D):
    def __init__(self, path_root, item_pointers=None, crawler_glob='*.nii.gz', transform=None, image_resize=None,
                 flip=False, image_crop=None, norm='znorm_clip', to_tensor=True):
        if item_pointers is None:
            item_pointers = []
        super().__init__(path_root, item_pointers, crawler_glob, transform, image_resize, flip, image_crop, norm,
                         to_tensor)
        df = pd.read_csv(self.path_root.parent / 'datasheet.csv')  # , header=[0, 2])

        df = df[[df.columns[0], df.columns[1]]]  # Only pick relevant columns: Patient ID, Tumor Side, Bilateral
        existing_folders = {folder.name for folder in Path(path_root).iterdir() if folder.is_dir()}
        self.df = df[df['PATIENT'].isin(existing_folders)]
        self.df = self.df.set_index('PATIENT', drop=True)
        self.item_pointers = self.df.index[self.df.index.isin(self.item_pointers)].tolist()

    def __getitem__(self, index):
        uid = self.item_pointers[index]
        # Directory path for the specific item
        item_dir = self.path_root / uid
        # Fetch all .nii.gz files in the directory
        nii_gz_files = list(item_dir.glob('**/*.nii.gz'))
        # Default file selection
        file_name = 'SUB_4.nii.gz'
        # If there are multiple .nii.gz files, prioritize SUB_4.nii.gz if it exists
        if len(nii_gz_files) > 1:
            sub_4_path = item_dir / file_name
            if sub_4_path in nii_gz_files:
                path_item = sub_4_path
            else:
                # If SUB_4.nii.gz is not found, just use the first file (or any other logic)
                path_item = nii_gz_files[0]
        elif nii_gz_files:
            # If there's only one .nii.gz file, use it
            path_item = nii_gz_files[0]
        else:
            # Handle the case where no .nii.gz files are found
            raise FileNotFoundError(f"No .nii.gz files found in {item_dir}")

        img = self.load_item(path_item)
        target = self.df.loc[uid]['Malign']
        return {'uid': uid, 'source': self.transform(img), 'target': target}

    @classmethod
    def run_item_crawler(cls, path_root, crawler_ext, **kwargs):
        return [path.relative_to(path_root).name for path in Path(path_root).iterdir() if path.is_dir()]

    def get_labels(self):
        # Assuming 'Malign' is a column in self.df after combining datasets
        # This method should return a list or array of labels corresponding to each item in the dataset
        return self.df['Malign'].values

    def __len__(self):
        return len(self.item_pointers)
