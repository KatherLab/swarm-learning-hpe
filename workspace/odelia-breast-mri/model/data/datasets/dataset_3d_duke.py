from pathlib import Path
import pandas as pd
from data.datasets import SimpleDataset3D


class DUKE_Dataset3D(SimpleDataset3D):
    def __init__(self, path_root, item_pointers=[], crawler_glob='*.nii.gz', transform=None, image_resize=None, flip=False, image_crop=None, norm='znorm_clip', to_tensor=True):
        super().__init__(path_root, item_pointers, crawler_glob, transform, image_resize, flip, image_crop, norm, to_tensor)
        df = pd.read_excel(self.path_root.parent/'Clinical_and_Other_Features.xlsx', header=[0, 1, 2])
        df = df[df[df.columns[38]] == 0] # check if cancer is bilateral=1, unilateral=0 or NC
        df = df[[df.columns[0], df.columns[36],  df.columns[38]]] # Only pick relevant columns: Patient ID, Tumor Side, Bilateral
        df.columns = ['PatientID', 'Location', 'Bilateral']  # Simplify columns as: Patient ID, Tumor Side
        dfs = []
        existing_folders = {folder.name for folder in Path(path_root).iterdir() if folder.is_dir()}

        for side in ["left", 'right']:
            dfs.append(pd.DataFrame({
                'PatientID': df["PatientID"].str.split('_').str[2] + f"_{side}",
                'Malign':df[["Location", "Bilateral"]].apply(lambda ds: (ds[0] == side[0].upper()) | (ds[1]==1), axis=1)} ))
        self.df = df[df['PatientID'].isin(existing_folders)]
        self.df = self.df.set_index('PatientID', drop=True)
        self.df = pd.concat(dfs,  ignore_index=True).set_index('PatientID', drop=True)
        self.item_pointers = self.df.index[self.df.index.isin(self.item_pointers)].tolist()

    def __getitem__(self, index):
        uid = self.item_pointers[index]
        path_item = [self.path_root / uid / name for name in ['sub.nii.gz']]
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


