from pathlib import Path
import pandas as pd 
from data.datasets import SimpleDataset3D


class DUKE_Dataset3D_collab(SimpleDataset3D):
    def __init__(self, path_root, item_pointers=None, crawler_glob='*.nii.gz', transform=None, image_resize=None,
                 flip=False, image_crop=None, norm='znorm_clip', to_tensor=True):
        if item_pointers is None:
            item_pointers = []
        super().__init__(path_root, item_pointers, crawler_glob, transform, image_resize, flip, image_crop, norm, to_tensor)
        df = pd.read_csv(self.path_root.parent/'datasheet.csv')#, header=[0, 2])
        #print('******')
        #print('self.path_root.parent', self.path_root.parent)
        #df = df[df[df.columns[38]] == 0] # check if cancer is bilateral=1, unilateral=0 or NC
        df = df[[df.columns[0], df.columns[1]]] # Only pick relevant columns: Patient ID, Tumor Side, Bilateral
        existing_folders = {folder.name for folder in Path(path_root).iterdir() if folder.is_dir()}
        #print(existing_folders)
        self.df = df[df['PATIENT'].isin(existing_folders)]
        #print(self.df)
        self.df = self.df.set_index('PATIENT', drop=True)
        #print(self.df)
        self.item_pointers = self.df.index[self.df.index.isin(self.item_pointers)].tolist()
        #print('Data_____frame')
        #print(self.df)
      

    def __getitem__(self, index):
        uid = self.item_pointers[index]
        #print(uid)
        #path_item = [self.path_root/uid/name for name in [ '**.nii.gz' ]]
        file_name = '*.nii.gz'
        #file_name = 'Sub.nii.gz' # For Aachen dataset
        path_item = Path.joinpath(self.path_root/uid, file_name)
        #print(path_item)
        img = self.load_item(path_item)
        #print(img)
        target = self.df.loc[uid]['Malign']
        #print(target)
        return {'uid':uid, 'source': self.transform(img), 'target':target}
    
    @classmethod
    def run_item_crawler(cls, path_root, crawler_ext, **kwargs):
        return [path.relative_to(path_root).name for path in Path(path_root).iterdir() if path.is_dir() ]

    
    def get_labels(self):
        # Assuming 'Malign' is a column in self.df after combining datasets
        # This method should return a list or array of labels corresponding to each item in the dataset
        return self.df['Malign'].values
    def __len__(self):
        return len(self.item_pointers)