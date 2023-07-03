from pathlib import Path 
import pandas as pd 
from odelia.data.datasets import SimpleDataset3D


class DUKE_Dataset3D_external(SimpleDataset3D):
    def __init__(self, path_root, item_pointers=None, crawler_glob='*.nii.gz', transform=None, image_resize=None, flip=False, image_crop=None, norm='znorm_clip', to_tensor=True):
        if item_pointers is None:
            item_pointers = []
        super().__init__(path_root, item_pointers, crawler_glob, transform, image_resize, flip, image_crop, norm, to_tensor)
        df = pd.read_csv(self.path_root.parent/'segmentation_metadata_unilateral.csv')#, header=[0, 2])
        print('self.path_root.parent', self.path_root.parent)
        #df = df[df[df.columns[38]] == 0] # check if cancer is bilateral=1, unilateral=0 or NC 
        df = df[[df.columns[0], df.columns[5]]] # Only pick relevant columns: Patient ID, Tumor Side, Bilateral
        #df.columns = ['PatientID', 'Malign']  # Simplify columns as: Patient ID, Tumor Side
        #print('Data_____frame')
        #print(df)
        #for side in ["left", 'right']:
            #dfs.append(pd.DataFrame({
                #'PatientID': df["PatientID"].str.split('_').str[2] + f"_{side}",
                #'Malign':df[["Location", "Bilateral"]].apply(lambda ds: (ds[0] == side[0].upper()) | (ds[1]==1), axis=1)} ))
        self.df = df.set_index('PATIENT', drop=True)
        self.item_pointers = self.df.index[self.df.index.isin(self.item_pointers)].tolist()
        #print('Data_____frame')
        #print(df)

    def __getitem__(self, index):
        uid = self.item_pointers[index]
        #print(uid)
        path_item = [self.path_root/uid/name for name in [ 'Sub.nii.gz' ]]
        img = self.load_item(path_item)
        #print(img)
        target = self.df.loc[uid]['Malign']
        #print(target)
        return {'uid':uid, 'source': self.transform(img), 'target':target}
    
    @classmethod
    def run_item_crawler(cls, path_root, crawler_ext, **kwargs):
        return [path.relative_to(path_root).name for path in Path(path_root).iterdir() if path.is_dir() ]

    

