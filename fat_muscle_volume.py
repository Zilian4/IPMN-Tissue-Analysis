import os
import nibabel as nib
import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
import pandas as pd

image_folder_path ='/data/Ziliang/IPMN/t1/t1/preprocessed'
f_m_path = '/data/Ziliang/IPMN/t1/fat_muscle'
tissue_v = {'name':[],'visceral fat':[],'subcutaneous fat':[],'muscle':[]}

for name in os.listdir(image_folder_path):
    muscle_path = os.path.join(f_m_path,name,'skeletal_muscle.nii.gz')
    subcutaneous_path = os.path.join(f_m_path,name,'subcutaneous_fat.nii.gz')
    torso_path = os.path.join(f_m_path,name,'torso_fat.nii.gz')

    torso_img = nib.load(torso_path).get_fdata()
    subcutaneous_img = nib.load(subcutaneous_path).get_fdata()
    muscle_img = nib.load(muscle_path).get_fdata()

    torso_img = torso_img/torso_img.max()
    subcutaneous_img = subcutaneous_img/subcutaneous_img.max()
    muscle_img = muscle_img/muscle_img.max()


    image_path = os.path.join(image_folder_path,name)
    img = nib.load(image_path)
    voxel_dims = img.header['pixdim'][1:4]
    voxel_volume = np.prod(voxel_dims)

    torso_nonzero = np.count_nonzero(torso_img)
    torso_volume = torso_nonzero * voxel_volume/1000
    subcutaneous_nonzero = np.count_nonzero(subcutaneous_img)
    subcutaneous_volume = subcutaneous_nonzero * voxel_volume/1000
    muscle_nonzero = np.count_nonzero(muscle_img)
    muscle_volume = muscle_nonzero * voxel_volume/1000

    tissue_v['name'].append(name)
    tissue_v['visceral fat'].append(torso_volume)
    tissue_v['subcutaneous fat'].append(subcutaneous_volume)
    tissue_v['muscle'].append(muscle_volume)
    print(name)
tissue_info = pd.DataFrame(tissue_v)
tissue_info.to_csv('./tissue_from_t1.csv', index=False)
