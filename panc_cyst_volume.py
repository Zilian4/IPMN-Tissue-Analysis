import os
import nibabel as nib
import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
import pandas as pd

image_folder_path ='/data/Ziliang/IPMN_cysts_20240909/index_cysts'
cyst_folder_path = '/data/Ziliang/IPMN_cysts_20240909/main_images'
pancreas_folder_path = '/data/Ziliang/IPMN_cysts_20240909/organ_masks'

tissue_v = {'name':[],'pancreas':[],'cyst':[]}

for name in os.listdir(image_folder_path):
    cyst_path = os.path.join(cyst_folder_path,name)
    pancreas_path = os.path.join(pancreas_folder_path,name)

    pancreas_img = nib.load(pancreas_path).get_fdata()
    cyst_img = nib.load(cyst_path).get_fdata()

    pancreas_img = pancreas_img/pancreas_img.max()
    cyst_img = cyst_img/cyst_img.max()

    image_path = os.path.join(image_folder_path,name)
    img = nib.load(image_path)
    voxel_dims = img.header['pixdim'][1:4]
    voxel_volume = np.prod(voxel_dims)


    pancreas_nonzero = np.count_nonzero(pancreas_img)
    pancreas_volume = pancreas_nonzero * voxel_volume/1000
    cyst_nonzero = np.count_nonzero(cyst_img)
    cyst_volume = cyst_nonzero * voxel_volume/1000

    tissue_v['name'].append(name)
    tissue_v['pancreas'].append(pancreas_volume)
    tissue_v['cyst'].append(cyst_volume)
    print(name)
tissue_info = pd.DataFrame(tissue_v)
tissue_info.to_csv('./tissue_from_t2.csv', index=False)
