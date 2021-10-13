import nibabel as nib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# script to convert .nii segmentation polygons generated in mipav into filled out segmentation masks suitable for
#training/testing



# need to manually store order of labels as they order they're exported changes in MIPAV...
default_label_dict = {'ventricles':1, 'haemorrhage':2, 'calcium':3}
label_dicts = {
'OneDrive_01_27-04-2021': {'ventricles':1, 'haemorrhage':3, 'calcium':2},
'OneDrive_02_27-04-2021': {'ventricles':2, 'haemorrhage':1, 'calcium':3},
'OneDrive_03_27-04-2021': {'ventricles':3, 'haemorrhage':1, 'calcium':3},
'OneDrive_04_27-04-2021': {'ventricles':2, 'haemorrhage':1, 'calcium':3},
'OneDrive_05_27-04-2021': {'ventricles':1, 'haemorrhage':3, 'calcium':1},
'OneDrive_06_27-04-2021': {'ventricles':1, 'haemorrhage':3, 'calcium':1},
'OneDrive_07_27-04-2021': {'ventricles':2, 'haemorrhage':1, 'calcium':3},
'OneDrive_08_27-04-2021': {'ventricles':2, 'haemorrhage':3, 'calcium':1},
'OneDrive_09_27-04-2021': {'ventricles':3, 'haemorrhage':1, 'calcium':2},
'OneDrive_10_27-04-2021': {'ventricles':3, 'haemorrhage':2, 'calcium':1},
'OneDrive_11_27-04-2021': {'ventricles':1, 'haemorrhage':3, 'calcium':2},
'OneDrive_12_27-04-2021': {'ventricles':2, 'haemorrhage':1, 'calcium':3},
'OneDrive_13_27-04-2021': {'ventricles':2, 'haemorrhage':1, 'calcium':3},
'OneDrive_14_27-04-2021': {'ventricles':2, 'haemorrhage':3, 'calcium':1},
'OneDrive_15_27-04-2021': {'ventricles':3, 'haemorrhage':2, 'calcium':1},
'OneDrive_16_27-04-2021':{'ventricles':1, 'haemorrhage':3, 'calcium':2},
'OneDrive_17_27-04-2021':{'ventricles':2, 'haemorrhage':3, 'calcium':1},
'OneDrive_18_27-04-2021':{'ventricles':1, 'haemorrhage':2, 'calcium':3},
'OneDrive_19_27-04-2021':{'ventricles':3, 'haemorrhage':1, 'calcium':2},
'OneDrive_20_27-04-2021':{'ventricles':3, 'haemorrhage':1, 'calcium':2},
'OneDrive_21_27-04-2021':{'ventricles':2, 'haemorrhage':3, 'calcium':1},
'OneDrive_22_27-04-2021': {'ventricles':2, 'haemorrhage':1, 'calcium':3},
'OneDrive_23_27-04-2021':{'ventricles':1, 'haemorrhage':2, 'calcium':3},
'OneDrive_24_27-04-2021':{'ventricles':1, 'haemorrhage':3, 'calcium':2},
'OneDrive_25_27-04-2021':{'ventricles':3, 'haemorrhage':1, 'calcium':2},
'OneDrive_26_27-04-2021':{'ventricles':3, 'haemorrhage':1, 'calcium':2},
'OneDrive_27_27-04-2021':{'ventricles':1, 'haemorrhage':3, 'calcium':2}
}

import scipy.ndimage as ndimage
from skimage.morphology import binary_closing
yee_data = Path('../../data/yee/masks201901t04/')
for subject in yee_data.glob('*'):
    print(subject)
    mask_path = list(subject.glob('*_smask.nii'))[0]
    mask_nib = nib.load(mask_path)
    mask_data = mask_nib.get_fdata()

    # make individual niis for each label type
    label_dict = label_dicts[subject.name]

    for name, index in label_dict.items():
        mask_data_filled = np.zeros_like(mask_data)
        for slice_num in range(mask_data.shape[2]):
            slice = (mask_data[:,:,slice_num]==index)
            slice = binary_closing(slice, selem=[[1,1,1],[1,1,1],[1,1,1]])
            slice = binary_closing(slice, selem=[[1,1,1],[1,1,1],[1,1,1]])
            slice = binary_closing(slice, selem=[[1,1,1],[1,1,1],[1,1,1]])
            #slice = binary_closing(slice)
            mask_data_filled[:,:,slice_num] = ndimage.binary_fill_holes(slice)

        new_img = nib.nifti1.Nifti1Image(mask_data_filled, None, header=mask_nib.header.copy())
        new_img_path = str(mask_path).replace('smask',name)
        new_img.to_filename(new_img_path)


    # make comvined nii for all label types
    mask_data_filled = np.zeros_like(mask_data)
    for name, index in label_dict.items():
        for slice_num in range(mask_data.shape[2]):
            slice = (mask_data[:,:,slice_num]==index)
            slice = binary_closing(slice, selem=[[1,1,1],[1,1,1],[1,1,1]])
            #slice = binary_closing(slice)
            mask_data_filled[:,:,slice_num] += ndimage.binary_fill_holes(slice) * index

    if len(np.unique(mask_data_filled)) > 4:
        print('Warning - mask overlap')
    new_img = nib.nifti1.Nifti1Image(mask_data_filled, None, header=mask_nib.header.copy())
    new_img_path = str(mask_path).replace('smask','all')
    new_img.to_filename(new_img_path)

    # plt.imshow(mask_data_filled[200:300,300:400,14])
    # plt.show()
    #print('debug')