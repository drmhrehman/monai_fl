# code to superpose masks on images for a dataset and save as image files, to allow the dataset to be rapidly examined.

from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2

dataroot = Path('/home/mark/projects/wellcome/ich_segmentation/data/cromis/processed_rigid')
#image_dir = dataroot / 'images'
#label_dir = dataroot / 'labels'

# petteri data
image_dir = Path('/home/mark/projects/wellcome/ich_segmentation/data/cromis/processed_petteri/CT/labeled/MNI_1mm_256vx-3D/data/BM4D_nonNaN_-100/')
label_dir = Path('/home/mark/projects/wellcome/ich_segmentation/data/cromis/processed_petteri/CT/labeled/MNI_1mm_256vx-3D/labels/voxel/hematomaAll/')


output_dir = Path('/home/mark/projects/wellcome/ich_segmentation/logging/misc_output')
output_dir = output_dir / 'rigid_images'
output_dir.mkdir(exist_ok=True)

# construct lists of image/label pairs
image_and_labels = []
for image in sorted(Path(image_dir).glob('*.nii*')):
    image_stem = '_'.join(image.name.split('_')[:2])
    if '_73020-0002-00001-000001' in image_stem:
        continue
    label_name = image_stem
    label = Path(label_dir) / label_name
    # hack to train on either my data or petteri's
    if not label.exists():
        label_name = image_stem + '_hematoma-Manual_MNI_1mm_256vx.nii.gz'
        label = Path(label_dir) / label_name

    # label = Path(self.label_dir) / image.name
    if label.is_file():
        image_and_labels.append({'img': str(image),
                                      'seg': str(label)})




for item in image_and_labels:
    if '01133' in item['img']:
        print('debug')
    image = nib.load(item['img']).get_fdata()
    label = nib.load(item['seg']).get_fdata()
    num_slices = image.shape[2]
    print(item['img'])
    print(np.unique(label))
    for slice in range(num_slices):
        image_slice = image[:,:,slice]
        label_slice = label[:,:,slice]
        if label[:,:,slice].sum() > 0:
            # rescale from [15 100] to [0 1]
            image_slice = image_slice + 15
            image_slice[image_slice<0]=0
            image_slice[image_slice>115]=115
            image_slice = image_slice / 115
            #create a three channel image
            im_combined = np.stack((image_slice,)*3, axis=-1)
            label_int = label_slice.astype(np.int)
            # burn mask into the red channel
            im_combined[:,:,0] = (1-label_int) * image_slice + 0.5*label_int + 0.5* label_int * label_int
            im_combined *= 255
            output_fname = output_dir / (Path(item['img']).name.split('.')[0] + f'_slice_{slice}_petteri.png')
            cv2.imwrite(str(output_fname), im_combined)

            #print('debug')





