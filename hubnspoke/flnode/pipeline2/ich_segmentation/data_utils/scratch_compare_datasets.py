from pathlib import Path
import nibabel as nib
import numpy as np
petteri_data = Path('/home/mark/projects/wellcome/ich_segmentation/data/cromis/processed_petteri/CT/labeled/MNI_1mm_256vx-3D/data/BM4D_brainWeighed_nonNaN_-100')
petteri_ims = list(petteri_data.glob('*.nii.gz'))

my_data = Path('/home/mark/projects/wellcome/ich_segmentation/data/cromis/processed/images')
label_dir = Path('/home/mark/projects/wellcome/ich_segmentation/data/cromis/processed/labels')
my_ims = list(my_data.glob('*.nii'))


# petteri_ims_name = ['CROMIS2ICH_' + im.name.split('_')[1] for im in petteri_ims]
# my_ims_name = [im.name.split('.')[0] for im in my_ims]
#
# print(set(my_ims_name).difference(set(petteri_ims_name)))
# print(set(petteri_ims_name).difference(set(my_ims_name)))
# print('debug')



raw_data_dir = Path('/home/mark/projects/wellcome/ich_segmentation/data/cromis/native/images/')
raw_label_root = Path('/home/mark/projects/wellcome/ich_segmentation/data/cromis/native/')
image_files = raw_data_dir.rglob('*')
for i, image in enumerate(image_files):
    label = raw_label_root / 'labels'/ 'hematomaAll' /image.name.replace('.nii.gz','_hematomaAll.nii.gz')

    try:
        image_nib = nib.load(image)
        label_nib = nib.load(label)
        if not np.allclose(image_nib.get_affine(), label_nib.get_affine()):
            print(image)
            print(image_nib.get_affine() - label_nib.get_affine())
            label_corrected = nib.Nifti1Image(label_nib.get_fdata(), image_nib.affine, image_nib.header)
            label_corrected.to_filename('test.nii.gz')
    except:
        print(f'Error with {image}')
    #print('debug')
