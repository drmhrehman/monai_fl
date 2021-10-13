from pathlib import Path
import nibabel as nib
import shutil
import numpy as np

data_dir = Path('/home/mark/projects/wellcome/ich_segmentation/data/cromis/processed_rigid_ints/')
image_out_dir =  data_dir /'images'
label_out_dir =  data_dir /'labels'
label_files = data_dir.rglob('*hematoma*')

image_out_dir.mkdir(exist_ok=True)
label_out_dir.mkdir(exist_ok=True)
for i, label in enumerate(label_files):
    print(label.name)
    try:
        # if '16031' in image.name:
        #     print('debug')
        image= label.parent / label.name.replace('_hematomaAll.nii','.nii')

        img = nib.load(image)
        data = img.get_fdata()

        print(data.mean())
        if data.mean() > 100:
            data -= 1024
        else:
            data[data == 0] = -1024

        # cast to float32
        new_dtype = np.float32
        data = data.astype(new_dtype)
        img.set_data_dtype(new_dtype)


        new_img = nib.Nifti1Image(data, img.affine, img.header)
        image_out = image_out_dir /image.name
        label_out = label_out_dir /image.name

        new_img.to_filename(image_out)

        # cast label file to int
        label = nib.load(label)
        label_data = label.get_fdata()
        label_data[label_data>0] = 1
        label_data = label_data.astype(np.uint8)
        label.set_data_dtype(np.uint8)
        new_label = nib.Nifti1Image(label_data, label.affine, label.header)
        new_label.to_filename(label_out)
        #shutil.copy(label, label_out)
    except:
        print(f'ERROR with {image.name}')
    # print('debug')

