from pathlib import Path
import nibabel as nib
import shutil
import numpy as np

data_dir = Path('/home/mark/projects/wellcome/ich_segmentation/data/yee/processed/')
image_out_dir =  data_dir /'images'
label_out_dir =  data_dir /'labels'
label_files = data_dir.rglob('*haemorrhage*')

image_out_dir.mkdir(exist_ok=True)
label_out_dir.mkdir(exist_ok=True)
for i, label in enumerate(label_files):
    print(label.name)
    try:
        # if '16031' in image.name:
        #     print('debug')
        image= label.parent / label.name.replace('_haemorrhage.nii','.nii')

        img = nib.load(image)
        data = img.get_fdata()

        print(data.mean())
        if data.mean() > 100:
            print('not here')
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
        shutil.copy(label, label_out)
    except:
        print(f'ERROR with {image.name}')
    # print('debug')

