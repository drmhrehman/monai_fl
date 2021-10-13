from pathlib import Path
import nibabel as nib
import shutil



data_dir = Path('/home/mark/projects/wellcome/ich_segmentation/data/yee/201901t04thincut/') # raw thincut data
thick_data_dir = Path('/home/mark/projects/wellcome/ich_segmentation/data/yee/masks201901t04') # raw thick data
processed_dir = Path('/home/mark/projects/wellcome/ich_segmentation/data/yee/processed_thincut') # output dir for processed thincut data
thick_processed_dir = Path('/home/mark/projects/wellcome/ich_segmentation/data/yee/processed') # output dir for processed thick data
subject_dirs = data_dir.rglob('*')

for i, subject in enumerate(subject_dirs):
    image_files = list(subject.rglob('*.nii'))
    for image in image_files:
        if 'Tilt' not in str(image):
            print(subject.name, image.name)
            acquistion_number = image.parent.name.replace('ich','')
            thick_data_target = thick_data_dir / f'OneDrive_{acquistion_number}_27-04-2021'
            target_images = thick_data_target.rglob('*.nii')
            for target in target_images:
                if len(str(target.name).split('_')) == 1:
                    original_filename=  processed_dir / f'wrorop{image.name}'
                    target_filename = processed_dir / f'wrorop{target.stem}_thincut.nii'
                    original_label_filename =  thick_processed_dir / f'wrop{target.stem}_haemorrhage.nii'
                    target_label_filename = processed_dir / f'wrop{target.stem}_thincut_haemorrhage.nii'

                    # make a copy of processed thincut voluem with correct name
                    shutil.copy(original_filename, target_filename)
                    shutil.copy(original_label_filename, target_label_filename)

