from pathlib import Path
import docker
from time import sleep
import nibabel as nib
import numpy as np

data_dir = Path('/home/mark/projects/wellcome/ich_segmentation/data/cromis/native/images/')
output_dir = '/home/mark/projects/wellcome/ich_segmentation/data/cromis/processed_rigid_ints/'
image_files = data_dir.rglob('*')
ctr = 0
for i, image in enumerate(image_files):
    output_name =  Path(output_dir)/ f'wcrrorop{image.stem}'
    if not output_name.exists():
        ctr += 1
        print(image.name)
        image_file = Path('/data') / 'images' / image.name
        label_file = Path('/data') / 'labels'/ 'hematomaAll' /image.name.replace('.nii.gz','_hematomaAll.nii.gz')

        # check the headers match and if they do not, fix this
        image_nib = nib.load(Path('/home/mark/projects/wellcome/ich_segmentation/data/cromis/native/') / 'images' / image.name)
        label_nib = nib.load(Path('/home/mark/projects/wellcome/ich_segmentation/data/cromis/native/') / 'labels'/ 'hematomaAll' /image.name.replace('.nii.gz','_hematomaAll.nii.gz'))

        # check to see if a modified version of the label is available
        if (Path('/home/mark/projects/wellcome/ich_segmentation/data/cromis/native/') / 'labels'/ 'hematomaAll_yee_edits' /image.name.replace('.nii.gz','_hematomaAll.nii.gz')).exists():
            label_file = Path('/data') / 'labels'/ 'hematomaAll_yee_edits' /image.name.replace('.nii.gz','_hematomaAll.nii.gz')
            print(f'Using an edited version of the label for {image.name}')
            # also use this file for the header check
            label_nib = nib.load(Path('/home/mark/projects/wellcome/ich_segmentation/data/cromis/native/') / 'labels' / 'hematomaAll_yee_edits' / image.name.replace('.nii.gz', '_hematomaAll.nii.gz'))
        else: continue

        if not np.allclose(image_nib.get_affine(), label_nib.get_affine()):
            print(f'Fixing header for {image_file}')
            label_corrected = nib.Nifti1Image(label_nib.get_fdata(), image_nib.affine, image_nib.header)
            label_out_file = Path(output_dir) /image.name.replace('.nii.gz','_hematomaAll.nii.gz')
            label_corrected.to_filename(label_out_file)

            # point docker to new file
            label_file = Path('/output') /image.name.replace('.nii.gz','_hematomaAll.nii.gz')

        client = docker.APIClient()
        container = client.create_container(
                    #image='pwrightkcl/test:preprocess',
                    image='spm_preprocess:latest',
                    #image='pwrightkcl/spm-clinical:20210625',
                    command=f'{image_file} {label_file} /output/',
                    volumes=['/data/',
                             '/output/',
                             '/scripts/'],
                    host_config=client.create_host_config(binds=[
                        '/home/mark/projects/wellcome/ich_segmentation/data/cromis/native/:/data/',
                        f'{output_dir}:/output/',
                        '/home/mark/projects/wellcome/ich_segmentation/ich_segmentation/data_utils:/scripts/'
                    ]),
                    detach=False,
                    entrypoint='/scripts/wrapper_script_rigid.sh',
                    user=1000
        )
        # To start the container and print the output
        response = client.start(container=container.get('Id'))
        print(client.logs(container.get('Id')).decode())
        if (ctr % 3 == 0) and ctr > 0:
            print('sleeping')
            sleep(45)
            # all_unwanted_files = []
            # for ext in ['iy_*', 'y_*', 'rop*', 'p*','cr*']:
            #     all_unwanted_files.extend(Path(output_dir).rglob(ext))
            # print('Deleting unwanted files')
            # for f in all_unwanted_files:
            #     f.unlink()


# print('sleeping')
# sleep(120)
# all_unwanted_files = []
# for ext in ['iy_*', 'y_*', 'rop*', 'p*','cr*']:
#     all_unwanted_files.extend(Path(output_dir).rglob(ext))
# print('Deleting unwanted files')
# for f in all_unwanted_files:
#     f.unlink()



