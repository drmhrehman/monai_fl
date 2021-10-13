from pathlib import Path
import docker
from time import sleep


data_dir = Path('/home/mark/projects/wellcome/ich_segmentation/data/yee/masks201901t04/')
subject_dirs = data_dir.rglob('*')
for i, subject in enumerate(subject_dirs):
    #client = docker.from_env()
    #container = client.containers.run('pwrightkcl/test:preprocess', '-it --rm -v /home/mark/projects/wellcome/ich_segmentation/data/yee/masks201901t04/OneDrive_27_27-04-2021/:/data/ --entrypoint /data/wrapper_script.sh pwrightkcl/test:preprocess /data/RJZ73212660.nii /data/RJZ73212660_haemorrhage.nii /data/output', detach=True)

    #print(container.logs().decode())
    #if '14_27-04-2021' not in str(subject): continue
    image_files = subject.rglob('*.nii')
    for image in image_files:
        if len(str(image.name).split('_')) == 1:
            print(subject.name, image.name)


            image_file = Path('/data') / subject.name /image.name
            label_file = Path('/data') / subject.name /image.name.replace('.nii','_haemorrhage.nii')
            client = docker.APIClient()
            container = client.create_container(
                        #image='pwrightkcl/test:preprocess',
                        image='spm_preprocess:latest',
                        command=f'{image_file} {label_file} /output/',
                        volumes=['/data/',
                                 '/output/',
                                 '/scripts/'],
                        host_config=client.create_host_config(binds=[
                            '/home/mark/projects/wellcome/ich_segmentation/data/yee/masks201901t04/:/data/',
                            '/home/mark/projects/wellcome/ich_segmentation/data/yee/processed:/output/',
                            '/home/mark/projects/wellcome/ich_segmentation/ich_segmentation/data_utils:/scripts/'
                        ]),
                        detach=False,
                        entrypoint='/scripts/wrapper_script_rigid.sh',
                        user=1000
            )

            # To start the container and print the output
            response = client.start(container=container.get('Id'))
            print(client.logs(container.get('Id')).decode())
            print(client.logs(container.get('Id')).decode())
            if (i % 3 == 0) and i > 0:
                print('sleeping')
                sleep(120)
                all_unwanted_files = []
                for ext in ['iy_*', 'y_*', 'rop*',  'cr*']:
                    all_unwanted_files.extend(
                        Path('/home/mark/projects/wellcome/ich_segmentation/data/yee/processed/').rglob(ext))
                print('Deleting unwanted files')
                for f in all_unwanted_files:
                    f.unlink()

print('sleeping')
sleep(120)
all_unwanted_files = []
for ext in ['iy_*', 'y_*', 'rop*', 'cr*']:
    all_unwanted_files.extend(
        Path('/home/mark/projects/wellcome/ich_segmentation/data/yee/processed/').rglob(ext))
print('Deleting unwanted files')
for f in all_unwanted_files:
    f.unlink()




