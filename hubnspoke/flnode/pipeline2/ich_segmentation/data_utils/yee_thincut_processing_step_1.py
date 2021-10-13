from pathlib import Path
import docker
from time import sleep


data_dir = Path('/home/mark/projects/wellcome/ich_segmentation/data/yee/201901t04thincut/')
subject_dirs = data_dir.rglob('*')
for i, subject in enumerate(subject_dirs):
    image_files = list(subject.rglob('*.nii'))
    for image in image_files:
        if 'Tilt' not in str(image):
            print(subject.name, image.name)
            image_file = Path('/data') / subject.name /image.name
            client = docker.APIClient()
            container = client.create_container(
                        #image='pwrightkcl/test:preprocess',
                        image='spm_preprocess:latest',
                        command=f'{image_file} /output/',
                        volumes=['/data/',
                                 '/output/',
                                 '/scripts/'],
                        host_config=client.create_host_config(binds=[
                            '/home/mark/projects/wellcome/ich_segmentation/data/yee/201901t04thincut/:/data/',
                            '/home/mark/projects/wellcome/ich_segmentation/data/yee/processed_thincut/:/output/',
                            '/home/mark/projects/wellcome/ich_segmentation/ich_segmentation/data_utils:/scripts/'
                        ]),
                        detach=False,
                        entrypoint='/scripts/wrapper_script_no_labels.sh'
            )

            # To start the container and print the output
            response = client.start(container=container.get('Id'))
            print(client.logs(container.get('Id')).decode())
            if (i % 2 == 0) and i > 0:
                print('sleeping')
                sleep(360)
                all_unwanted_files = []
                for ext in ['iy_*', 'y_*', 'rop*','p*']:
                    all_unwanted_files.extend(Path('/home/mark/projects/wellcome/ich_segmentation/data/yee/processed_thincut/').rglob(ext))
                print('Deleting files')
                for f in all_unwanted_files:
                    f.unlink()

print('sleeping')
sleep(360)
all_unwanted_files = []
for ext in ['iy_*', 'y_*', 'rop*', 'p*']:
    all_unwanted_files.extend(
        Path('/home/mark/projects/wellcome/ich_segmentation/data/yee/processed_thincut/').rglob(ext))
print('Deleting files')
for f in all_unwanted_files:
    f.unlink()




