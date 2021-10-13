from numpy import float32
from torchvision import datasets, transforms
from base import BaseDataLoader
from monai.data import ImageDataset, Dataset
from pathlib import Path
from monai.transforms import *
from torch import float32 as tfloat32

class CTDataLoader(BaseDataLoader):
    def __init__(self, image_dir, label_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.image_dir = image_dir
        self.label_dir = label_dir


        # construct lists of image/label pairs
        self.image_and_labels = []
        for image in sorted(Path(self.image_dir).glob('*.nii*')):
            image_stem = '_'.join(image.name.split('_')[:2])
            if '_73020-0002-00001-000001' in image_stem:
                continue
            label_name = image_stem
            label = Path(self.label_dir) / label_name
            # hack to train on either my data or petteri's
            if not label.exists():
                label_name = image_stem + '_hematoma-Manual_MNI_1mm_256vx.nii.gz'
                label = Path(self.label_dir) / label_name

            #label = Path(self.label_dir) / image.name
            if label.is_file():
                self.image_and_labels.append({'img': str(image),
                                         'seg': str(label)})

            else:
                print(f'Cannot find label: \n{label} \nto match image: \n{image}')
        print(f'Found {len(self.image_and_labels)} image and label pairs.')

        # define transforms
        self.train_transforms = Compose(
                            [LoadImageD(keys=['img', 'seg'], reader='NiBabelReader', as_closest_canonical=True),
                             AddChannelD(keys=['img', 'seg']),
                             AddCoordinateChannelsd(keys=['img'], spatial_channels=(1, 2, 3)),
                             Rand3DElasticD(keys=['img', 'seg'], sigma_range=(1, 3), magnitude_range=(-10, 10), prob=0.5,
                                            mode=('bilinear', 'nearest'),
                                            rotate_range=(-0.34, 0.34),
                                            scale_range=(-0.1, 0.1), spatial_size=None),
                             SplitChannelD(keys=['img']),
                             ScaleIntensityRanged(keys=['img_0'], a_min=-15, a_max=100, b_min=-1, b_max=1, clip=True),
                             ConcatItemsD(keys=['img_0', 'img_1', 'img_2', 'img_3'], name='img'),
                             DeleteItemsD(keys=['img_0', 'img_1', 'img_2', 'img_3']),
                             #RandGaussianNoised(keys=['img'], prob=0.5, mean=0, std=0.1),
                             RandSpatialCropD(keys=['img', 'seg'], roi_size=(128, 128, 128), random_center=True, random_size=False),
                             ToTensorD(keys=['img', 'seg']),
                             CastToTypeD(keys=['img'], dtype=tfloat32)
                            ])

        self.val_transforms = Compose(
                            [LoadImageD(keys=['img', 'seg'], reader='NiBabelReader', as_closest_canonical=True),
                             AddChannelD(keys=['img', 'seg']),
                             AddCoordinateChannelsd(keys=['img'], spatial_channels=(1, 2, 3)),
                             SplitChannelD(keys=['img']),
                             ScaleIntensityRanged(keys=['img_0'], a_min=-15, a_max=100, b_min=-1, b_max=1, clip=True),
                             ConcatItemsD(keys=['img_0', 'img_1', 'img_2', 'img_3'], name='img'),
                             DeleteItemsD(keys=['img_0', 'img_1', 'img_2', 'img_3']),
                             ToTensorD(keys=['img', 'seg']),
                             CastToTypeD(keys=['img'], dtype=tfloat32)
                            ])

        self.train_dataset = Dataset(self.image_and_labels, transform=self.train_transforms)
        self.val_dataset = Dataset(self.image_and_labels, transform=self.val_transforms)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers)

        #self.data_check()
        #self.preview_augmentations()
        #self.compare_mark_petteri()

    def compare_mark_petteri(self):
        mark_image_dir  = "../data/cromis/processed2/images/"
        mark_label_dir = "../data/cromis/processed2/labels/"
        petteri_image_dir =  "../data/cromis/processed_petteri/CT/labeled/MNI_1mm_256vx-3D/data/BM4D_nonNaN_-100/"
        petteri_label_dir = "../data/cromis/processed_petteri/CT/labeled/MNI_1mm_256vx-3D/labels/voxel/hematomaAll/"

        # construct lists of image/label pairs
        image_and_labels_mark = []
        for image in sorted(Path(mark_image_dir).glob('*.nii*')):
            image_stem = '_'.join(image.name.split('_')[:2])
            if '_73020-0002-00001-000001' in image_stem:
                continue
            label_name = image_stem
            label = Path(mark_label_dir) / label_name
            # hack to train on either my data or petteri's
            if not label.exists():
                label_name = image_stem + '_hematoma-Manual_MNI_1mm_256vx.nii.gz'
                label = Path(mark_label_dir) / label_name

            #label = Path(self.label_dir) / image.name
            if label.is_file():
                image_and_labels_mark.append({'img': str(image),
                                         'seg': str(label)})

            else:
                print(f'Cannot find label: \n{label} \nto match image: \n{image}')
        print(f'Found {len(image_and_labels_mark)} image and label pairs.')

        # construct lists of image/label pairs
        image_and_labels_petteri = []
        for image in sorted(Path(petteri_image_dir).glob('*.nii*')):
            image_stem = '_'.join(image.name.split('_')[:2])
            if '_73020-0002-00001-000001' in image_stem:
                continue
            label_name = image_stem
            label = Path(petteri_label_dir) / label_name
            # hack to train on either my data or petteri's
            if not label.exists():
                label_name = image_stem + '_hematoma-Manual_MNI_1mm_256vx.nii.gz'
                label = Path(petteri_label_dir) / label_name

            #label = Path(self.label_dir) / image.name
            if label.is_file():
                image_and_labels_petteri.append({'img': str(image),
                                         'seg': str(label)})

            else:
                print(f'Cannot find label: \n{label} \nto match image: \n{image}')
        print(f'Found {len(image_and_labels_petteri)} image and label pairs.')

        mark_dataset = Dataset(image_and_labels_mark, transform=self.train_transforms)
        petteri_dataset = Dataset(image_and_labels_petteri, transform=self.train_transforms)

        # compare with plots
        idx = 120
        self.train_transforms.set_random_state(0)
        mark_data  = mark_dataset[idx]
        im_mark, seg_mark = mark_data['img'], mark_data['seg']
        self.train_transforms.set_random_state(0)
        petteri_data = petteri_dataset[idx]
        im_petteri, seg_petteri = petteri_data['img'], petteri_data['seg']

        slice = 40
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2)
        ax[0][0].imshow(im_mark[0, :, :, slice])
        ax[1][0].imshow(seg_mark[0, :, :, slice])
        ax[0][1].imshow(im_petteri[0, :, :, slice])
        ax[1][1].imshow(seg_petteri[0, :, :, slice])
        plt.show()
        print('debug')


    def data_check(self):
        import matplotlib.pyplot as plt
        idx = 120
        data = self.dataset[idx]
        im, label = data['img'], data['seg']
        print(f'Image shape: {im.shape}, \nLabel shape: {label.shape}')

        slice=48
        fig, ax = plt.subplots(2,2)
        ax[0][0].imshow(im[0,:,:,slice])
        ax[0][1].imshow(label[0,:,:,slice])

        im, label = data['img'], data['seg']
        ax[1][0].imshow(im[0,:,:,slice], vmin=-1, vmax=1)
        ax[1][1].imshow(label[0,:,:,slice])
        plt.show()


        print('debug')


    def preview_augmentations(self):
        # define transforms
        train_transforms = Compose(
                            [LoadImageD(keys=['img', 'seg'], reader='NiBabelReader', as_closest_canonical=True),
                             AddChannelD(keys=['img', 'seg']),
                             AddCoordinateChannelsd(keys=['img'], spatial_channels=(1, 2, 3)),
                             Rand3DElasticD(keys=['img', 'seg'], sigma_range=(1, 3), magnitude_range=(-10, 10), prob=1,
                                            mode=('bilinear', 'nearest'),
                                            rotate_range=(-0.34, 0.34),
                                            scale_range=(-0.1, 0.1), spatial_size=None),
                             SplitChannelD(keys=['img']),
                             ScaleIntensityRanged(keys=['img_0'], a_min=-15, a_max=100, b_min=-1, b_max=1, clip=True),
                             ConcatItemsD(keys=['img_0','img_1','img_2','img_3'],name='img'),
                             DeleteItemsD(keys=['img_0','img_1','img_2','img_3']),
                             #RandGaussianNoised(keys=['img'], prob=1, mean=0, std=0.1),
                             RandSpatialCropD(keys=['img','seg'], roi_size=(128, 128, 128), random_center=True, random_size=False),
                             ToTensorD(keys=['img', 'seg'])
                            ])
        train_transforms2 = Compose(
                            [LoadImageD(keys=['img', 'seg'], reader='NiBabelReader', as_closest_canonical=True),
                             AddChannelD(keys=['img', 'seg']),
                             AddCoordinateChannelsd(keys=['img'], spatial_channels=(1, 2, 3)),
                             Rand3DElasticD(keys=['img', 'seg'], sigma_range=(1, 3), magnitude_range=(-10, 10), prob=0,
                                            mode=('bilinear', 'nearest'),
                                            rotate_range=(-0.34, 0.34),
                                            scale_range=(-0.1, 0.1), spatial_size=None),
                             SplitChannelD(keys=['img']),
                             ScaleIntensityRanged(keys=['img_0'], a_min=-15, a_max=100, b_min=-1, b_max=1, clip=True),
                             ConcatItemsD(keys=['img_0','img_1','img_2','img_3'],name='img'),
                             DeleteItemsD(keys=['img_0','img_1','img_2','img_3']),
                             #RandGaussianNoised(keys=['img'], prob=1, mean=0, std=0.1),
                             RandSpatialCropD(keys=['img','seg'], roi_size=(128, 128, 128), random_center=True, random_size=False),
                             ToTensorD(keys=['img', 'seg'])
                            ])
        train_transforms.set_random_state(0)
        train_transforms2.set_random_state(0)
        dataset = Dataset(self.image_and_labels, transform=train_transforms)
        dataset2 = Dataset(self.image_and_labels, transform=train_transforms2)

        import matplotlib.pyplot as plt
        idx = 60
        data = dataset[idx]
        im, label = data['img'], data['seg']
        print(f'Image shape: {im.shape}, \nLabel shape: {label.shape}')

        slice=60
        fig, ax = plt.subplots(3,2)
        ax[0][0].imshow(im[0,:,:,slice])
        ax[0][1].imshow(im[1,:,:,slice])
        data2 = dataset2[idx]
        im2, label2 = data2['img'], data2['seg']

        ax[1][0].imshow(im2[0,:,:,slice], vmin=-1, vmax=1)
        ax[1][1].imshow(im2[1,:,:,slice])

        ax[2][0].imshow(im2[0,:,:,slice]-im[0,:,:,slice], vmin=-1, vmax=1)
        ax[2][1].imshow(im2[1,:,:,slice]-im[1,:,:,slice])
        plt.show()

        # save transformed images as nifits
        import nibabel as nib
        import numpy as np
        out = Path('augmentation_niftis')
        out.mkdir(exist_ok=True)
        nifti_img = nib.Nifti1Image(np.moveaxis(im.cpu().numpy().squeeze(), 0, 3), np.eye(4))
        nifti_img.to_filename(str(out / f'im.nii.gz'))
        nifti_img = nib.Nifti1Image(np.moveaxis(im2.cpu().numpy().squeeze(), 0, 3), np.eye(4))
        nifti_img.to_filename(str(out / f'im2.nii.gz'))
        print('debug')




