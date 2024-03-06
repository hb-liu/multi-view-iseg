import numpy as np
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor

def get_train_generator(trainloader, config):
    # parameters
    rot = config.train.aug.rot
    scale = config.train.aug.scale
    patch_size = config.model.patch_size
    # for random rotation
    angle_x = angle_y = angle_z = (-np.deg2rad(rot), np.deg2rad(rot))
    # for mirror
    mirror_axes = (0, 1, 2)
    
    transforms = []
    # spatial transformation 
    transforms.extend([
        SpatialTransform(
            # output patch size
            patch_size=patch_size,
            # how to get data and label
            data_key='data', label_key='label',
            # rotation
            do_rotation=True, angle_x=angle_x, angle_y=angle_y, angle_z=angle_z, p_rot_per_sample=0.2,
            # scaling
            do_scale=True, scale=scale, p_scale_per_sample=0.2,
            # elastic
            do_elastic_deform=True, p_el_per_sample = 0.1,
            # others
            border_mode_data='constant',
            random_crop=False
        )
    ])
    # Mirror
    transforms.extend([MirrorTransform(axes=mirror_axes, data_key='data', label_key='label')])
    # ToTensor
    transforms.extend([NumpyToTensor(keys=['data', 'label'], cast_to='float')])
    
    transforms = Compose(transforms)
    batch_generator = MultiThreadedAugmenter(
        data_loader=trainloader,
        transform=transforms,
        num_processes=config.misc.num_workers,
        pin_memory=True
    )
    return batch_generator