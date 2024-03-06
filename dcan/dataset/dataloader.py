import os
import numpy as np
from collections import OrderedDict
from utils.utils import get_patch_size
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

class DataLoader3D(SlimDataLoaderBase):
    def __init__(self, data, config):
        batch_size = config.train.batch_size
        patch_size = config.model.patch_size
        super().__init__(data, batch_size, None)
        # augmentation parameters
        rot = config.train.aug.rot
        scale = config.train.aug.scale
        rot_x = rot_y = rot_z = (-np.deg2rad(rot), np.deg2rad(rot))
        self.patch_size = get_patch_size(patch_size, rot_x, rot_y, rot_z, scale)
    
    def generate_train_batch(self):
        # random select data
        sels = np.random.choice(list(self._data.keys()), self.batch_size, True)
        # read data, form patch
        images, labels = [], []
        for i, name in enumerate(sels):
            data = np.load(self._data[name]['path'])
            # randomly select patch, centered at (z, x, y)
            z = np.random.choice(data.shape[1])
            x = np.random.choice(data.shape[2])
            y = np.random.choice(data.shape[3])
            loc = np.array((z, x, y))
            # crop
            shape = np.array(data.shape[1:])
            left = np.clip(loc - self.patch_size // 2, a_min=0, a_max=None)
            right = np.clip(loc + (self.patch_size - self.patch_size // 2), a_min=None, a_max=shape)
            data = data[:, left[0]:right[0], left[1]:right[1], left[2]:right[2]]
            # pad
            shape = np.array(data.shape[1:])
            pad_length = self.patch_size - shape
            pad_left = pad_length // 2
            pad_right = pad_length - pad_length // 2
            data = np.pad(data, ((0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1]), (pad_left[2], pad_right[2])))
            images.append(data[:-1])
            # here, label signifies foreground brain region
            labels.append(data[:1] != 0)
        image = np.stack(images)
        label = np.stack(labels)
        return {'data': image, 'label': label}
    
def get_trainloader(data_dir, config):
    dataset = OrderedDict()
    names = os.listdir(data_dir)
    for name in names:
        dataset[name] = OrderedDict()
        dataset[name]['path'] = os.path.join(data_dir, name)
    return DataLoader3D(dataset, config)