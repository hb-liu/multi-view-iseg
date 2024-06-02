import os
import pickle
import numpy as np
from collections import OrderedDict
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

class DataLoader3D(SlimDataLoaderBase):
    def __init__(self, data, config):
        batch_size = config.train.batch_size
        patch_size = config.model.patch_size
        super().__init__(data, batch_size, None)
        self.sample_foreground_prob = 1/3
        self.patch_size = np.array(patch_size)
    
    def generate_train_batch(self):
        # random select data
        sels = np.random.choice(list(self._data.keys()), self.batch_size, True)
        # read data, form patch
        images, labels = [], []
        for i, name in enumerate(sels):
            src = np.load(self._data[name]['src_path'])
            syn = np.load(self._data[name]['syn_path'])
            data = np.concatenate([src[:-1], syn])
            # whether to select foreground
            if i < round(self.batch_size * (1 - self.sample_foreground_prob)):
                force_fg = False
            else:
                force_fg = True
            if force_fg:
                # select patch containing foreground class
                locs = self._data[name]['locs']
                cls = np.random.choice(list(locs.keys()))
                locs = locs[cls]
                loc = locs[np.random.choice(len(locs))]
            else:
                # randomly select patch, centered at (sel_z, sel_x, sel_y)
                sel_z = np.random.choice(data.shape[1])
                sel_x = np.random.choice(data.shape[2])
                sel_y = np.random.choice(data.shape[3])
                loc = np.array((sel_z, sel_x, sel_y))
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
            labels.append(data[-1:])
        image = np.stack(images)
        label = np.stack(labels)
        return {'data': image, 'label': label}

def get_trainloader(cases, config):
    dataset = OrderedDict()
    for name in cases:
        dataset[name] = OrderedDict()
        dataset[name]['src_path'] = os.path.join(config.dataset.src_dir, name+'.npy')
        dataset[name]['syn_path'] = os.path.join(config.dataset.syn_dir, name+'.npy')
        with open(os.path.join(config.dataset.src_dir, name+'.pkl'), 'rb') as f:
            dataset[name]['locs'] = pickle.load(f)
    return DataLoader3D(dataset, config)