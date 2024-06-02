import os
import numpy as np
import torchio as tio

def get_validset(cases, dir):
    subjects = []
    for name in cases:
        data = np.load(os.path.join(dir, name+'.npy'))
        subject = tio.Subject(
            data = tio.ScalarImage(tensor=data[:-1]),
            label = tio.LabelMap(tensor=data[-1:]),
            name = name
        )
        subjects.append(subject)
    dataset = tio.SubjectsDataset(subjects)
    return dataset