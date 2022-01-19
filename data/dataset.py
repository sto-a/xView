import os
import time
import numpy as np
from PIL import Image
import setuptools 

from torch.utils.data import Dataset

from .wv_util import get_labels

class XViewDataset(Dataset):
    def __init__(self, root, ann_file):
        super().__init__()
        
        self.root = root
        self.ann_file = ann_file
        
        if not os.path.isdir(root):
            raise RuntimeError('Dataset not found.')
        
        if not os.path.isfile(ann_file):
            raise RuntimeError('Annotation file not found.')
        
        print('Loading annotations into memory ...', end=' ')
        tb = time.time()
        coords, chips, classes = get_labels(ann_file)
        print("Done (t={:.2f}s)".format(time.time()-tb))
        
        print('Annotation rebuilding ...', end=' ')
        tb = time.time()
        self.ann = []
        img_idx = np.unique(chips)
        for idx in img_idx:
            index = chips == idx
            self.ann.append((idx, coords[index], classes[index]))
        print("Done (t={:.2f}s)".format(time.time()-tb))
        
        print('Creating index ...', end='')
        tb = time.time()
        self.labels = {}
        with open('../utils/xview_class_labels.txt') as f:
            for line in f:
                key, val = line[:-1].split(':')
                self.labels[int(key)] = val
        print("Done (t={:.2f}s)".format(time.time()-tb))
        
    def __len__(self):
        return len(self.ann)  
    
    def __getitem__(self, idx):
        img_name, coords, classes = self.ann[idx]
        
        img = Image.open(os.path.join(self.root, img_name)).convert('RGB')
        
        return img, coords, classes
    


        