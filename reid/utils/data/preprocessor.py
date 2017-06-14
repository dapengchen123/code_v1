from __future__ import absolute_import
import os.path as osp
import torch

from PIL import Image


class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath_img = osp.join(self.root[0], fname)
            fpath_flow = osp.join(self.root[1], fname)
            imgrgb = Image.open(fpath_img).convert('RGB')
            flowrgb = Image.open(fpath_flow).convert('RGB')

            if self.transform is not None:
                imgrgb = self.transform(imgrgb)
                flowrgb = self.transform(flowrgb)
                img = torch.cat([imgrgb, flowrgb[1:3]], 0)
            else:
                raise RuntimeError("transformation should be given")
        else:
            raise RuntimeError("root should be given")

        return img, fname, pid, camid
