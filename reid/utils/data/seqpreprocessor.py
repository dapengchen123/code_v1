from __future__ import absolute_import
import os.path as osp
import torch

from PIL import Image

class SeqPreprocessor(object):
    def __init__(self, seqset, dataset, transform=None):
        super(SeqPreprocessor, self).__init__()
        self.seqset = seqset
        self.identities  = dataset.identities
        self.transform = transform
        self.root = [dataset.images_dir]
        if dataset.other_dir is not None:
            self.root.append(dataset.other_dir)

    def __len__(self):
        return len(self.seqset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)


    def _get_single_item(self, index):

        start_ind, end_ind, pid, camid = self.seqset[index]

        if len(self.root)==1:
             fname = self.identities[pid][camid][start_ind]
             fpath_img = osp.join(self.root[0], fname)
             imgrgb = Image.open(fpath_img).convert('RGB')



        elif len(self.root)==2:
            imgs = []
            flows = []
            for ind in range(start_ind,end_ind):
                fname = self.identities[pid][camid][ind]
                fpath_img = osp.join(self.root[0], fname)
                imgrgb = Image.open(fpath_img).convert('RGB')
                fpath_flow = osp.join(self.root[1], fname)
                flowrgb = Image.open(fpath_flow).convert('RGB')
                imgs.append(imgrgb)
                flows.append(flowrgb)





        else:
            raise RuntimeError("The root is not validate")



        return 1






