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
            seq = []
            for ind in range(start_ind, end_ind):
                imgseq = []
                fname = self.identities[pid][camid][ind]
                fpath_img = osp.join(self.root[0], fname)
                imgrgb = Image.open(fpath_img).convert('RGB')
                imgseq.append(imgrgb)
                seq = [imgseq]


        elif len(self.root)==2:
            imgseq = []
            flowseq = []
            for ind in range(start_ind,end_ind):
                fname = self.identities[pid][camid][ind]
                fpath_img = osp.join(self.root[0], fname)
                imgrgb = Image.open(fpath_img).convert('RGB')
                fpath_flow = osp.join(self.root[1], fname)
                flowrgb = Image.open(fpath_flow).convert('RGB')
                imgseq.append(imgrgb)
                flowseq.append(flowrgb)
                seq = [imgseq, flowseq]

        else:
            raise RuntimeError("The root is not validate")

        #
        if self.transform is not None:
            seq = self.transform(seq)

        img_tensor = torch.stack(seq[0], 0)
        flow_tensor = torch.stack(seq[1], 0)


        return img_tensor, flow_tensor, pid, camid, start_ind, end_ind






