from __future__ import absolute_import
import os
import os.path as osp

import numpy as np

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json



class iLIDSVID(Dataset):
    url = 'http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar'
    md5 = '7752bd15b611558701bb0e2380ed8950'
    def __init__(self, root, split_id=0, num_val=0.0, download=False):
        super(iLIDSVID, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_val)

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return


        import hashlib
        import tarfile
        from glob import glob
        import shutil
        from scipy.misc import imsave, imread
        from six.moves import urllib
        import scipy.io as sio


        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(raw_dir, 'iLIDS-VID.tar')

        if osp.isfile(fpath) and \
            hashlib.md5(open(fpath,'rb').read()).hexdigest() == self.md5:
            print("Using download file: " + fpath)

        else:
            print("Downloading {} to {}".format(self.url, fpath))
            urllib.request.urlretrieve(self.url, fpath)

        # Extract the file
        exdir = osp.join(raw_dir, 'iLIDS-VID')
        if not osp.isdir(exdir):
            print("Extracting tar file")
            cwd = os.getcwd()
            tar = tarfile.open(fpath, 'r:')
            mkdir_if_missing(exdir)
            os.chdir(exdir)
            tar.extractall()
            tar.close()
            os.chdir(cwd)

        # reorganzing the dataset
        # Format
        temp_images_dir = osp.join(self.root, 'temp_images')
        mkdir_if_missing(temp_images_dir)
        images_dir = osp.join(self.root,'images')
        mkdir_if_missing(images_dir)
        fpaths = sorted(glob(osp.join(exdir, 'i-LIDS-VID', 'sequences', '*/*/*.png')))

        identities_raw = [[[] for _ in range(2)] for _ in range(319)]


        for fpath in fpaths:
            fname = osp.basename(fpath)
            fname_list = fname.split('_')
            cam_name = fname_list[0]
            pid_name = fname_list[1]
            cam = int(cam_name[-1])
            pid = int(pid_name[-3:])
            temp_fname = ('{:08d}_{:02d}_{:04d}.png'
                     .format(pid, cam, len(identities_raw[pid-1][cam-1])))
            identities_raw[pid-1][cam-1].append(temp_fname)
            shutil.copy(fpath, osp.join(temp_images_dir, temp_fname))

        identities_temp = [x for x in identities_raw if x != [[], []]]
        identities = identities_temp
        for pid in range(len(identities_temp)):
            for cam in range(2):
                for img in range(len(identities_temp[pid][cam])):
                    temp_fname = identities_temp[pid][cam][img]
                    fname = ('{:08d}_{:02d}_{:04d}.png'
                             .format(pid, cam, img))
                    identities[pid][cam][img] = fname
                    shutil.copy(osp.join(temp_images_dir, temp_fname), osp.join(images_dir, fname))

        # Save  meta information into a json file
        meta = {'name': 'iLIDS-VID', 'shot': 'single', 'num_cameras': 2,
                'identities': identities}

        write_json(meta, osp.join(self.root, 'meta.json'))

        # Consider fixed training and testing split
        splitmat_name = osp.join(exdir, 'train-test people splits', 'train_test_splits_ilidsvid.mat')
        data = sio.loadmat(splitmat_name)
        person_list = data['ls_set']
        num = len(identities)
        splits = []
        for i in range(10):
            pids = (person_list[i]-1).tolist()
            trainval_pids = sorted(pids[:num // 2])
            test_pids = sorted(pids[num // 2:])
            split = {'trainval': trainval_pids,
                     'query': test_pids,
                     'gallery': test_pids}
            splits.append(split)
        write_json(splits, osp.join(self.root, 'splits.json'))

























