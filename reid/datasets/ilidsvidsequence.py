from __future__ import absolute_import
import os
import os.path as osp
import numpy as np
from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json

class iLIDSVIDSEQUENCE(Dataset):
    def __init__(self, root, split_id=0, num_val=0.0, download=False):
        super(iLIDSVIDSEQUENCE, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                           "You can use download=True to download it.")

    ## rewrite load sequences
    #self.load(num_val)

    @property
    def other_dir(self):
        return osp.join(self.root, 'others')

    def download(self):

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        import tarfile
        from glob import glob
        import shutil
        import scipy.io as sio

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        fpath1 = osp.join(raw_dir, 'iLIDS-VID.tar')
        fpath2 = osp.join(raw_dir, 'Farneback.tar.gz')

        if osp.isfile(fpath1) and osp.isfile(fpath2):
            print("Using the download file:" + fpath1 + " " + fpath2)
        else:
            print("Please firstly download the files")
            raise RuntimeError("Downloaded file missing!")

        exdir1 = osp.join(raw_dir, 'iLIDS-VID')
        exdir2 = osp.join(raw_dir, 'Farneback')

        if not osp.isdir(exdir1):
            print("Extracting tar file")
            cwd = os.getcwd()
            tar = tarfile.open(fpath1, 'r:')
            mkdir_if_missing(exdir1)
            os.chdir(exdir1)
            tar.extractall()
            tar.close()
            os.chdir(cwd)

        if not osp.isdir(exdir2):
            print("Extracting tar file")
            cwd = os.getcwd()
            tar = tarfile.open(fpath2)
            mkdir_if_missing(exdir2)
            os.chdir(exdir2)
            tar.extractall()
            tar.close()
            os.chdir(cwd)

        # reorganzing the dataset
        # Format

        temp_images_dir = osp.join(self.root, 'temp_images')
        mkdir_if_missing(temp_images_dir)

        temp_others_dir = osp.join(self.root, 'temp_others')
        mkdir_if_missing(temp_others_dir)

        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        others_dir = osp.join(self.root, 'others')
        mkdir_if_missing(others_dir)

        fpaths1 = sorted(glob(osp.join(exdir1, 'i-LIDS-VID', 'sequences', '*/*/*.png')))
        fpaths2 = sorted(glob(osp.join(exdir2, 'Farneback', '*/*/*.png')))

        identities_imgraw = [[[] for _ in range(2)] for _ in range(319)]
        identities_otherraw = [[[] for _ in range(2)] for _ in range(319)]

        for fpath in fpaths1:
            fname = osp.basename(fpath)
            fname_list = fname.split('_')
            cam_name = fname_list[0]
            pid_name = fname_list[1]
            cam = int(cam_name[-1])
            pid = int(pid_name[-3:])
            temp_fname = ('{:08d}_{:02d}_{:04d}.png'
                          .format(pid, cam, len(identities_imgraw[pid - 1][cam - 1])))
            identities_imgraw[pid - 1][cam - 1].append(temp_fname)
            shutil.copy(fpath, osp.join(temp_images_dir, temp_fname))

        identities_temp = [x for x in identities_imgraw if x != [[], []]]
        identities_images = identities_temp

        for pid in range(len(identities_temp)):
            for cam in range(2):
                for img in range(len(identities_images[pid][cam])):
                    temp_fname = identities_temp[pid][cam][img]
                    fname = ('{:08d}_{:02d}_{:04d}.png'
                             .format(pid, cam, img))
                    identities_images[pid][cam][img] = fname
                    shutil.copy(osp.join(temp_images_dir, temp_fname), osp.join(images_dir, fname))

        shutil.rmtree(temp_images_dir)


        for fpath in fpaths2:
            fname = osp.basename(fpath)
            fname_list = fname.split('_')
            cam_name = fname_list[0]
            pid_name = fname_list[1]
            cam = int(cam_name[-1])
            pid = int(pid_name[-3:])
            temp_fname = ('{:08d}_{:02d}_{:04d}.png'
                          .format(pid, cam, len(identities_otherraw[pid - 1][cam - 1])))
            identities_otherraw[pid - 1][cam - 1].append(temp_fname)
            shutil.copy(fpath, osp.join(temp_others_dir, temp_fname))

        identities_temp = [x for x in identities_otherraw if x != [[], []]]
        identities_others = identities_temp

        for pid in range(len(identities_temp)):
            for cam in range(2):
                for img in range(len(identities_others[pid][cam])):
                    temp_fname = identities_temp[pid][cam][img]
                    fname = ('{:08d}_{:02d}_{:04d}.png'
                             .format(pid, cam, img))
                    identities_others[pid][cam][img] = fname
                    shutil.copy(osp.join(temp_others_dir, temp_fname), osp.join(others_dir, fname))

        shutil.rmtree(temp_others_dir)