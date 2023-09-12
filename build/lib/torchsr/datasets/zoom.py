from .common import pil_loader, FolderByDir

import os
from typing import Callable, List, Optional, Tuple, Union
import numpy as np

class Zoom(FolderByDir):
    track_dirs = {
        ('lr', 'val', 2) : os.path.join('val')
      , ('hr', 'val', 2) : os.path.join('val')
      , ('lr', 'traval', 2) : os.path.join('traval')
      , ('hr', 'traval', 2) : os.path.join('traval')
      , ('lr', 'traonly', 2) : os.path.join('traonly')
      , ('hr', 'traonly', 2) : os.path.join('traonly')
    }

    def __init__(
            self,
            root: str,
            scale: int = 2,
            track: Union[str, List[str]] = 'lr',
            split: str = 'val',
            transform: Optional[Callable] = None,
            loader: Callable = pil_loader,
            download: bool = False,
            predecode: bool = False,
            preload: bool = False,
            align: bool = False):
        if scale is None:
            raise ValueError("RealSR dataset does not support getting HR images only")
        super(Zoom, self).__init__(os.path.join(root, 'Zoom'),
                                     scale, track, split, transform,
                                     loader, download, predecode, preload)
        self.align = align

    def list_samples_zoom(self, track, split, scale):
        track_dir = self.get_dir(track, split, scale)
        all_samples = sorted(os.listdir(track_dir))
        all_samples = [os.path.join(track_dir, s, 'aligned') for s in all_samples]
        # all_samples = [s for s in all_samples if s.lower().endswith(self.extensions)]
        # all_samples = [os.path.join(track_dir, s) for s in all_samples]

        i = np.random.randint(low=1, high=5, size=len(all_samples))

        hr_samples = [os.path.join(s, ('0000' + str(i[j].item()) + '.JPG')) for j, s in enumerate(all_samples)]
        if self.split=='val' or self.split=='traval':
            lr_samples = [os.path.join(s, ('0000' + str(i[j].item()) + '_LR.JPG')) for j, s in enumerate(all_samples)]
        else:
            if np.random.uniform(0, 1) > 0.5:
                lr_samples = [os.path.join(s, ('0000' + str(i[j].item()+1) + '_LR.JPG')) for j, s in enumerate(all_samples)]
            else:
                lr_samples = [os.path.join(s, ('0000' + str(i[j].item()) + '_LR.JPG')) for j, s in enumerate(all_samples)]
            if len(lr_samples) != len(hr_samples):
                raise ValueError(f"Number of files for {track}X{scale} {split} does not match between HR and LR")
        return hr_samples, lr_samples

    def init_samples(self):
        if len(self.tracks) != 1:
            raise RuntimeError("Only one scale can be used at a time for RealSR dataset")
        hr_samples, lr_samples = self.list_samples_zoom(self.tracks[0], self.split, self.scales[0])
        self.samples = list(zip(hr_samples, lr_samples))
