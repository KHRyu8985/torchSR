from .common import pil_loader, FolderByDir

import os
from typing import Callable, List, Optional, Tuple, Union


class ImagePairs(FolderByDir):
    track_dirs = {
        ('lr', 'val', 2) : os.path.join('ImagePairs_Test')
      , ('hr', 'val', 2) : os.path.join('ImagePairs_Test')
      , ('lr', 'traval', 2) : os.path.join('ImagePairs_Testval')
      , ('hr', 'traval', 2) : os.path.join('ImagePairs_Testval')
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
            preload: bool = False):
        if scale is None:
            raise ValueError("RealSR dataset does not support getting HR images only")
        super(ImagePairs, self).__init__(os.path.join(root, 'ImagePairs'),
                                     scale, track, split, transform,
                                     loader, download, predecode, preload)

    def list_samples_imagepairs(self, track, split, scale):
        track_dir = self.get_dir(track, split, scale)
        all_samples = sorted(os.listdir(track_dir))
        all_samples = [s for s in all_samples if s.lower().endswith(self.extensions)]
        all_samples = [os.path.join(track_dir, s) for s in all_samples]
        hr_samples = [s for s in all_samples if os.path.splitext(s)[0].lower().endswith("_gt")]
        lr_samples = [s for s in all_samples if not os.path.splitext(s)[0].lower().endswith("_gt")]
        if len(lr_samples) != len(hr_samples):
            raise ValueError(f"Number of files for {track}X{scale} {split} does not match between HR and LR")
        return hr_samples, lr_samples

    def init_samples(self):
        if len(self.tracks) != 1:
            raise RuntimeError("Only one scale can be used at a time for RealSR dataset")
        hr_samples, lr_samples = self.list_samples_imagepairs(self.tracks[0], self.split, self.scales[0])
        self.samples = list(zip(hr_samples, lr_samples))
