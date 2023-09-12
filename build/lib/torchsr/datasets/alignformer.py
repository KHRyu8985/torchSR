import os
from typing import Callable, List, Optional, Tuple, Union

from .common import FolderByDir, pil_loader


class AlignFormer(FolderByDir):
    track_dirs = {
        ('hr', 'train', 1) : os.path.join('ref', 'train')
      , ('hr', 'traonly', 1) : os.path.join('ref', 'traonly')
      , ('hr', 'traval', 1) : os.path.join('AlignFormer', 'traval')
      , ('real', 'train', 1) : os.path.join('lq', 'train')
      , ('real', 'traonly', 1) : os.path.join('lq', 'traonly')
      , ('real', 'traval', 1) : os.path.join('lq', 'traval')
      , ('hr', 'val', 1) : os.path.join('AlignFormer', 'test_sub')
      , ('real', 'val', 1) : os.path.join('lq', 'test_sub')
      , ('real', 'test', 1) : os.path.join('lq', 'test_sub')
    }

    def __init__(
            self,
            root: str,
            scale: Union[int, List[int], None] = None,
            track: Union[str, List[str]] = 'real',
            split: str = 'train',
            transform: Optional[Callable] = None,
            loader: Callable = pil_loader,
            download: bool = False,
            predecode: bool = False,
            preload: bool = False):
        super(AlignFormer, self).__init__(os.path.join(root, 'AlignFormer_dataset', 'iphone_dataset'),
                                    scale, track, split, transform,
                                    loader, download, predecode, preload)
