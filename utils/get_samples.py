""" Script for getting samples of 100 from each of 1000 classes in imagenet"""

import os
import numpy as np
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.

    Taken from PyTorch ImageFolder Source
    https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def main():
    """
    Some code borrowed from:

    PyTorch ImageFolder `make_dataset` method
    https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    np.random.seed(2022)
    ROOT = '/imagenet/train'
    outfile = 'samples.npy'
    samples = []
    sample_size = 100

    directory = os.path.expanduser(ROOT)

    # get classes
    classes, _ = find_classes(directory)

    for target_class in classes:
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            if len(fnames) < sample_size:
                raise ValueError("Fewer than 100 samples in class")

            # get 100 samples
            rand_indices = np.random.choice(len(fnames), sample_size, replace=False)
            fnames = [fnames[i] for i in rand_indices]

            # add to list of all samples
            samples.extend(fnames)

    np.save(outfile, samples)


if __name__ == '__main__':
    main()