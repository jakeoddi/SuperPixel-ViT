"""
This module provides custom Datasets that implement superpixel segmentation

Based on:
https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html

Author: Jake Oddi
Last Modified: 05-20-2022
"""
import os
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from skimage.segmentation import slic
from torchvision.datasets import CIFAR10, ImageFolder, folder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union


def create_segments(img_arr, superpixels):
    """
    Creates segments for one image at a time. Then parallelized accross
    the entire batch.

    Parameters
    ----------
    img_arr: np.Array() of an image, with shape 
    (in_chans x img_size x img_size)


    Returns
    -------
    segmented: torch.Tensor() with shape 
    (num_patches x largest_patch_length*in_chans)
    """
    # reshape input array to be passed through segmentation
    # (img_size x img_size x in_chans) -> (in_chans x img_size x img_size)
    img_arr = np.transpose(img_arr, (1, 2, 0))
    # compute segments array
    segment_map = slic(img_arr, 
                    n_segments = superpixels, 
                    sigma=3, 
                    channel_axis=2,
                    #slic_zero=True
                ) 
    # store segment labels
    segment_labels = np.unique(segment_map)

    # get length of largest superpixel
    largest = get_largest(segment_map, segment_labels)

    return segment_map, segment_labels, largest


def get_largest(smap, labels): 
    """
    function for getting size of largest superpixel

    Parameters
    ----------
    smap: img_size x img_size array mapping each pixel to its superpixel/segment
    labels: array containing unique segment labels

    Returns
    -------
    largest: int representing size of largest superpixel
    """
    largest = 0
    # # get list of unique superpixel labels
    # unique = np.unique(segment_map)
    # track occurences of each superpixel
    v = [0 for i in labels]
    tracker = dict(zip(np.unique(smap), v))

    # loop through segment map and count each occurence
    for i in smap.flatten():
        # increment count of each sp label
        tracker[i]+=1
        # check if largest. If so, update largest
        if(tracker[i] > largest):
            largest = tracker[i]

    # check that no values are missing
    assert smap.flatten().shape[0] == sum(tracker.values()), 'Values are missing'

    return largest


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    Taken and modifed from: 
    https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html#ImageFolder

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)
    SAMPLES = set(np.load('samples.npy'))
    print('len samples:', len(SAMPLES))

    if class_to_idx is None:
        _, class_to_idx = folder.find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return folder.has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)


    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            fnames = [f for f in fnames if f in SAMPLES] # ------- MODIFICATION
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


class CIFAR10MeanEmbed(CIFAR10):
    """CIFAR10 with Mean Embedded Superpixels Computed at Load Time"""

    def __init__(
        self,
        superpixels, 
        root='', 
        train=True, 
        download=False, 
        transform=None, 
        ):
        super().__init__(root=root, 
            train=train, 
            download=download, 
            transform=transform
        )
        self.superpixels = superpixels
        self.transform = transform

        # pre-save superpixels
        # set device
        if False:
            os.makedirs('superpixels_CIFAR10MeanEmbed/', exist_ok=True)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            for img_ind in tqdm(range(self.__len__())):
                img, target = super().__getitem__(0)
                # get image as array to compute superpixels
                img_arr = img.cpu().detach().numpy()
                # compute superpixel segmentation
                seg_map, seg_unique, largest_sp  = create_segments(
                    img_arr, 
                    self.superpixels
                    )
                # get segment map as tensor
                seg_map_tens = torch.from_numpy(seg_map)
                # initialize empty list to store superpixels for image `x`
                masks = []

                # compute mean embedding for each superpixel
                for m in seg_unique:
                    # compute mask with only the value of the mth superpixel
                    mask = torch.eq(seg_map_tens, m)#.to(device)
                    # ensuring the input is on device

                    # add to list of masks
                    masks.append(mask)
                # duplicate masks to make up to 64: Elena
                if len(masks) < self.superpixels:
                    rand_ind = random.sample(range(len(masks)), self.superpixels-len(masks))
                    for ind in rand_ind:
                        masks.append(masks[ind])

                # stack superpixels for image `x`
                masks = torch.stack(masks)

                #from pdb import set_trace
                #set_trace()
                np.save('superpixels_CIFAR10MeanEmbed/'+str(img_ind)+'.npy', masks.numpy())

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        # set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if False: # use pre-computed masks
            masks_np = np.load('superpixels_CIFAR10MeanEmbed/'+str(index)+'.npy')
            masks = torch.Tensor(masks_np).to(torch.bool) #torch.Size([64, 32, 32])
            #print("masks.shape ", masks.shape)
            #from pdb import set_trace
            #set_trace()

        else:
             # get image as array to compute superpixels
            img_arr = img.cpu().detach().numpy()

            # compute superpixel segmentation
            seg_map, seg_unique, largest_sp  = create_segments(
                img_arr,
                self.superpixels
                )
            # get segment map as tensor
            seg_map_tens = torch.from_numpy(seg_map)
            # initialize empty list to store superpixels for image `x`
            masks = []

            # compute mean embedding for each superpixel
            for m in seg_unique:
                # compute mask with only the value of the mth superpixel
                mask = torch.eq(seg_map_tens, m)#.to(device)
                # ensuring the input is on device

                # add to list of masks
                masks.append(mask)
            # duplicate masks to make up to 64: Elena
            # case not enough superpixels
            while len(masks) < self.superpixels:
                rand_ind = random.sample(range(len(masks)), self.superpixels-len(masks))
                # for ind in rand_ind:
                #     masks.append(masks[ind])
                masks.append(masks[rand_ind[0]])
            
            # case too many superpixels
            if len(masks) > self.superpixels:
                masks = masks[:self.superpixels]

            # stack superpixels for image `x`
            masks = torch.stack(masks)
            # masks = torch.Tensor(tuple(masks))
            #print('HERE')

            if masks.shape[0]!=self.superpixels:
                print(masks.shape[0])
                from pdb import set_trace
                set_trace()
                assert(masks.shape[0]==self.superpixels) 

            # from pdb import set_trace
            # set_trace()
        #print("img.shape  ", img.shape, " masks.shape ", masks.shape, " target ", target)
        
        return (img, masks), target


# class ImageFolderSampled(ImageFolder):
#     """
#     ImageFolder with downsampling based on number of classes
    
#     Code from https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html#ImageFolder


#     DO NOT USE

#     Args
#     ----
#     root: (String) root directory of dataset
#     transform: (torchvision.transforms) transformation to be applied to data
#     sample_size: (int) desired number of classes in downsampled dataset

#     """

#     def __init__(
#         self, 
#         root='',  
#         transform=None,
#         sample_size=None 
#         ):
#         super().__init__(root=root,  
#             transform=transform
#         )
#         classes, class_to_idx = super().find_classes(self.root)
#         if sample_size:
#             # downsample list and dictionary of classes and their indices
#             # according to `sample_size`
#             classes = classes[:sample_size]
#             d = class_to_idx
#             class_to_idx = {k: d[k] for k in d if d[k] < sample_size}

#         samples = super().make_dataset(
#             self.root, 
#             class_to_idx, 
#             self.extensions, 
#             # self.is_valid_file
#         )

#         self.classes = classes
#         self.class_to_idx = class_to_idx
#         self.samples = samples
#         self.targets = [s[1] for s in samples]

#         self.transform = transform
    
#     def __len__(self):
#         return super().__len__()   

class ImageFolderSampledAllClasses(ImageFolder):
    """
    ImageFolder with downsampling by each class.

    Takes a 
    """
    def __init__(self, root='', transform=None, sample=True):
        super().__init__(
            root=root, 
            transform=transform,
        )
        if sample:
            classes, class_to_idx = super().find_classes(self.root)

            samples = self.make_dataset(
                directory=self.root,
                class_to_idx=class_to_idx,
                extensions=self.extensions,
            )

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Taken from https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html#ImageFolder

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def __len__(self):
            return super().__len__()


class ImageFolderSampledMeanEmbedAllClasses(ImageFolderSampledAllClasses):
    """ImageNet with Mean Embedded Superpixels Computed at Load Time"""

    def __init__(
        self,
        superpixels, 
        root='',  
        transform=None, 
        ):
        super().__init__(
            root=root,  
            transform=transform,
        )
        self.superpixels = superpixels
        # pre-save superpixels
        # set device
        if False:
            os.makedirs('superpixels_ImageNetMeanEmbed/', exist_ok=True)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            for img_ind in tqdm(range(self.__len__())):
                img, target = super().__getitem__(0)
                # get image as array to compute superpixels
                img_arr = img.cpu().detach().numpy()
                # compute superpixel segmentation
                seg_map, seg_unique, largest_sp  = create_segments(
                    img_arr,
                    self.superpixels
                    )
                # get segment map as tensor
                seg_map_tens = torch.from_numpy(seg_map)
                # initialize empty list to store superpixels for image `x`
                masks = []

                # compute mean embedding for each superpixel
                for m in seg_unique:
                    # compute mask with only the value of the mth superpixel
                    mask = torch.eq(seg_map_tens, m)#.to(device)
                    # ensuring the input is on device

                    # add to list of masks
                    masks.append(mask)
                # duplicate masks to make up to 64: Elena
                if len(masks) < self.superpixels:
                    rand_ind = random.sample(range(len(masks)), self.superpixels-len(masks))
                    for ind in rand_ind:
                        masks.append(masks[ind])

                # stack superpixels for image `x`
                masks = torch.stack(masks)

                #from pdb import set_trace
                #set_trace()
                np.save('superpixels_ImageNetMeanEmbed/'+str(img_ind)+'.npy', masks.numpy())

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        start_time = time.time()

        # set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if False: # use pre-computed masks
            masks_np = np.load('superpixels_ImageNetMeanEmbed/'+str(index)+'.npy')
            masks = torch.Tensor(masks_np).to(torch.bool) #torch.Size([64, 32, 32])
            #print("masks.shape ", masks.shape)
            #from pdb import set_trace
            #set_trace()

        else:
             # get image as array to compute superpixels
            img_arr = img.cpu().detach().numpy()
            new_time = time.time()
            # compute superpixel segmentation
            seg_map, seg_unique, largest_sp  = create_segments(
                img_arr,
                self.superpixels
                )
            slic_time = time.time() - new_time # time
            # get segment map as tensor
            seg_map_tens = torch.from_numpy(seg_map)
            # initialize empty list to store superpixels for image `x`
            masks = defaultdict(list) # dictionary<int, [(int, int)]>
            new_time = time.time()
            # get masks for mean embedding for each superpixel
            # for m in seg_unique:
            #     # compute mask with only the value of the mth superpixel
            #     mask = torch.eq(seg_map_tens, m)#.to(device)
            #     # ensuring the input is on device

            #     # add to list of masks
            #     masks.append(mask)

            # loop over each pixel in image
            for i in range(seg_map.shape[0]):
                for j in range(seg_map.shape[1]):
                    # sp value at [i, j] is key, the coordinates tuple (i, j)
                    # is then appended to a list at that dictionary value
                    masks[seg_map[i][j]].append((i, j)) # this is like 3x slower

            feature_time = time.time() - new_time # time
            # duplicate masks to make up to 64: Elena
            # case not enough superpixels
            while len(masks) < self.superpixels:
                # print('too few','len(masks):', len(masks), '   masks.keys()[-1]:', list(masks.keys())[-1])                
                rand_ind = random.sample(range(len(masks)), self.superpixels-len(masks))
                next_ind = len(masks) + 1
                masks[next_ind] = masks[rand_ind[0]]
            
            # case too many superpixels
            while len(masks) > self.superpixels:
                # print('too many','len(masks):', len(masks), '   masks.keys()[-1]:', list(masks.keys())[-1])
                del masks[len(masks)-1] 

            # stack superpixels for image `x`
            # masks = torch.stack(masks)

            if len(masks)!=self.superpixels:
                print(len(masks))
                from pdb import set_trace
                set_trace()
                assert(len(masks)==self.superpixels) 

        #print("img.shape  ", img.shape, " masks.shape ", masks.shape, " target ", target)
        # print('slic time: {0}       compute feature time: {1}'.format(
        #     slic_time,
        #     feature_time 
        #     )
        # )
        pix_count = 0
        for v in masks.values():
            pix_count += len(v)
        
        if pix_count != img.size()[0]**2:
            print('pix_count:',pix_count)
        
        print('img.size():', img.size(), 'len(masks):', len(masks), 'target.size():', target)
        
        return (img, masks), target 


class ImageFolderMeanEmbedAllClasses(ImageFolder):
    """ImageNet with Mean Embedded Superpixels Computed at Load Time"""

    def __init__(
        self,
        superpixels, 
        root='',  
        transform=None, 
        ):
        super().__init__(
            root=root,  
            transform=transform,
        )
        self.superpixels = superpixels
        # pre-save superpixels
        # set device
        if False:
            os.makedirs('superpixels_ImageNetMeanEmbed/', exist_ok=True)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            for img_ind in tqdm(range(self.__len__())):
                img, target = super().__getitem__(0)
                # get image as array to compute superpixels
                img_arr = img.cpu().detach().numpy()
                # compute superpixel segmentation
                seg_map, seg_unique, largest_sp  = create_segments(
                    img_arr,
                    self.superpixels
                    )
                # get segment map as tensor
                seg_map_tens = torch.from_numpy(seg_map)
                # initialize empty list to store superpixels for image `x`
                masks = []

                # compute mean embedding for each superpixel
                for m in seg_unique:
                    # compute mask with only the value of the mth superpixel
                    mask = torch.eq(seg_map_tens, m)#.to(device)
                    # ensuring the input is on device

                    # add to list of masks
                    masks.append(mask)
                # duplicate masks to make up to 64: Elena
                if len(masks) < self.superpixels:
                    rand_ind = random.sample(range(len(masks)), self.superpixels-len(masks))
                    for ind in rand_ind:
                        masks.append(masks[ind])

                # stack superpixels for image `x`
                masks = torch.stack(masks)

                #from pdb import set_trace
                #set_trace()
                np.save('superpixels_ImageNetMeanEmbed/'+str(img_ind)+'.npy', masks.numpy())

    def __len__(self):
        return super().__len__()

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        start_time = time.time()

        # set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if False: # use pre-computed masks
            masks_np = np.load('superpixels_ImageNetMeanEmbed/'+str(index)+'.npy')
            masks = torch.Tensor(masks_np).to(torch.bool) #torch.Size([64, 32, 32])
            #print("masks.shape ", masks.shape)
            #from pdb import set_trace
            #set_trace()

        else:
             # get image as array to compute superpixels
            img_arr = img.cpu().detach().numpy()
            new_time = time.time()
            # compute superpixel segmentation
            seg_map, seg_unique, largest_sp  = create_segments(
                img_arr,
                self.superpixels
                )
            slic_time = time.time() - new_time # time
            # get segment map as tensor
            seg_map_tens = torch.from_numpy(seg_map)
            # initialize empty list to store superpixels for image `x`
            masks = defaultdict(list) # dictionary<int, [(int, int)]>
            new_time = time.time()
            # get masks for mean embedding for each superpixel
            # for m in seg_unique:
            #     # compute mask with only the value of the mth superpixel
            #     mask = torch.eq(seg_map_tens, m)#.to(device)
            #     # ensuring the input is on device

            #     # add to list of masks
            #     masks.append(mask)

            # loop over each pixel in image
            for i in range(seg_map.shape[0]):
                for j in range(seg_map.shape[1]):
                    # sp value at [i, j] is key, the coordinates tuple (i, j)
                    # is then appended to a list at that dictionary value
                    masks[seg_map[i][j]].append((i, j)) # this is like 3x slower

            feature_time = time.time() - new_time # time
            # duplicate masks to make up to 64: Elena
            # case not enough superpixels
            while len(masks) < self.superpixels:
                # print('too few','len(masks):', len(masks), '   masks.keys()[-1]:', list(masks.keys())[-1])                
                rand_ind = random.sample(range(len(masks)), self.superpixels-len(masks))
                next_ind = len(masks) + 1
                masks[next_ind] = masks[rand_ind[0]]
            
            # case too many superpixels
            while len(masks) > self.superpixels:
                # print('too many','len(masks):', len(masks), '   masks.keys()[-1]:', list(masks.keys())[-1])
                del masks[len(masks)-1] 

            # stack superpixels for image `x`
            # masks = torch.stack(masks)
            #print('HERE')

            if len(masks)!=self.superpixels:
                print(len(masks))
                from pdb import set_trace
                set_trace()
                assert(len(masks)==self.superpixels) 

            # from pdb import set_trace
            # set_trace()
        #print("img.shape  ", img.shape, " masks.shape ", masks.shape, " target ", target)
        print('slic time: {0}       compute feature time: {1}'.format(
            slic_time,
            feature_time 
            )
        )

        
        return (img, [masks]), target 