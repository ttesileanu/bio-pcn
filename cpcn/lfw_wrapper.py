"""Wrapper for LFW dataset."""

import os.path as osp
import time

import torchvision
import torchvision.transforms as T

import torch

from typing import Tuple, Any


class LFWSanitized(torchvision.datasets.LFWPairs):
    """A version of the LFW dataset sanitized for use with BioPCN.
    
    In particular, this removes the integer label and returns only pairs of images.

    It also calculates mean and standard deviation *on the training set* and normalizes
    accordingly (if desired). The mean and standard deviation are cached to file.

    Finally, it flattens (if desired).
    """

    def __init__(self, *args, normalize: bool = True, flatten: bool = True, **kwargs):
        self.original_root = kwargs.get("root", args[0] if len(args) > 0 else None)
        super().__init__(*args, **kwargs)

        self.normalize = normalize
        self.flatten = flatten
        if self.normalize:
            self.mu, self.std = self._load_normalization_parameters()
            normalization = T.Normalize(mean=self.mu, std=self.std)
            if self.transform is None:
                self.transform = normalization
            else:
                self.transform = T.Compose([self.transform, normalization])
        else:
            self.mu = None
            self.std = None

        if self.flatten:
            flattening = T.Lambda(torch.flatten)
            if self.transform is None:
                self.transform = flattening
            else:
                self.transform = T.Compose([self.transform, flattening])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Drop the integer label, returning only pairs of images."""
        original = super().__getitem__(index)

        return original[:2]

    def _load_normalization_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # search for saved params
        mean_path = osp.join(self.root, f"sample_mean.pt")
        std_path = osp.join(self.root, f"sample_std.pt")
        try:
            mean = torch.load(mean_path)
            std = torch.load(std_path)
            print("Loaded normalization parameters from file.")
        except:
            # not found; calculate!
            print("Calculating normalization on train split...", flush=True)
            t0 = time.time()

            trainset = torchvision.datasets.LFWPairs(
                self.original_root,
                download=True,
                split="Train",
                transform=self.transform,
            )

            samples = []
            # XXX might be better to have different means for each image
            for sample in trainset:
                samples.append(sample[0])
                samples.append(sample[1])
            torch_samples = torch.cat(samples, 0)

            mean = torch_samples.mean(dim=0)
            std = torch_samples.std(dim=0)

            torch.save(mean, mean_path)
            torch.save(std, std_path)

            t1 = time.time()
            print(f"Took {t1 - t0:.1f} seconds.")

        return mean, std
