import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler

from .kaggle_dataset import KaggleAerialDataset
from .landcover_dataset import LandCoverDataset

class UnifiedSSLDataset(ConcatDataset):
    def __init__(self, config, transform=None):
        self.datasets_list = []
        self.weights_per_dataset = []
        
        # Load Kaggle Dataset
        kaggle_cfg = config.DATA.datasets.kaggle
        if kaggle_cfg.weight > 0:
            kaggle_ds = KaggleAerialDataset(
                data_dir=kaggle_cfg.path,
                transform=transform
            )
            if len(kaggle_ds) > 0:
                self.datasets_list.append(kaggle_ds)
                self.weights_per_dataset.append(kaggle_cfg.weight)
                
        # Load LandCover Dataset
        landcover_cfg = config.DATA.datasets.landcover
        if landcover_cfg.weight > 0:
            landcover_ds = LandCoverDataset(
                data_dir=landcover_cfg.path,
                patch_size=landcover_cfg.patch_size,
                stride=landcover_cfg.stride,
                transform=transform
            )
            if len(landcover_ds) > 0:
                self.datasets_list.append(landcover_ds)
                self.weights_per_dataset.append(landcover_cfg.weight)
                
        super().__init__(self.datasets_list)
        
    def get_sampler(self, num_samples=None):
        """
        Creates a WeightedRandomSampler that balances the datasets according
        to their specified weights, regardless of their size.
        """
        sample_weights = []
        
        for ds, ds_weight in zip(self.datasets, self.weights_per_dataset):
            # The weight of a single sample in this dataset is:
            # (overall dataset weight) / (number of samples in the dataset)
            weight_per_sample = ds_weight / len(ds)
            sample_weights.extend([weight_per_sample] * len(ds))
            
        if num_samples is None:
            # By default, an epoch size is the sum of all samples,
            # but balanced based on probabilities.
            num_samples = len(self)
            
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True
        )
