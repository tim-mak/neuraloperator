import matplotlib
matplotlib.use('Agg') # Forces headless rendering, preventing DataLoader crashes
from logging import config
import torch
import neuralop
from neuralop import data
from neuralop.data.datasets.pt_dataset import PTDataset
import numpy as np
from pathlib import Path
from typing import List, Union, Optional
import json
from neuralop.data.datasets.tensor_dataset import TensorDataset
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from tims.airfrans_panelfoil_linear_jac.Airfrans_DataProcessor_Linear_Jacobian import AirfransDataProcessor_Linear_Jacobian
from tims.airfrans_panelfoil_linear_jac.Airfrans_Evaluator import AirfoilEvaluator
from tims.airfrans_panelfoil.SelectiveUnitGaussianNormalizer import SelectiveUnitGaussianNormalizer
from torch.utils.data._utils.collate import default_collate

import zencfg
from neuralop.models.base_model import get_model
from tqdm import tqdm

input_names =  ["X", "Y", "U_x_pot", "U_y_pot","Cp_pot", "exp_sdf","x_xi","x_eta","y_xi","y_eta","det_J"]

output_names = ["Cp_delta", "U_x_delta", "U_y_delta", "log_nut_ratio"]

def collate_batch_with_props(batch):
    """
    batch: a list of dicts from Dataset.__getitem__
    """
    # 1. Separate the props from the tensors
    props = [item.pop('props') for item in batch]
    
    # 2. Use the standard collate for x and y (which are same shape)
    # This creates the standard [B, C, H, W] tensors
    collated_batch = default_collate(batch)

    # 3. Put the props back as a raw list of dictionaries
    collated_batch['props'] = props
    
    return collated_batch

def verify_input_encoder(encoder):
    print(f"\n{'='*20} INPUT ENCODER AUDIT {'='*20}")
    if encoder is None:
        print("No input encoder detected. Skipping audit.")
        return
    # 1. Check Channel Dimensions
    mean = encoder.mean.flatten()
    std = encoder.std.flatten()
    print(f"Stats Shape: {list(encoder.mean.shape)} | Channels detected: {len(mean)}")

    # 2. Check Physical Mapping
    # We expect 5 channels of stats representing [u_inf, v_inf, mask, sdf, log_Re]
    # Mask should be unaltered min=0, max=1
    #names = ["x (inf)", "v_velocity (inf)", "mask", "SDF (geometry)", "log_Re"]
    #names = ["X", "Y", "U_x_pot", "U_y_pot","Cp_pot", "exp_sdf","x_xi","x_eta","y_xi","y_eta","det_J"]

    print(f"\n{'Channel':<20} | {'Mean':>10} | {'Std':>10}")
    print("-" * 45)
    for i, name in enumerate(input_names):
        m, s = mean[i].item(), std[i].item()
        print(f"{name:<20} | {m:>10.4f} | {s:>10.4f}")

    # 3. Verify Selective Logic
    channels = getattr(encoder, 'channels_to_normalize', [])
    print(f"\nActive Channels for Normalization: {channels}")
    
    if 5 in channels:
        print("!! WARNING: Channel 5 (exp_sdf) is set to be normalized! This will corrupt geometry.")
    else:
        print("✓ SUCCESS: Channel 5 (exp_sdf) will be passed through untouched.")
    print(f"{'='*63}\n")

def verify_output_encoder(encoder):
    print(f"\n{'='*20} OUTPUT ENCODER AUDIT {'='*20}")
    if encoder is None:
        print("No output encoder detected. Skipping audit.")
        return
    # 1. Check Channel Dimensions
    mean = encoder.mean.flatten()
    std = encoder.std.flatten()
    print(f"Stats Shape: {list(encoder.mean.shape)} | Channels detected: {len(mean)}")

    # 2. Check Physical Mapping
    # We expect 4 channels of stats representing [u_deficit, v_deficit, Cp, log_nut_ratio]
    # which will be applied to indices [0, 1, 2, 3] of the 4D output.
    
    print(f"\n{'Channel':<20} | {'Mean':>10} | {'Std':>10}")
    print("-" * 45)
    for i, name in enumerate(output_names):
        m, s = mean[i].item(), std[i].item()
        print(f"{name:<20} | {m:>10.4f} | {s:>10.4f}")

    print(f"{'='*63}\n")

def get_dataset_stats(training_file):
    
    print(f"--- Loading Training Split: {training_file} ---")
    data = torch.load(training_file)
    x = data['x']  # [N, 12, Gx, Gy]
    y = data['y']  # [N, 4, Gx, Gy
    # need to drop sdf channel for stats reporting since it is not used as input to the model and will not be normalized
    x =torch.cat([x[:, :5, :, :], x[:, 6:, :, :]], dim=1) 

    print(f"Getting Raw Dataset Stats from {training_file}")
    print(f"Size of dataset {x.shape[0]} samples with input shape {x.shape} and target shape {y.shape}")


    def print_stats(tensor, names, title):
        print(f"\n{'='*85}")
        print(f"{title:^85}")
        print(f"{'='*85}")
        print(f"{'CHANNEL':<20} | {'MEAN':<10} | {'STD':<10} | {'MIN':<10} | {'MAX':<10}")
        print(f"{'-'*85}")
    
        # Calculating across Batch (0), Height (2), and Width (3)
        # We flatten (0, 2, 3) to easily get global min/max per channel
        for i, name in enumerate(names):
            channel_data = tensor[:, i, :, :]
            
            mean = channel_data.mean().item()
            std = channel_data.std().item()
            c_min = channel_data.min().item()
            c_max = channel_data.max().item()
    
            print(f"{name:<20} | {mean:>10.4f} | {std:>10.4f} | {c_min:>10.4f} | {c_max:>10.4f}")
    
    print_stats(x, input_names, "INPUT CALIBRATION (x)")
    print_stats(y, output_names, "TARGET CALIBRATION (y)")

# Define a custom dataset class that includes properties dictionary with sample information
class TensorDatasetWithProps(Dataset):
    def __init__(self, x, y, props, transform_x=None, transform_y=None):
        assert x.size(0) == y.size(0), "Size mismatch between tensors"
        assert x.size(0) == len(props), "Size mismatch between tensors and properties dict"
        self.x = x
        self.y = y
        self.props = props
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        props = self.props[index]
        if self.transform_x is not None:
            x = self.transform_x(x)

        if self.transform_y is not None:
            y = self.transform_y(y)

        return {"x": x, "y": y , "props": props}    

    def __len__(self):
        return self.x.size(0)
    

class AirfransDataset(PTDataset):
    def __init__(
        self,
        data_dir: Union[Path, str],
        dataset_name: str = 'airfoil',
        train_split: str = 'scarce_train',
        test_splits: List[str] = ['full_test','aoa_test'],
        batch_size: int = 16,
        test_batch_sizes: List[int] = [32, 32],
        train_resolution: tuple = (256,32),
        test_resolutions: List[tuple] = [(256,32),(512,64)],
        xlim: float = 6.0,
        ylim: float = 3.0,
        encode_input: bool = True,   # normalize u,v,sdf,log_nu but NOT mask
        encode_output: bool = True,  # normalize u,v,p,log_nut_ratio
        encoding: str = "channel-wise",
        channel_dim: int = 1,
        channels_squeezed: bool = False,  # Our data already has explicit channel dims
    ):
        """
        Initialize AirfoilDataset with AirFRANS-specific parameters.
        
        Args:
            train_split: Split name for training ('full_train', 'scarce_train', 'aoa_train', 'reynolds_train')
            test_splits: List of split names for testing ( 'full_test', 'aoa_test', 'reynolds_test')
            xlim, ylim: Domain size parameters
            ... other standard PTDataset parameters
        """
        
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        
        # Load manifest to get actual sample counts
        manifest_path = data_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
        else:
            raise FileNotFoundError(f"Manifest file not found at {manifest_path}")
        
        # Calculate actual sample counts from manifest
        train_foil_name = manifest_data.get(train_split, [])
        n_train = len([d for d in train_foil_name ])
        
        n_tests = []
        for test_split in test_splits:
            test_foil_name = manifest_data.get(test_split, [])
            n_test = len([d for d in test_foil_name ])
            n_tests.append(n_test)
        
        # Store AirFRANS-specific parameters
        self.train_split = train_split
        self.test_splits = test_splits
        self.xlim = xlim
        self.ylim = ylim
        
        # Store dataloader properties (from PTDataset)
        self.batch_size = batch_size
        self.test_resolutions = test_resolutions
        self.test_batch_sizes = test_batch_sizes
        
        # Load training data with custom filename
        train_file = data_dir / "train"/ f"{dataset_name}_{train_split}_CMesh_{train_resolution[0]}x{train_resolution[1]}.pt"
        train_data = torch.load(train_file.as_posix(),weights_only=False)
        
        x_train =train_data["x"]
        # x channels
        # 0  = x
        # 1  = y
        # 2  = U_x_pot
        # 3  = U_y_pot
        # 4  = Cp_pot
        # 5  = sdf    # To be dropped
        # 6  = exp_sdf
        # 7  = x_xi  => 6
        # 8  = x_eta => 7
        # 9  = y_xi => 8
        # 10 = y_eta => 9
        # 11 = det_J = 10

        # y channels
        # 0 = Cp_delta
        # 1 = U_x_delta
        # 2 = U_y_delta
        # 3 = log_nut_ratio

        # drop the sdf channel
        x_train = torch.cat([x_train[:, :5, :, :], x_train[:, 6:, :, :]], dim=1) 
        # modify the input channels to be in a reasonable range before normalization 
        ##x_train[:,7:9, :, :] = torch.arcsinh(x_train[:, 7:9, :, :] ) 
        # modify jacobian channel to log(det_J) to keep values in a reasonable range for normalization
        ##x_train[:, 10, :, :] = torch.log(x_train[:, 10, :, :] + 1e-8)

        y_train = train_data["y"].clone()

        y_train = y_train[:, :, :, :]  # keep all channels

        # ... loading x_train, y_train ...

        if 'props' not in train_data:
            raise KeyError(f"CRITICAL ERROR: 'props' not found in {train_file}. "
                           "Cannot calculate aerodynamic coefficients without metadata.")

        raw_props = train_data['props']
        
        # Ensure self.train_props is always a LIST of dictionaries
        # This makes indexing self.train_props[idx] consistent later
        if isinstance(raw_props, dict) and 'foil_coords' in raw_props:
            # If it's a 'flipped' dictionary (dict of lists/tensors), un-flip it
            print("Detected collated dictionary in archive. Un-flipping...")
            self.train_props = [
                {k: v[i] for k, v in raw_props.items()} 
                for i in range(x_train.size(0))
            ]
        elif isinstance(raw_props, list):
            self.train_props = raw_props
        else:
            raise TypeError(f"Unknown props format in {train_file}: {type(raw_props)}")

        # Validation Print
        print(f"✅ Successfully loaded {len(self.train_props)} property entries.")
        #if 'airfoil_points' not in self.train_props[0]:
        #    raise KeyError(f"Props found, but 'airfoil_points' is missing in {train_file}")
        print(f"Loading train db for {train_split} resolution {train_resolution} with {n_train} samples")

        print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

        print(f"Input channels : {x_train.shape[1]}")
        print(f"Output channels: {y_train.shape[1]}")

        del train_data
        
        # Fit optional encoders to train data (from PTDataset logic)
        # For inputs: normalize u_input, v_input, sdf_linear, log_nu but NOT mask_binary (channel 2)
        if encode_input:
            if encoding == "channel-wise":
                reduce_dims = list(range(x_train.ndim))
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]
            
            # Create separate normalizers for input and output
            num_channels = x_train.shape[channel_dim]  # This should be the number of channels in x_train
            # mask exp_sdf channel ( not normalizing channel 5 which is exp_sdf)
            mask_channels = [5]  # This is the index of the exp_sdf channel in the original data (before dropping sda
            print(f"Creating SelectiveUnitGaussianNormalizer for input with {num_channels} channels, masking channel {mask_channels}")
            input_encoder = SelectiveUnitGaussianNormalizer(dim=reduce_dims, num_channels=num_channels, eps=1e-5, mask_channels=mask_channels)
            # feed whole x_train, the normalizer will handle selective normalization
            input_encoder.fit(x_train)
            # force the mask channel stats to 0 mean, 1 std
            input_encoder.mean[:, input_encoder.mask_channels, ...] = 0.0
            input_encoder.std[:, input_encoder.mask_channels, ...] = 1.0
            print(f" Channels to normalize {input_encoder.channels_to_normalize}")
            print(f"✓ Mask channel {input_encoder.mask_channels} will be passed through.")
            print(f"Input Normalizer mean shape: {input_encoder.mean.shape}, std shape: {input_encoder.std.shape}")
            print(f"Input encoder mean: {input_encoder.mean}")
            print(f"Input encoder std: {input_encoder.std}")
            print(f"Input encoder mask channel: {input_encoder.mask_channels}")
        else:
            input_encoder = None

        if encode_output:
            if encoding == "channel-wise":
                reduce_dims = list(range(y_train.ndim))
                reduce_dims.pop(channel_dim)
            elif encoding == "pixel-wise":
                reduce_dims = [0]
            output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
            output_encoder.fit(y_train)
        else:
            output_encoder = None
        
        # Create train dataset
        self._train_db = TensorDatasetWithProps(x_train, y_train, props=self.train_props)

        # create a Domain padder fully symmetric in radial direction 10% extra in tangential

        
        # Create custom data processor that handles selective input normalization
        self._data_processor = AirfransDataProcessor_Linear_Jacobian(
            in_normalizer=input_encoder, 
            out_normalizer=output_encoder,
            xi_pad_frac=0.1
        )
        self._evaluator = AirfoilEvaluator(processor=self._data_processor)
        # Load test data with custom filenames
        self._test_dbs = {}
        for (res_xi,res_eta), n_test, test_split in zip(test_resolutions, n_tests, test_splits):
            print(f"Loading test db for {test_split} resolution {res_xi}x{res_eta} with {n_test} samples")
            
            test_file = data_dir / "test" / f"{dataset_name}_{test_split}_CMesh_{res_xi}x{res_eta}.pt"
            test_data = torch.load(test_file.as_posix())
            
            x_test = test_data["x"].type(torch.float32).clone()

            # drop the sdf channel
            x_test = torch.cat([x_test[:, :5, :, :], x_test[:, 6:, :, :]], dim=1) 
            # modify the input channels to be in a reasonable range before normalization 
            x_test[:,7:9, :, :] = torch.arcsinh(x_test[:, 7:9, :, :] ) 
            # modify jacobian channel to log(det_J) to keep values in a reasonable range for normalization
            x_test[:, 10, :, :] = torch.log(x_test[:, 10, :, :] + 1e-8)

            y_test = test_data["y"].clone()
            y_test = y_test[:, :, :, :]  # keep all channels

            props = test_data.get('props', None)
            # add empty props if none found
            if props is None:
                props = [{} for _ in range(x_test.size(0))]
            

            test_db = TensorDatasetWithProps(x_test, y_test,props)
            self._test_dbs[(res_xi, res_eta)] = test_db

            if 'props' in test_data:
                setattr(self, f"{test_split}_props", test_data['props'])
            else:
                setattr(self, f"{test_split}_props", None)
            
            del test_data

def load_airfrans_dataset(
    data_dir,
    dataset_name,    
    train_split,
    test_splits,
    batch_size,
    test_batch_sizes,
    train_resolution=(256,32),
    test_resolutions=[(128,32), (256,32)],
    encode_input=True,
    encode_output=True,
    encoding="channel-wise",
    channel_dim=1,
):
    
    collate_with_props_fn = collate_batch_with_props

    dataset = AirfransDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        train_split=train_split,
        test_splits=test_splits,
        batch_size=batch_size,
        test_batch_sizes=test_batch_sizes,
        train_resolution=train_resolution,
        test_resolutions=test_resolutions,
        encode_input=encode_input,
        encode_output=encode_output,
        channel_dim=channel_dim,
        encoding=encoding,
    )

    # return dataloaders for backwards compat
    train_loader = DataLoader(
        dataset.train_db,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        collate_fn= collate_with_props_fn,
        persistent_workers=False,
    )

    test_loaders = {}
    for (res_xi, res_eta), test_bsize in zip(test_resolutions, test_batch_sizes):
        print(f"Creating DataLoader for test resolution {res_xi}x{res_eta} with batch size {test_bsize}")
        test_loaders[(res_xi, res_eta)] = DataLoader(
            dataset.test_dbs[(res_xi, res_eta)],
            batch_size=test_bsize,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            collate_fn=collate_with_props_fn,
            persistent_workers=False,
        )

    return train_loader, test_loaders, dataset.data_processor,
