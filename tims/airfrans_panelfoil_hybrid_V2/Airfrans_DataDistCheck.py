import csv
from pathlib import Path
import sys
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader
import wandb
import pyvista as pv
from neuralop import H1Loss, LpLoss, Trainer, get_model
from neuralop.training import setup, AdamW
from neuralop.mpu.comm import get_local_rank
from neuralop.utils import get_wandb_api_key, count_model_params    
from tims.airfrans_panelfoil_hybrid_V2.HybridChannelWiseSpectralPadder import HybridChannelWiseSpectralPadder
from tims.airfrans_panelfoil_hybrid_V2.WeightedFieldwiseAggregatorLoss import WeightedFieldwiseAggregatorLoss
from tims.airfrans_panelfoil_hybrid_V2.Airfrans_Delta_Dataset_V2 import load_airfrans_dataset, get_dataset_stats, verify_input_encoder, verify_output_encoder, plot_dataset_distributions
from tims.airfrans_panelfoil_hybrid_V2.Airfrans_Delta_Config_V2 import Default
from tims.airfrans_panelfoil_hybrid_V2.Airfrans_Delta_Trainer import AirfransDeltaTrainer
from tims.airfrans_panelfoil_hybrid_V2.fnoCustomPadding import FNOCustomPadding
from tims.airfrans_panelfoil_hybrid_V2.SobolevLoss import SobolevLoss

from zencfg import make_config_from_cli
import pandas as pd

from neuralop.losses import H1Loss

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.serialization.add_safe_globals([np._core.multiarray.scalar,
                                    np.dtype, 
                                    np.dtypes.Float64DType,
                                    np.dtypes.Float32DType,
                                    np._core.multiarray._reconstruct, 
                                    pv.core.pyvista_ndarray])




def save_checkpoint(checkpoint_dir, model, data_processor, epoch, filename):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Handle DDP: get the internal module
    unwrapped_model = model.module if hasattr(model, 'module') else model
    
    torch.save({
        'epoch': epoch,
        'model': unwrapped_model.state_dict(),
        'data_processor': data_processor.state_dict(),
        'config': config
    }, checkpoint_dir / filename)


config = make_config_from_cli(Default)
config = config.to_dict()

# Distributed training setup, if enabled
device, is_logger = setup(config)


# Make sure we only print information when needed
config.verbose = config.verbose and is_logger

# Print configuration details
if config.verbose and is_logger:
    print(f"##### CONFIG #####\n")
    print(config)
    sys.stdout.flush()

# Load the Airfrans dataset
data_dir = Path(config.data.data_dir).expanduser()

train_loader, test_loaders, data_processor = load_airfrans_dataset(
        data_dir = config.data.data_dir,
        dataset_name = config.data.dataset_name,
        train_split = config.data.train_split,
        test_splits = config.data.test_splits,
        batch_size = config.data.batch_size,
        test_batch_sizes = config.data.test_batch_sizes,
        test_resolutions = config.data.test_resolutions,
        encode_input = config.data.encode_input,    
        encode_output = config.data.encode_output, 
        encoding = config.data.encoding,
        channel_dim = config.data.channel_dim,
    )

# check dataset stats
print(" =" * 80)
print("\nDataset Statistics Audit:")
get_dataset_stats(Path(config.data.data_dir) / "train" / f"{config.data.dataset_name}_{config.data.train_split}_CMesh_V2_{config.data.train_resolution[0]}x{config.data.train_resolution[1]}.pt")

distPath = Path(config.data.data_dir) / "train" / f"Dist_Hybrid_{config.data.dataset_name}_{config.data.train_split}_CMesh_V2_{config.data.train_resolution[0]}x{config.data.train_resolution[1]}"
plot_dataset_distributions(Path(config.data.data_dir) / "train" / f"{config.data.dataset_name}_{config.data.train_split}_CMesh_V2_{config.data.train_resolution[0]}x{config.data.train_resolution[1]}.pt", save_dir=distPath)

# check data_processor
print(" =" * 80)
#verify_input_encoder(data_processor.in_normalizer)

#verify_output_encoder(data_processor.out_normalizer)
print(" =" * 80)
