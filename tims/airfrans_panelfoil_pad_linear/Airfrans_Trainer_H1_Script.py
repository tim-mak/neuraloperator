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
from tims.airfrans_panelfoil_pad_linear.WeightedFieldwiseAggregatorLoss import WeightedFieldwiseAggregatorLoss
from tims.airfrans_panelfoil_pad_linear.Airfrans_Delta_Linear_Jacobian_Dataset import load_airfrans_dataset, get_dataset_stats, verify_input_encoder, verify_output_encoder
from tims.airfrans_panelfoil_pad_linear.Airfrans_Delta_Config_H1_Mirror_Y import Default
from tims.airfrans_panelfoil_pad_linear.Airfrans_Delta_Trainer import AirfransDeltaTrainer
from tims.airfrans_panelfoil_pad_linear.fnoCustomPadding import FNOCustomPadding
from tims.airfrans_panelfoil_pad_linear.MirrorPadding import MirrorPaddingY

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



def plot_convergence(csv_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['u_err'], label='U-Velocity Error %')
    plt.plot(df['epoch'], df['v_err'], label='V-Velocity Error %')
    #plt.plot(df['epoch'], df['cp_err'], label='Cp Error %')
    plt.yscale('log') # Log scale is best for observing convergence plateaus
    plt.xlabel('Epoch')
    plt.ylabel('Relative Error (%)')
    plt.title('Channel-Wise Convergence Audit')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()




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

# Set up WandB logging
wandb_args = None
if config.wandb.log and is_logger:
    try:
        # Try to login with existing credentials first
        wandb.login()
    except Exception:
        # Fallback to API key file if available
        try:
            wandb.login(key=get_wandb_api_key())
        except Exception as e:
            print(f"Warning: Could not log into WandB: {e}")
            print("Continuing without WandB logging...")
            config.wandb.log = False
    if config.wandb.log and is_logger:
        if config.wandb.name:
            wandb_name = config.wandb.name
        else:
            wandb_name = "_".join(
                f"{var}"
                for var in [
                    config.model.model_arch,
                    config.model.n_layers,
                    config.model.n_modes,
                    config.model.hidden_channels,
                ]
            )

        wandb_args = dict(
            config=config,
            name=wandb_name,
            group=config.wandb.group,
            project=config.wandb.project,
            entity=config.wandb.entity,
        )
        if config.wandb.sweep:
            for key in wandb.config.keys():
                config.params[key] = wandb.config[key]
        wandb.init(**wandb_args)

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
get_dataset_stats(Path(config.data.data_dir) / "train" / f"{config.data.dataset_name}_{config.data.train_split}_CMesh_{config.data.train_resolution[0]}x{config.data.train_resolution[1]}.pt")

# check data_processor
print(" =" * 80)
verify_input_encoder(data_processor.in_normalizer)

verify_output_encoder(data_processor.out_normalizer)
print(" =" * 80)


loss_mappings = {
    'Cp_delta': 0,
    'U_x_delta': 1,
    'U_y_delta': 2,
    'log_nutratio': 3
}

# 2. The Specific Loss Algorithms
# loss_functions = {
#     'Cp_delta': LpLoss(d=2, p=2, reduction="mean"),   # 2D spatial domain, L2 norm
#     'U_x_delta': LpLoss(d=2, p=2, reduction="mean"),
#     'U_y_delta': LpLoss(d=2, p=2, reduction="mean"),
#     'log_nutratio': nn.MSELoss()         # Standard point-wise mean squared error
# }
# switch to Pure H1 Loss to capture both value and gradient errors, which are crucial for fluid dynamics
loss_functions = {
    'Cp_delta': H1Loss(d=2, reduction="mean", periodic_in_x=False, periodic_in_y=False),   # 2D spatial domain, 
    'U_x_delta': H1Loss(d=2, reduction="mean", periodic_in_x=False, periodic_in_y=False),
    'U_y_delta': H1Loss(d=2, reduction="mean",periodic_in_x=False, periodic_in_y=False),
    'log_nutratio': nn.L1Loss(reduction="mean")        
}

print(f"Weights for Weighted Loss: {config.data.weights}")
# 3. The Multi-Task Weights
# Velocity dictates pressure and turbulence. Give velocity higher initial priority
# to force the network to learn the primary kinematics first.
loss_weights = {
    'Cp_delta': config.data.weights[0],  # e.g., 1.0
    'U_x_delta': config.data.weights[1],
    'U_y_delta': config.data.weights[2],
    'log_nutratio': config.data.weights[3],
}

# Instantiate the final loss object
train_loss_fn = WeightedFieldwiseAggregatorLoss(
    losses=loss_functions, 
    mappings=loss_mappings, 
    weights=loss_weights,
    logging_enabled=True
)


# MAE (L1) gives you literal physical error (e.g., "off by 0.5 m/s")
mae_functions = {
    'Cp_delta': nn.L1Loss(),
    'U_x_delta': nn.L1Loss(),
    'U_y_delta': nn.L1Loss(),
    'log_nutratio': nn.L1Loss()
}

# H1 Loss measures the error of the values AND their spatial gradients (shear/vorticity)
h1_functions = {
    'Cp_delta': H1Loss(d=2),
    'U_x_delta': H1Loss(d=2),
    'U_y_delta': H1Loss(d=2),
    'log_nutratio': H1Loss(d=2) # Or leave as MSE if H1 is too heavy for turbulence
}


# Evaluation losses with no logging to avoid the tuple error
eval_lp_loss_object = WeightedFieldwiseAggregatorLoss( loss_functions,
                                                   loss_mappings,
                                                     loss_weights, 
                                                     logging_enabled=False)

eval_mae_loss_object = WeightedFieldwiseAggregatorLoss( mae_functions,
                                                   loss_mappings,
                                                     loss_weights, 
                                                     logging_enabled=False)

eval_h1_loss_object = WeightedFieldwiseAggregatorLoss( h1_functions,
                                                   loss_mappings,
                                                     loss_weights, 
                                                     logging_enabled=False)


eval_losses = {"Weighted_Relative_Lp2": eval_lp_loss_object,
               "Weighted_Absolute_MAE" : eval_mae_loss_object,
               "Weighted_Physics_H1" : eval_h1_loss_object}


# Model initialization
#model = get_model(config)

model = FNOCustomPadding(
    n_modes=config.model.n_modes,
    in_channels=config.model.data_channels,  # +2 for grid embedding (X, Y)
    out_channels=config.model.out_channels,
    hidden_channels=config.model.hidden_channels,
    lifting_channel_ratio=config.model.lifting_channel_ratio,
    projection_channel_ratio=config.model.projection_channel_ratio,
    n_layers=config.model.n_layers,
    use_channel_mlp=config.model.use_channel_mlp,
    channel_mlp_expansion=config.model.channel_mlp_expansion,
    stabilizer=None,
    domain_padding=MirrorPaddingY(),
)


# Move model to device
model = model.to(device)
print(f"Model moved to device: {device}")
# Move data processor to device if it has normalizers
if data_processor is not None:
    data_processor = data_processor.to(device)


# Create the optimizer
optimizer = AdamW(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)

if config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        mode="min",
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.opt.scheduler_T_max
    )
elif config.opt.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
else:
    raise ValueError(f"Got scheduler={config.opt.scheduler}")



if config.verbose and is_logger:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_loss_fn}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n Log Training Loss: {config.wandb.log_output}")
    print("\n### Log Verbose ###\n", config.verbose)
    
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()

print("Starting training loop...")
# Log model parameter count
if is_logger:
    n_params = count_model_params(model)

    if config.verbose:
        print(f"\nModel n_params: {n_params}")
        sys.stdout.flush()

    if config.wandb.log:
        to_log = {"n_params": n_params}
        if config.n_params_baseline is not None:
            to_log["n_params_baseline"] = (config.n_params_baseline,)
            to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
            to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
        wandb.log(to_log, commit=False)
        wandb.watch(model)


## --- Define the outer loop here ---
best_test_loss = float('inf')
improvement_threshold = 0.01 

model_save_dir = config.data.model_storage_dir

checkpoint_dir = Path(model_save_dir) / "checkpoints"
if checkpoint_dir.exists():
    print(f"Checkpoint directory {checkpoint_dir} already exists. Checkpointing will overwrite existing files.")
else:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)   
# Ensure the trainer is set to 1 epoch internally
#trainer.n_epochs = 1
output_dir = Path(model_save_dir) / "results"



trainer = AirfransDeltaTrainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    data_processor=data_processor,
    device=device,
    mixed_precision=config.opt.mixed_precision,
    eval_interval=config.opt.eval_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose,
    wandb_log=config.wandb.log,
)


# Log model parameter count
if is_logger:
    n_params = count_model_params(model)

    if config.verbose:
        print(f"\nModel n_params: {n_params}")
        sys.stdout.flush()

    if config.wandb.log:
        to_log = {"n_params": n_params}
        if config.n_params_baseline is not None:
            to_log["n_params_baseline"] = (config.n_params_baseline,)
            to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
            to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
        wandb.log(to_log, commit=False)
        wandb.watch(model)

# Start training process
trainer.train(
    train_loader,
    test_loaders,
    optimizer,
    scheduler,
    regularizer=False,
    training_loss=train_loss_fn,
    eval_losses=eval_losses,
    save_every=20,
    #save_best='128_weightedField',  # Save based on weighted L2 at 128 res
    save_dir=checkpoint_dir,
    sample_idx=13,  # Consistent sample for diagnostic plots
)

# Finalize WandB logging
if config.wandb.log and is_logger:
    wandb.finish()