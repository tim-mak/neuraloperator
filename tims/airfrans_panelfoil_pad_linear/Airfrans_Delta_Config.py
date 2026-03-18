from typing import Any, List, Optional

from zencfg import ConfigBase
from config.distributed import DistributedConfig
from config.models import FNOConfig, ModelConfig, FNO_Small2d
from config.opt import OptimizationConfig, PatchingConfig
from config.wandb import WandbConfig


class AirfransOptConfig(OptimizationConfig):
    n_epochs: int = 1000  # check cli arguments
    learning_rate: float = 5e-4
    training_loss: str = "weighted_l2"
    weight_decay: float = 1e-4
    scheduler: str = "CosineAnnealingLR"
    scheduler_T_max: int = 1000 
    mixed_precision: bool = True  
    save_interval: int = 20       
    step_size: int = 60
    gamma: float = 0.5
    eval_interval: int = 20


class AirfransDatasetConfig(ConfigBase):
    data_dir: str = "/home/timm/storage/AF_NO_DATASET"
    dataset_name: str = "airfoil"
    batch_size: int = 64
    train_split: str = "full_train"
    train_resolution: tuple = (256, 32)
    test_splits: List[str] = ["full_test", "full_test"]
    test_resolutions: List[tuple] = [(256, 32), (256, 32)]
    test_batch_sizes: List[int] = [64, 64]
    encode_input: bool = True
    encode_output: bool = True
    encoding: str = "channel-wise"
    channel_dim: int = 1
    weights: List[float] = [1.0, 1.0, 1.0, 1.0]  # Weights for [delta_Cp, delta_U_x, delta_U_y, log_nut_ratio] in loss calculation
    model_storage_dir: str = "/home/timm/storage/AF_NO_DATASET/Model_FNO_pad_linear_Modes_32_8_V1"

class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    arch: str = "fno"
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = FNOConfig(
        data_channels=11,    #  names =[ "X", "Y", "U_x", "U_y", "Cp_pot", "exp_sdf","x_xi","x_eta","y_xi","y_eta","det_J"]
        out_channels=4,     # [delta_Cp, delta_U_x, delta_U_y, log_nut_ratio]
        n_modes=[64,16],   # Need to double the modes in each direction to account for the padding (256,32) -> (512,64)
        hidden_channels=32,
        lifting_channel_ratio=1,
        projection_channel_ratio=1,
        n_layers=3,
        use_channel_mlp=True,
        channel_mlp_expansion=1,
        stabilizer="None",
        domain_padding=[0.5, 0.5]  # 50% padding each side: (256,32) -> (512,64) powers of 2 for cuFFT half precision

    )
    device: str = "cuda"
    opt: OptimizationConfig = AirfransOptConfig()
    data: AirfransDatasetConfig = AirfransDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    save_interval: int = 50
    wandb: WandbConfig = WandbConfig()