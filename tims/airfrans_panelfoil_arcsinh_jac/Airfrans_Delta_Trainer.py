from tims.airfrans_panelfoil_arcsinh_jac.Airfrans_Evaluator import AirfoilEvaluator
from neuralop.training.trainer import Trainer
from typing import Union
from pathlib import Path
from neuralop.losses import LpLoss
from torch import nn
import os
import torch
import wandb
import matplotlib.pyplot as plt
import warnings
import torch.distributed as dist
import sys
from timeit import default_timer
import numpy as np

class AirfransDeltaTrainer(Trainer):

    def __init__(self, *, model, n_epochs, wandb_log=False, device="cpu", mixed_precision=False, data_processor=None, eval_interval=1, log_output=False, use_distributed=False, verbose=False):
        self.evaluator = AirfoilEvaluator(processor=data_processor, device=device)
        super().__init__(model=model, n_epochs=n_epochs, wandb_log=wandb_log, device=device, mixed_precision=mixed_precision, data_processor=data_processor, eval_interval=eval_interval, log_output=log_output, use_distributed=use_distributed, verbose=verbose)

    def eval_one_batch(self, sample: dict, eval_losses: dict, return_output: bool = False):
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            sample = {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}
        
        self.n_samples += sample["y"].size(0)
        out = self.model(**sample)

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)
        
        loss_sample = {k: v for k, v in sample.items() if torch.is_tensor(v) or k == 'y'}
        eval_step_losses = {}

        for loss_name, loss_fn in eval_losses.items():
            y_target = loss_sample.get('y', None)
            loss_kwargs = {k: v for k, v in loss_sample.items() if k != 'y'}
            
            if y_target is not None:
                res = loss_fn(out, y_target, **loss_kwargs)
            else:
                res = loss_fn(out, **loss_sample)

            if isinstance(res, tuple):
                val_loss_out = res[0]
            else:
                val_loss_out = res

            eval_step_losses[loss_name] = val_loss_out.detach().item()

        if return_output:
            if isinstance(out, dict):
                out = out.get('y')
            return eval_step_losses, out
        else:
            return eval_step_losses, None

    def evaluate(self, loss_dict, data_loader, log_prefix="", mode="single_step", **kwargs):
        self.model.eval()
        if self.data_processor:
            self.data_processor.eval()

        errors = {f"{log_prefix}_{loss_name}": 0.0 for loss_name in loss_dict.keys()}
        self.n_samples = 0
        local_n_samples = 0

        with torch.no_grad():
            for idx, sample in enumerate(data_loader):
                eval_step_losses, outs = self.eval_one_batch(
                    sample, loss_dict, return_output=(idx == 0)
                )
                batch_size = sample['y'].size(0)
                local_n_samples += batch_size

                for loss_name, val_loss in eval_step_losses.items():
                    if torch.is_tensor(val_loss):
                        errors[f"{log_prefix}_{loss_name}"] += batch_size * val_loss.item()
                    else:
                        errors[f"{log_prefix}_{loss_name}"] += batch_size * val_loss

        for key in errors.keys():
            if local_n_samples > 0:
                errors[key] /= local_n_samples
                print(f"Eval {key}: {errors[key]:.6f} over {local_n_samples} samples")
            else:
                print("Warning: local_n_samples is 0 during evaluation. Check data loader.")

        return errors

    def train(self, train_loader, test_loaders, optimizer, scheduler, regularizer=None, training_loss=None, eval_losses=None, eval_modes=None, save_every: int = None, save_best: int = None, save_dir: Union[str, Path] = "./ckpt", resume_from_dir: Union[str, Path] = None, max_autoregressive_steps: int = None, sample_idx: int = 0):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.regularizer = regularizer if regularizer else None

        if training_loss is None:
            training_loss = LpLoss(d=2)

        if hasattr(training_loss, "reduction") and training_loss.reduction == "mean":
            warnings.warn(f"{training_loss.reduction=}. The Trainer expects losses to sum across the batch dim.")

        if eval_losses is None:
            eval_losses = dict(l2=training_loss)

        self.wandb_epoch_metrics = None
        eval_modes = eval_modes or {}
        self.save_every = save_every
        self.save_best = save_best

        if resume_from_dir is not None:
            self.resume_state_from_dir(resume_from_dir)

        self.model = self.model.to(self.device)

        if self.data_processor is not None:
            self.data_processor = self.data_processor.to(self.device)
            self.data_processor.train() 
            
            if not self.use_distributed or (dist.is_initialized() and dist.get_rank() == 0):
                save_dir.mkdir(parents=True, exist_ok=True)
                self.save_dir = Path(save_dir)
                torch.save(self.data_processor.state_dict(), self.save_dir / "data_processor.pt")
                if self.verbose:
                    print(f"✅ DataProcessor locked and saved to {save_dir}/data_processor.pt")
        else:
            if self.verbose:
                print("⚠️ No DataProcessor provided; ensure data is preprocessed appropriately.")

        if self.save_best is not None:
            metrics = [f"{name}_{metric}" for name in test_loaders.keys() for metric in eval_losses.keys()]
            if self.save_best not in metrics:
                raise AssertionError(f"Error: 'save_best' metric '{self.save_best}' not found. Available: {metrics}")
            best_metric_value = float("inf")
            self.save_every = None

        if self.verbose:
            print(f"Training on {len(train_loader.dataset)} samples")
            print(f"Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples on resolutions {[name for name in test_loaders]}.")
            sys.stdout.flush()

        for epoch in range(self.start_epoch, self.n_epochs):
            train_err, avg_loss, avg_lasso_loss, epoch_train_time = self.train_one_epoch(epoch, train_loader, training_loss)
            epoch_metrics = dict(train_err=train_err, avg_loss=avg_loss, avg_lasso_loss=avg_lasso_loss, epoch_train_time=epoch_train_time)

            if epoch % self.eval_interval == 0:
                eval_metrics = self.evaluate_all(epoch=epoch, eval_losses=eval_losses, test_loaders=test_loaders, eval_modes=eval_modes, max_autoregressive_steps=max_autoregressive_steps)
                epoch_metrics.update(**eval_metrics)
                
                if save_best is not None and eval_metrics[save_best] < best_metric_value:
                    best_metric_value = eval_metrics[save_best]
                    self.checkpoint(save_dir)
                
                self.plot_diagnostic_grid(train_loader, epoch, save_dir=save_dir, sample_idx=sample_idx, training_loss=training_loss)

            if self.save_every is not None and epoch % self.save_every == 0:
                self.checkpoint(save_dir)

        return epoch_metrics

    def train_one_epoch(self, epoch, train_loader, training_loss):
        self.on_epoch_start(epoch)
        avg_loss = 0
        avg_lasso_loss = 0
        self.model.train()
        if self.data_processor:
            self.data_processor.train()
        
        t1 = default_timer()
        train_err = 0.0
        self.n_samples = 0
        n_batches = len(train_loader)

        # Dynamic channel metric tracking
        epoch_channel_metrics = {}

        for idx, sample in enumerate(train_loader):
            loss, loss_per_channel = self.train_one_batch(idx, sample, training_loss)
            loss.backward()
            self.optimizer.step()

            train_err += loss.item()
            with torch.no_grad():
                avg_loss += loss.item()
                if self.regularizer:
                    avg_lasso_loss += self.regularizer.loss

                # DYNAMIC ACCUMULATION
                if isinstance(loss_per_channel, dict):
                    for key, val in loss_per_channel.items():
                        epoch_channel_metrics[key] = epoch_channel_metrics.get(key, 0.0) + val
                elif isinstance(loss_per_channel, (list, torch.Tensor)):
                    for c, c_loss in enumerate(loss_per_channel):
                        key = f"channel_{c}_loss"
                        epoch_channel_metrics[key] = epoch_channel_metrics.get(key, 0.0) + c_loss.item()

                if (epoch % self.eval_interval == 0) and (idx == 0):
                    self.model.eval() 
                    with torch.no_grad():
                        x = sample['x'].to(self.device)
                        y_pred = self.model(x)
                        tensor_sample = {k: v for k, v in sample.items() if isinstance(v, torch.Tensor)}
                        y_decoded, _ = self.data_processor.postprocess(y_pred, tensor_sample)
                    self.model.train() 

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(train_err)
        else:
            self.scheduler.step()

        epoch_train_time = default_timer() - t1
        train_err /= n_batches
        avg_loss /= self.n_samples

        # Average dynamic metrics
        channel_metrics = {k: v / n_batches for k, v in epoch_channel_metrics.items()}
        avg_lasso_loss = (avg_lasso_loss / self.n_samples) if self.regularizer else None
        
        lr = next((pg["lr"] for pg in self.optimizer.param_groups), None)

        if self.verbose and epoch % self.eval_interval == 0:
            self.log_training(
                epoch=epoch,
                time=epoch_train_time,
                avg_loss=avg_loss,
                train_err=train_err,
                channel_metrics=channel_metrics,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr,
            )

        return train_err, avg_loss, avg_lasso_loss, epoch_train_time

    def train_one_batch(self, idx, sample, training_loss):
        self.optimizer.zero_grad(set_to_none=True)
        if self.regularizer:
            self.regularizer.reset()
            
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            sample = {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}

        self.n_samples += sample["y"].shape[0] if isinstance(sample["y"], torch.Tensor) else 1

        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                out = self.model(**sample)
        else:
            out = self.model(**sample)

        loss = 0.0
        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                loss_out = training_loss(out, **sample)
        else:
            loss_out = training_loss(out, **sample)

        if isinstance(loss_out, tuple):
            batch_loss, channel_batch_losses = loss_out
        else:   
            batch_loss, channel_batch_losses = loss_out, {}

        loss += batch_loss
        if self.regularizer:
            loss += self.regularizer.loss

        return loss, channel_batch_losses
        
    def log_training(self, epoch: int, time: float, avg_loss: float, train_err: float, channel_metrics: dict = None, avg_lasso_loss: float = None, lr: float = None):
        channel_metrics = channel_metrics or {}
        
        if self.log_output:
            if self.wandb_log:
                values_to_log = dict(train_err=train_err, time=time, avg_loss=avg_loss, avg_lasso_loss=avg_lasso_loss, lr=lr, **channel_metrics)
                wandb.log(data=values_to_log, step=epoch + 1, commit=False)
            
            log_path = self.save_dir / "loss_history.csv"
            keys = list(channel_metrics.keys())
            
            # Write dynamic header
            if not log_path.exists():
                with open(log_path, 'w') as f:
                    header_str = "epoch,time,train_err," + ",".join(keys) + ",lr\n"
                    f.write(header_str)
            
            # Write dynamic values
            with open(log_path, 'a') as f:
                vals_str = ",".join([f"{channel_metrics[k]:.6f}" for k in keys])
                f.write(f"{epoch},{time:.2f},{train_err:.6f},{vals_str},{lr:.2e}\n")

            # Dynamic Console Output
            msg = f"[{epoch}] time={time:.2f}, avg_loss={avg_loss:.4f}, train_err={train_err:.4f}"
            for k in keys:
                msg += f", {k}={channel_metrics[k]:.4f}"
            print(msg)

        if not self.regularizer and not self.log_output:
            print(f"Logging disabled on {self.device} not logging metrics.")

    def plot_diagnostic_grid(self, loader, epoch, save_dir="plots", sample_idx=0, prefix="prediction", training_loss=None):
            # Prevent secondary GPUs from overlapping file writes
            if self.use_distributed and dist.is_initialized() and dist.get_rank() != 0:
                return

            self.model.eval()
            if self.data_processor:
                self.data_processor.eval()
            
            batch = next(iter(loader))
            target_losses = training_loss.losses
            target_weights = training_loss.weights      

            with torch.no_grad():
                x_raw = batch['x'][sample_idx:sample_idx+1].to(self.device)
                y_raw = batch['y'][sample_idx:sample_idx+1].to(self.device)
                
                sample = self.data_processor.preprocess({'x': x_raw, 'y': y_raw})
                x_input = sample['x'].to(self.device)

                # ---- 1. ATTACH THE WIRETAP ----
                hook_handles, fft_storage = self._register_fft_hook()
                
                # ---- 2. FORWARD PASS ----
                y_norm_pred = self.model(x_input)
                
                # ---- 3. DETACH THE WIRETAP ----
                for handle in hook_handles:
                                handle.remove()
                
                y_norm_truth = self.data_processor.out_normalizer.transform(y_raw)
                y_phys_pred, _ = self.data_processor.postprocess(y_norm_pred.clone(), sample)

                if isinstance(y_phys_pred, dict):
                    y_phys_pred = y_phys_pred['y']

                residual_norm = y_norm_truth - y_norm_pred
                n_channels = y_norm_truth.shape[1]

                decoded_channel_losses = []
                
                for i, field_name in enumerate(target_losses):
                    loss_fn = target_losses.get(field_name, LpLoss(d=2, p=2, reduction='mean'))
                    weight = target_weights.get(field_name, 1.0)
                    channel_residual = residual_norm[:, i:i+1, ...]
                    
                    encoded_channel_loss = loss_fn(y_norm_pred[:, i:i+1, ...], y_norm_truth[:, i:i+1, ...])
                    encoded_mse_loss = torch.mean(channel_residual**2).item()
                    
                    decoded_channel_loss = loss_fn(y_phys_pred[:, i:i+1, ...], y_raw[:, i:i+1, ...])
                    decoded_channel_losses.append(decoded_channel_loss)
                    print(f"Epoch {epoch} Sample {sample_idx} Channel {field_name} - Loss: {encoded_channel_loss.item():.6f} Weighted Loss: {(weight * encoded_channel_loss).item():.6f}  MSE {encoded_mse_loss:.6f}   Decoded Loss: {decoded_channel_loss.item():.6f}")

                residual = y_raw - y_phys_pred
                resolution_h, resolution_w = x_raw.shape[-2], x_raw.shape[-1]

            # --- Plotting Main Grid ---
            n_out = y_raw.shape[1]
            fig, axes = plt.subplots(n_out, 3, figsize=(25, 4 * n_out))
            fig.suptitle(f"Airfrans Prediction: Epoch {epoch}: Sample {sample_idx}", fontsize=20)
            
            if n_out == 4:
                n_labels = ['Delta C_p', 'Delta U_x', 'Delta U_y', 'log_10(nu_t/nu)']
                residual_cmap_ranges = [0.01, 0.01, 0.01, 4.0]
            else:
                n_labels = [f'Channel {i}' for i in range(n_out)]
                residual_cmap_ranges = [0.05] * n_out

            if n_out == 1:
                axes = np.expand_dims(axes, axis=0)

            for i in range(n_out):
                cp_min, cp_max = y_raw[0, i].min().item(), y_raw[0, i].max().item()
                res = residual[0, i].cpu().numpy()
                stats_text = f"Decoded Range: [{cp_min:.1f}, {cp_max:.1f}]\nMAE: {np.abs(res).mean():.4f}"

                im0 = axes[i, 0].imshow(y_raw[0, i].cpu().numpy().T, origin='lower', vmin=cp_min, vmax=cp_max)
                axes[i, 0].set_title(f"Truth {n_labels[i]} \n{stats_text}", fontsize=10, loc='left')
                plt.colorbar(im0, ax=axes[i, 0])

                im3 = axes[i, 1].imshow(y_phys_pred[0, i].cpu().numpy().T, origin='lower', vmin=cp_min, vmax=cp_max)
                axes[i, 1].set_title(f"Pred {n_labels[i]}  Residual {decoded_channel_losses[i].item():.6f} \n")
                plt.colorbar(im3, ax=axes[i, 1])

                max_err = np.max(np.abs(residual_cmap_ranges[i]))
                im4 = axes[i, 2].imshow(res.T, origin='lower', cmap='RdBu_r', vmin=-max_err, vmax=max_err)
                axes[i, 2].set_title(f"Residual {decoded_channel_losses[i].item():.6f}")
                plt.colorbar(im4, ax=axes[i, 2])

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # --- FIXED PATH LOGIC ---
            # Forces it to save exactly next to where the checkpoints save
            safe_base_dir = self.save_dir if hasattr(self, 'save_dir') else Path(save_dir)
            output_dir = safe_base_dir / "diagnostic_plots"
            output_dir.mkdir(parents=True, exist_ok=True)    
            
            plt.savefig(f"{output_dir}/{prefix}_fields_{resolution_h}x{resolution_w}_sample_{sample_idx}_epoch_{epoch:04d}.png")
            plt.close(fig)

    # --- Plotting All Spectrograms ---
            if fft_storage:
                import matplotlib.patches as patches
                
                num_layers = len(fft_storage)
                # Create a wide figure to fit all layers side-by-side
                fig_fft, axes_fft = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5))
                
    # 1. Calculate GLOBAL min and max across all layers for consistent color scaling
                global_vmin = min([np.min(data) for data in fft_storage.values()])
                global_vmax = max([np.max(data) for data in fft_storage.values()])
                
                for ax, layer_name in zip(axes_fft, sorted(fft_storage.keys())):
                    spec_data = fft_storage[layer_name]
                    
                    n_xi, n_eta = spec_data.shape
                    freq_extent = [-n_xi//2, n_xi//2, -n_eta//2, n_eta//2]
                    
                    # 2. Lock the color scale using vmin and vmax
                    im_spec = ax.imshow(
                        spec_data.T, 
                        origin='lower', 
                        cmap='magma', 
                        extent=freq_extent,
                        aspect='auto',
                        vmin=global_vmin,  # Locked global min
                        vmax=global_vmax   # Locked global max
                    )
                    
                    ax.set_title(f"Layer: {layer_name}", fontsize=14)
                    ax.set_xlabel("Chordwise Wavenumber ($k_\\xi$)")
                    if ax == axes_fft[0]:
                        ax.set_ylabel("Normal Wavenumber ($k_\\eta$)")
                    
                    plt.colorbar(im_spec, ax=ax, fraction=0.046, pad=0.04, label="Log Magnitude")
                    
                    if hasattr(self.model, 'n_modes'):
                        m_xi, m_eta = self.model.n_modes[0], self.model.n_modes[1]
                        
                        rect = patches.Rectangle(
                            (-m_xi, -m_eta), 
                            2 * m_xi,        
                            2 * m_eta,       
                            linewidth=2, 
                            edgecolor='cyan', 
                            facecolor='none', 
                            linestyle='--'
                        )
                        ax.add_patch(rect)

                plt.tight_layout()                
                # Save the multi-layer plot
                spec_dir = safe_base_dir / "spectrograms"
                spec_dir.mkdir(parents=True, exist_ok=True)
                
                spec_path = spec_dir / f"{prefix}_fft_all_layers_sample_{sample_idx}_epoch_{epoch:04d}.png"
                plt.savefig(spec_path, bbox_inches='tight')
                print(f"📻 Saved Multi-Layer Spectrogram to: {spec_path}")
                plt.close(fig_fft)

            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            peak = torch.cuda.max_memory_allocated(0) / (1024**3)
            print(f"Current VRAM: {allocated:.2f} GB | Peak VRAM: {peak:.2f} GB")

    def _register_fft_hook(self):
            """
            Attaches a hook to EVERY Fourier layer to intercept data, perform a 2D FFT, 
            and save the centered spectrograms.
            Returns a list of handles (to remove later) and the dictionary of data.
            """
            storage = {}
            handles = []
            
            # We need a factory function to avoid Python's late-binding loop closure trap
            def get_hook(layer_name):
                def hook(module, input, output):
                    # Intercept the input to the layer (Shape: [Batch, Channels, Xi, Eta])
                    x = input[0].detach()
                    
                    # 1. Perform full 2D FFT
                    x_ft = torch.fft.fft2(x)
                    
                    # 2. Shift the zero-frequency (DC) to the center
                    x_ft_shifted = torch.fft.fftshift(x_ft, dim=(-2, -1))
                    
                    # 3. Log Magnitude
                    mag = torch.log1p(torch.abs(x_ft_shifted))
                    
                    # 4. Average across hidden channels for Sample 0
                    storage[layer_name] = mag[0].mean(dim=0).cpu().numpy()
                return hook

            # Loop through the model and hook all spectral conv blocks
            for name, module in self.model.named_modules():
                # Targets NeuralOperator's standard naming: fno_blocks.convs.0, .1, .2, etc.
                if "fno_blocks.convs." in name and name.split(".")[-1].isdigit():
                    handles.append(module.register_forward_hook(get_hook(name)))
                    
            return handles, storage