from networkx import config

from tims.airfrans_panelfoil_pad_linear.Airfrans_Evaluator import AirfoilEvaluator
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
import math

class AirfransDeltaTrainer(Trainer):

    def __init__(self, *, model, n_epochs, wandb_log=False, device="cpu", mixed_precision=False, data_processor=None, eval_interval=1, log_output=False, use_distributed=False, verbose=False, grad_clip=None):
        self.evaluator = AirfoilEvaluator(processor=data_processor, device=device)
        self.grad_clip = grad_clip   # Max global gradient norm, e.g. 1.0
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
                #print(f"Eval {key}: {errors[key]:.6f} over {local_n_samples} samples")
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

        epoch_metrics = {}

        for epoch in range(self.start_epoch, self.n_epochs):
            train_err, avg_loss, avg_lasso_loss, epoch_train_time, channel_metrics, lr = self.train_one_epoch(epoch, train_loader, training_loss)
            epoch_metrics = dict(train_err=train_err, avg_loss=avg_loss, avg_lasso_loss=avg_lasso_loss, epoch_train_time=epoch_train_time)

            combined_eval_metrics = {}
            # Evaluate on all test loaders and aggregate metrics
            if epoch % self.eval_interval == 0:
                # Evaluate on all test loaders and aggregate metrics
                test_eval_metrics = self.evaluate_all(epoch=epoch, eval_losses=eval_losses, test_loaders=test_loaders, eval_modes=eval_modes, max_autoregressive_steps=max_autoregressive_steps)
                epoch_metrics.update(**test_eval_metrics)
                # Evaluate on all train loaders and aggregate metrics
                train_eval_metrics = self.evaluate_all(epoch=epoch, eval_losses=eval_losses, test_loaders={"TrainData_Eval": train_loader}, eval_modes=eval_modes, max_autoregressive_steps=max_autoregressive_steps)
                
                epoch_metrics.update(**train_eval_metrics)
                if save_best is not None and test_eval_metrics[save_best] < best_metric_value:
                    best_metric_value = test_eval_metrics[save_best]
                    self.checkpoint(save_dir)

                if self.verbose and epoch % self.eval_interval == 0:
                    #Dict to store all eval metrics for logging
                    combined_eval_metrics = {**test_eval_metrics, **train_eval_metrics}

                    # --- NEW SUMMARY BLOCK START ---
                    print(f"\n{'='*25} 📊 EVALUATION SUMMARY {'='*25}")
                    groups = {}
                    for k, v in combined_eval_metrics.items():
                        # Extract the dataset name and the metric type
                        parts = k.split('_Weighted_')
                        prefix = parts[0]
                        metric_name = parts[-1] if len(parts) > 1 else k
                        
                        if prefix not in groups: groups[prefix] = []
                        groups[prefix].append(f"{metric_name}={v:.4f}")

                    for prefix, metrics in groups.items():
                        label = "🚀 TRAIN" if "Train" in prefix else f"🧪 TEST {prefix}"
                        print(f"{label:<25} | " + ", ".join(metrics))
                    print(f"{'='*72}\n")
                    # --- NEW SUMMARY BLOCK END ---
                # Plot diagnostic sample using training loader         
                
                self.plot_diagnostic_grid(train_loader, epoch, save_dir=save_dir, prefix="Train", sample_idx=sample_idx, training_loss=training_loss)
                self.plot_physical_mesh(train_loader, epoch,  training_loss=training_loss, save_dir=save_dir, prefix="Train", sample_idx=sample_idx)
                self.plot_pure_fourier_features(train_loader, epoch, save_dir=save_dir, sample_idx=sample_idx, prefix="fourier_train")

                # Plot diagnostic sample using test loader
                # make sure sample_idx is valid for test loader, otherwise it will error out. You can set it to 0 to always plot the first sample in the test set.
                for test_name, test_loader in test_loaders.items():
                    # should this use eval_losses or training_loss? For now we will use training_loss to be consistent with the diagnostic grid, but it could be interesting to use the specific eval loss for that test set if it's different from the training loss.
                    self.plot_diagnostic_grid(test_loader, epoch, save_dir=save_dir, prefix=test_name, sample_idx=sample_idx, training_loss=training_loss)
                    self.plot_physical_mesh(test_loader, epoch, save_dir=save_dir, prefix=test_name, sample_idx=sample_idx, training_loss=training_loss)
            
            if self.verbose:
                self.log_training(
                    epoch=epoch,
                    time=epoch_train_time,
                    avg_loss=avg_loss,
                    train_err=train_err,
                    channel_metrics=channel_metrics,
                    eval_metrics=combined_eval_metrics,  # Empty 95% of the time, full on eval intervals
                    avg_lasso_loss=avg_lasso_loss,
                    lr=lr,
                )
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

            # Gradient clipping
            if hasattr(self, 'grad_clip') and self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

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

        return train_err, avg_loss, avg_lasso_loss, epoch_train_time,channel_metrics, lr

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
        
    def log_training(self, 
                     epoch: int, 
                     time: float, 
                     avg_loss: float, 
                     train_err: float, 
                     channel_metrics: dict = None, 
                     eval_metrics: dict = None,
                     avg_lasso_loss: float = None, lr: float = None):

        channel_metrics = channel_metrics or {}
        eval_metrics = eval_metrics or {}

        if self.use_distributed and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        if not self.log_output:
            return

        combined_metrics = {**channel_metrics, **eval_metrics}

        if self.wandb_log:
            values_to_log = {
                "train_err": train_err,
                "time": time,
                "avg_loss": avg_loss,
                "avg_lasso_loss": avg_lasso_loss,
                "lr": lr,
                **combined_metrics
            }
            wandb.log(data=values_to_log, step=epoch)

        # --- Robust CSV logging (handles commas in quoted headers + evolving columns) ---
        import csv

        log_path = self.save_dir / "loss_history.csv"
        row_data = {
            "epoch": epoch,
            "time": f"{time:.2f}",
            "train_err": f"{train_err:.6f}",
            "avg_loss": f"{avg_loss:.6f}",
            "lr": f"{lr:.2e}" if lr is not None else "",
            **combined_metrics
        }

        existing_rows = []
        if log_path.exists():
            with open(log_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                existing_rows = list(reader)
        else:
            fieldnames = []

        # Add any new keys (do NOT ignore them)
        for k in row_data.keys():
            if k not in fieldnames:
                fieldnames.append(k)

        # Rewrite file with updated schema, then append current row
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for old_row in existing_rows:
                writer.writerow(old_row)
            writer.writerow(row_data)
        # ---------------------------------------------------------
        # 4. CONSOLE OUTPUT (Clean Summary)
        # ---------------------------------------------------------
        if epoch % self.eval_interval == 0:
            # Option A: Only print if we are doing a diagnostic/save (e.g., every 20 epochs)
            if epoch % self.eval_interval == 0: 
                print(f"[{epoch}] time={time:.2f}s | avg_loss={avg_loss:.5f} | train_err={train_err:.5f}")
                train_msg = "      Train Ch: " + ", ".join([f"{k}={v:.4f}" for k, v in channel_metrics.items()])
                print(train_msg)
                
                if combined_metrics:
                    eval_msg = "      Eval Met: " + ", ".join([f"{k}={v:.4f}" for k, v in combined_metrics.items() if k not in channel_metrics])
                    print(eval_msg)

            if not hasattr(self, '_plot_history'):
                self._plot_history = {'epoch': [], 'train_err': [], 'lr': []}
            
            self._plot_history['epoch'].append(epoch)
            self._plot_history['train_err'].append(train_err)
            self._plot_history['lr'].append(lr)
            
            # 1. Update lists with incoming metrics
            for k, v in combined_metrics.items():
                if k not in self._plot_history:
                    # Pad it with NaNs for all previous epochs
                    # so it catches up to the current epoch length!
                    self._plot_history[k] = [float('nan')] * (len(self._plot_history['epoch']) - 1)
                self._plot_history[k].append(v)
                
            # 2. The Great Equalizer: Pad any missing metrics with NaN
            target_length = len(self._plot_history['epoch'])
            for k in self._plot_history.keys():
                if len(self._plot_history[k]) < target_length:
                    self._plot_history[k].append(float('nan'))


            # Create a figure with 2 vertically stacked subplots that share the X-axis
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # ---------------------------------------------------------
            # TOP PANEL: Encoded Training Losses (Gradient Data)
            # ---------------------------------------------------------
            ax1.plot(self._plot_history['epoch'], self._plot_history['train_err'], color='black', label="Total Train Err", linewidth=2)
            
            cmap = plt.get_cmap('tab10')
            for i, k in enumerate(channel_metrics.keys()):
                ax1.plot(self._plot_history['epoch'], self._plot_history[k], color=cmap(i), linestyle='-', label=f"Train: {k}")
                
            ax1.grid(True, which="both", ls="--", alpha=0.5)
            ax1.set_ylabel("Encoded Loss Magnitude")
            ax1.set_yscale('log')  
            ax1.set_title(f'Optimizer Gradients (Epoch {epoch})')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # ---------------------------------------------------------
            # BOTTOM PANEL: Decoded Evaluation Metrics (Physical Data)
            # ---------------------------------------------------------
            eval_keys = [k for k in self._plot_history.keys() if k not in channel_metrics and k not in ['epoch', 'train_err', 'lr']]
                        
            for i, k in enumerate(eval_keys):
                # Filter out the NaNs so Matplotlib connects the dots!
                valid_epochs = [e for e, v in zip(self._plot_history['epoch'], self._plot_history[k]) if not math.isnan(v)]
                valid_vals = [v for v in self._plot_history[k] if not math.isnan(v)]
                
                # Only plot if we actually have data (prevents crash on Epoch 0 if eval hasn't run yet)
                if valid_epochs:
                    ax2.plot(valid_epochs, valid_vals, color=cmap(i + len(channel_metrics)), linestyle='--', marker='o', markersize=4, label=f"Eval: {k}")
            
            ax2.grid(True, which="both", ls="--", alpha=0.5)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Decoded Physical Error")
            ax2.set_yscale('log')  
            ax2.set_title('Physical Evaluation Metrics')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout() 
            
            file = self.save_dir / "training_history.png"
            plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=150, facecolor="white")
            plt.close("all")

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
                
                for handle in hook_handles:
                    handle.remove()
                
                y_norm_truth = self.data_processor.out_normalizer.transform(y_raw)
                y_phys_pred, _ = self.data_processor.postprocess(y_norm_pred.clone(), sample)

                if isinstance(y_phys_pred, dict):
                    y_phys_pred = y_phys_pred['y']

                residual_norm = y_norm_truth - y_norm_pred
                
                # Build the LIST that the Matplotlib axes loop expects!
                decoded_channel_losses_list = []

                # --- Initialize Accumulators ---
                total_encoded_weighted = 0.0
                total_decoded_weighted = 0.0
                total_encoded_mse = 0.0

                # Calculate the sum of all weights for normalization!
                sum_of_weights = sum(target_weights.get(f, 1.0) for f in target_losses.keys())
                sum_of_weights = sum_of_weights if sum_of_weights > 0 else 1.0

                for i, field_name in enumerate(target_losses):
                    loss_fn = target_losses.get(field_name) # Fallback handled by your original logic if needed
                    loss_fn_name = type(loss_fn).__name__
                    weight = target_weights.get(field_name, 1.0) / sum_of_weights  # Normalize the weight so they sum to 1
                    
                    channel_residual = residual_norm[:, i:i+1, ...]
                    
                    # Explicit, safe per-channel calculation
                    encoded_channel_loss = loss_fn(y_norm_pred[:, i:i+1, ...], y_norm_truth[:, i:i+1, ...])
                    encoded_mse_loss = torch.mean(channel_residual**2).item()
                    
                    decoded_channel_loss = loss_fn(y_phys_pred[:, i:i+1, ...], y_raw[:, i:i+1, ...])
                    decoded_channel_losses_list.append(decoded_channel_loss)

                    # Calculate and accumulate RAW weighted values
                    enc_weighted = (weight * encoded_channel_loss).item()
                    dec_weighted = (weight * decoded_channel_loss).item()
                    
                    total_encoded_weighted += enc_weighted
                    total_decoded_weighted += dec_weighted
                    total_encoded_mse += encoded_mse_loss
                    
                    print(f"Epoch {epoch}  Dataset {prefix} Sample {sample_idx} Channel {field_name} ({loss_fn_name}) - "
                          f"Encoded Loss: {encoded_channel_loss.item():.6f} | Normalized Weight: {(weight * encoded_channel_loss).item():.6f} || "
                          f"Decoded Loss: {decoded_channel_loss.item():.6f} | MSE: {encoded_mse_loss:.6f}")
                print("-" * 110)
                print(f"Epoch {epoch}  Dataset {prefix} Sample {sample_idx} NORMALIZED SUM - "
                      f"Encoded: {total_encoded_weighted:.6f} || "
                      f"Decoded: {total_decoded_weighted:.6f} | MSE: {total_encoded_mse:.6f}")
                print("=" * 110)
                # Reassign to the variable name the rest of your plotting function uses
                decoded_channel_losses = decoded_channel_losses_list
                
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
            output_dir = safe_base_dir / f"diagnostic_{prefix}"
            output_dir.mkdir(parents=True, exist_ok=True)    
            
            plt.savefig(f"{output_dir}/{prefix}_fields_{resolution_h}x{resolution_w}_sample_{sample_idx}_epoch_{epoch:04d}.png")
            plt.close(fig)

            if fft_storage:
                import matplotlib.patches as patches
                
                num_layers = len(fft_storage)
                # Create a 2-row grid: Row 0 = Inputs, Row 1 = Outputs
                fig_fft, axes_fft = plt.subplots(2, num_layers, figsize=(6 * num_layers, 10))
                
                # Ensure axes_fft is a 2D array even if there's only 1 layer
                if num_layers == 1:
                    axes_fft = np.expand_dims(axes_fft, axis=1)
                    
                fig_fft.suptitle(f"FNO Spatial Frequencies (Positive Quadrant) : Epoch {epoch}", fontsize=20, y=1.02)
                
                # 1. Calculate GLOBAL min and max across ALL inputs and outputs
                all_data = []
                for layer_data in fft_storage.values():
                    all_data.append(layer_data['input'])
                    all_data.append(layer_data['output'])
                global_vmin = min([np.min(d) for d in all_data])
                global_vmax = max([np.max(d) for d in all_data])
                
                # 2. Loop through the layers to populate the columns
                for col, layer_name in enumerate(sorted(fft_storage.keys())):
                    # Get the top and bottom axes for this specific layer column
                    ax_in = axes_fft[0, col]
                    ax_out = axes_fft[1, col]
                    
                    data_dict = fft_storage[layer_name]
                    
                    # Loop to plot Input (Top) then Output (Bottom)
                    for ax, data, label in zip([ax_in, ax_out], 
                                            [data_dict['input'], data_dict['output']], 
                                            ['Input to', 'Output of']):
                        
                        # Full Complex FFT in X direction but only positive frequencies in Y direction

                        # A. Slice to keep FULL X (Chordwise) and ONLY Positive Y (Normal)
                        # The FFT data is centered, so we take the right half for positive frequencies
                        n_xi, n_eta = data.shape
                        center_xi, center_eta = n_xi // 2, n_eta // 2
                        
                        # Keep all rows (:), but only the right half of the columns (center_eta:)
                        half_spec = data[:, center_eta:]
                        
                        # B. Update extent so X goes from -center to +center, and Y starts at 0
                        freq_extent = [-center_xi, center_xi, 0, center_eta]
                        
                        # C. Plot with locked global scaling
                        im_spec = ax.imshow(
                            half_spec.T, 
                            origin='lower', 
                            cmap='magma', 
                            extent=freq_extent,
                            aspect='auto',
                            vmin=global_vmin,
                            vmax=global_vmax
                        )
                        
                        ax.set_title(f"{label} {layer_name}", fontsize=14)
                        ax.set_xlabel("Chordwise Wavenumber ($k_\\xi$)")
                        if col == 0:
                            ax.set_ylabel("Normal Wavenumber ($k_\\eta$)")
                        
                        # D. Draw the Symmetrical Cutoff Box
                        if hasattr(self.model, 'n_modes'):
                            # 1. X-Axis: 32 positive + 32 negative
                            m_xi = self.model.n_modes[0] // 2 
                            
                            # 2. Y-Axis: Handle the physical reflection padding
                            m_eta_raw = self.model.n_modes[1]
                            if hasattr(self.model, 'domain_padding') and self.model.domain_padding is not None:
                                if self.model.domain_padding.__class__.__name__ == "MirrorPaddingY"  or self.model.domain_padding.__class__.__name__ == "LinearXMirrorYPadding":
                                    m_eta = m_eta_raw // 2  
                                else:
                                    m_eta = m_eta_raw
                            else:
                                m_eta = m_eta_raw

                            rect = patches.Rectangle(
                                (-m_xi, 0),    # Anchor shifts left to capture negative X
                                2 * m_xi,      # Width is now total modes (positive + negative)
                                m_eta,         # Height remains just the positive Y
                                linewidth=2, 
                                edgecolor='cyan', 
                                facecolor='none', 
                                linestyle='--'
                            )
                            ax.add_patch(rect)

                # Add a single colorbar to the right of the entire figure
                cbar_ax = fig_fft.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
                fig_fft.colorbar(im_spec, cax=cbar_ax, label="Log Magnitude")
                
                plt.subplots_adjust(left=0.05, right=0.9, wspace=0.2, hspace=0.3)
                
                spec_dir = safe_base_dir / f"spectrograms_{prefix}"
                spec_dir.mkdir(parents=True, exist_ok=True)
                
                spec_path = spec_dir / f"{prefix}_fft_in_out_sample_{sample_idx}_epoch_{epoch:04d}.png"
                plt.savefig(spec_path, bbox_inches='tight')
                print(f"📻 Saved Input/Output Spectrogram to: {spec_path}")
                plt.close(fig_fft)

            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            peak = torch.cuda.max_memory_allocated(0) / (1024**3)
            print(f"Current VRAM: {allocated:.2f} GB | Peak VRAM: {peak:.2f} GB")


    

    def plot_physical_mesh(self, loader, epoch, training_loss, save_dir="plots", sample_idx=0, prefix="prediction"):
        """Plots the physical Truth, Prediction, and Residual mapped onto the actual 2D CFD mesh."""
        
        # 1. DDP Safety Guard
        if self.use_distributed and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
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

            # Forward pass & Decode
            y_norm_pred = self.model(x_input)
            y_phys_pred, _ = self.data_processor.postprocess(y_norm_pred, sample)

            if isinstance(y_phys_pred, dict):
                y_phys_pred = y_phys_pred['y']

        # 2. Extract Data to CPU Numpy Arrays
        # User specified: Input Channel 0 = X coords, Channel 1 = Y coords
        grid_x = x_raw[0, 0, ...].cpu().numpy()
        grid_y = x_raw[0, 1, ...].cpu().numpy()
        
        truth = y_raw[0, ...].cpu().numpy()
        pred = y_phys_pred[0, ...].cpu().numpy()
        residual = truth - pred
        
        channel_names = list(target_losses.keys())
        num_channels = len(channel_names)
        
        # 3. Setup the Figure (Rows = Channels, Cols = Truth, Pred, Residual)
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, axes = plt.subplots(num_channels, 3, figsize=(18, 4 * num_channels), constrained_layout=True)
        # Handle the edge case of 1 channel (axes is 1D instead of 2D)
        if num_channels == 1:
            axes = np.expand_dims(axes, axis=0)
            
        fig.suptitle(f"[{prefix}] Physical Mesh Projection - Epoch {epoch}", fontsize=16, fontweight='bold')
        
        n_out = y_raw.shape[1]

        if n_out == 4:
            n_labels = ['Delta C_p', 'Delta U_x', 'Delta U_y', 'log_10(nu_t/nu)']
            output_cmap_ranges = [(truth[:, i].min(), truth[:, i].max()) for i in range(n_out)]
            residual_cmap_ranges = [0.01, 0.01, 0.01, 4.0]
        else:
            n_labels = [f'Channel {i}' for i in range(n_out)]
            residual_cmap_ranges = [0.05] * n_out


        # 4. Plotting Loop
        for i, field_name in enumerate(channel_names):
            c_truth = truth[i]
            c_pred = pred[i]
            c_resid = residual[i]
            c_min, c_max = y_raw[0, i].min().item(), y_raw[0, i].max().item()


            
            # Column 0: Ground Truth
            vmin, vmax = output_cmap_ranges[i]
            im0 = axes[i, 0].pcolormesh(grid_x, grid_y, c_truth, cmap='viridis', shading='auto', vmin=c_min, vmax=c_max)
            axes[i, 0].set_title(f"{field_name} (Truth)")
            fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            # Column 1: Prediction
            im1 = axes[i, 1].pcolormesh(grid_x, grid_y, c_pred, cmap='viridis', shading='auto', vmin=c_min, vmax=c_max)
            axes[i, 1].set_title(f"{field_name} (Prediction)")
            fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
            
            # Column 2: Residual (Truth - Pred)
            im2 = axes[i, 2].pcolormesh(grid_x, grid_y, c_resid, cmap='coolwarm', shading='auto', vmin=-residual_cmap_ranges[i], vmax=residual_cmap_ranges[i])
            axes[i, 2].set_title(f"{field_name} (Residual)")
            fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

            # Formatting for all subplots in this row
            for j in range(3):
                axes[i, j].set_aspect('equal') # Keeps the airfoil physically proportioned
                axes[i, j].set_xlabel("X (m)")
                axes[i, j].set_ylabel("Y (m)")
                
                # Optional: Zoom in on the airfoil (uncomment to restrict field of view)
                axes[i, j].set_xlim([-1.5, 2.5])
                axes[i, j].set_ylim([-1.5, 1.5])

        # 5. Save and Close
        mesh_save_dir = save_dir / f"mesh_plots_{prefix}"
        mesh_save_dir.mkdir(parents=True, exist_ok=True)
        file_path = mesh_save_dir / f"mesh_sample_{prefix}_{sample_idx}_epoch_{epoch}.png"
        
        plt.savefig(file_path, dpi=200, bbox_inches='tight', facecolor="white")
        plt.close(fig)

    def plot_pure_fourier_features(self, loader, epoch, save_dir="plots", sample_idx=0, prefix="prediction"):
        """Hooks the final inverse FFT output and plots it as a rectangular computational grid."""
        
        # 1. DDP Safety Guard
        if self.use_distributed and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        self.model.eval()
        if self.data_processor:
            self.data_processor.eval()

        # 2. Find the last Spectral Convolution layer
        spectral_layers = [m for m in self.model.modules() if 'SpectralConv' in type(m).__name__]
        if not spectral_layers:
            return
        last_spectral_layer = spectral_layers[-1]

        # 3. Setup the Wiretap
        ifft_output = {}
        def hook_fn(module, input, output):
            ifft_output['flowy_tensor'] = output.detach().clone()
            
        handle = last_spectral_layer.register_forward_hook(hook_fn)

        # 4. Forward Pass
        batch = next(iter(loader))
        with torch.no_grad():
            x_raw = batch['x'][sample_idx:sample_idx+1].to(self.device)
            y_raw = batch['y'][sample_idx:sample_idx+1].to(self.device)
            sample = self.data_processor.preprocess({'x': x_raw, 'y': y_raw})
            
            _ = self.model(sample['x'].to(self.device))
            
        handle.remove()

        # 5. Extract Data & UNPAD
        flowy_tensor = ifft_output['flowy_tensor']
        
        # Strip away the mirror and linear bridge padding!
        if hasattr(self.model, 'domain_padding') and self.model.domain_padding is not None:
            flowy_tensor = self.model.domain_padding.unpad(flowy_tensor)

        flowy_data = flowy_tensor[0].cpu().numpy()

        # 6. Plotting the Rectangular Matrix
        import matplotlib.pyplot as plt
        import numpy as np
        
        num_plots = min(3, flowy_data.shape[0])
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5), constrained_layout=True)
        if num_plots == 1: axes = [axes]
        
        fig.suptitle(f"[{prefix}] Latent Computational Domain - Epoch {epoch}", fontsize=16, fontweight='bold', color='teal')

        for i in range(num_plots):
            c_data = flowy_data[i]
            
            # imshow maps the matrix directly to pixels. 
            # We transpose (.T) so X is horizontal and Y is vertical.
            # aspect='auto' lets the rectangle stretch to fit the figure nicely.
            im = axes[i].imshow(c_data.T, origin='lower', cmap='twilight_shifted', aspect='auto')
            
            axes[i].set_title(f"Latent Wave Channel {i}", fontsize=12)
            fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
            axes[i].set_xlabel("Computational X Index (Chordwise)")
            if i == 0: 
                axes[i].set_ylabel("Computational Y Index (Normal)")

        # 7. Save
        out_dir = save_dir / f"fourier_features_{prefix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / f"{prefix}_flowy_features_sample_{sample_idx}_epoch_{epoch:04d}.png"
        
        plt.savefig(file_path, dpi=200, bbox_inches='tight', facecolor="white")
        plt.close(fig)    

    def _register_fft_hook(self):
            storage = {}
            handles = []
            
            def get_hook(layer_name):
                def hook(module, input, output):
                    # --- 1. Process INPUT ---
                    x_in = input[0].detach()
                    x_in_ft = torch.fft.fftshift(torch.fft.fft2(x_in), dim=(-2, -1))
                    mag_in = torch.log1p(torch.abs(x_in_ft))[0].mean(dim=0).cpu().numpy()
                    
                    # --- 2. Process OUTPUT ---
                    x_out = output.detach()
                    x_out_ft = torch.fft.fftshift(torch.fft.fft2(x_out), dim=(-2, -1))
                    mag_out = torch.log1p(torch.abs(x_out_ft))[0].mean(dim=0).cpu().numpy()
                    
                    # Store both
                    storage[layer_name] = {'input': mag_in, 'output': mag_out}
                return hook

            for name, module in self.model.named_modules():
                if "fno_blocks.convs." in name and name.split(".")[-1].isdigit():
                    handles.append(module.register_forward_hook(get_hook(name)))
                    
            return handles, storage