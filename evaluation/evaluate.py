"""
Evaluate script: visualize reconstruction errors by concatenating sequences.
Creates plots with normal sequences (top) and anomalous sequences (bottom) for each channel.
Divides into blocks of max 500 sequences per plot.
"""

import torch
import yaml
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import math

from utils.load_model import get_model


def load_checkpoint(checkpoint_path):
    """Load checkpoint and extract all necessary info."""
    print(f"\n📦 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    cfg = checkpoint['cfg']
    metric_dataset_path = checkpoint.get('metric_dataset_path', None)
    scaler_params = checkpoint.get('scaler_params_pre_training',
                                   checkpoint.get('scaler_params_fine_tuning', None))

    print(f"   ✓ Config loaded")
    print(f"   ✓ Metric dataset path: {metric_dataset_path}")

    return checkpoint, cfg, metric_dataset_path, scaler_params


def load_metric_dataset(metric_dataset_path, batch_size=32):
    """Load metric dataset from saved file."""
    print(f"\n📊 Loading metric dataset: {metric_dataset_path}")

    saved_dict = torch.load(metric_dataset_path, map_location='cpu')
    metric_dataset = saved_dict['dataset']
    metadata = saved_dict.get('metadata', {})

    print(f"   ✓ Sequences: {len(metric_dataset)}")
    print(f"   ✓ Standardized: {metadata.get('is_standardized', 'N/A')}")
    print(f"   ✓ Strategy: {metadata.get('strategy', 'N/A')}")

    # Print corruption info
    corruption_config = metadata.get('corruption_config', {})
    print(f"   ✓ Corruption ratio: {corruption_config.get('corruption_ratio', 'N/A')}")

    # Create DataLoader
    metric_loader = DataLoader(
        metric_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return metric_loader, metadata, metric_dataset


def compute_reconstruction_errors(model, dataloader, device='cpu', use_error='abs',
                                  weighting_factor=False, epsilon=1e-3,
                                  metric_dataset_path=None):
    """
    Compute reconstruction errors for all sequences.
    Uses corruption_mapping from dataset to properly pair sequences.
    """
    import os  # ← Aggiungi qui se non è già in cima al file

    model.eval()
    model.to(device)

    # Detect last layer type
    last_layer_name = None
    for name, module in model.named_modules():
        last_layer_name = type(module).__name__
    is_conv2d = (last_layer_name == "Conv2d")

    print(f"\n🔍 Computing reconstructions...")
    print(f"   - Last layer type: {last_layer_name}")
    print(f"   - Will squeeze dimension 1: {is_conv2d}")
    print(f"   - Weighting factor: {weighting_factor}")

    # Load corruption mapping from dataset file
    corruption_mapping = None
    if metric_dataset_path and os.path.exists(metric_dataset_path):
        print(f"\n   📂 Loading corruption mapping from dataset...")
        saved_dict = torch.load(metric_dataset_path, map_location='cpu')
        dataset = saved_dict['dataset']

        if hasattr(dataset, 'corruption_mapping'):
            corruption_mapping = dataset.corruption_mapping
            print(f"      ✓ Corruption mapping found:")
            print(f"         - Clean sequences: {corruption_mapping['n_clean']}")
            print(f"         - Corrupted sequences: {corruption_mapping['n_corrupted']}")
            print(f"         - Mapping pairs: {len(corruption_mapping['original_indices'])}")
        else:
            print(f"      ⚠️  No corruption_mapping found in dataset!")

    # Store ALL sequences in order
    all_inputs, all_targets, all_recons, all_errors, all_masks = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            x, target, mask, *_ = batch

            x = x.to(device)
            target = target.to(device)

            # Forward pass
            recon = model(x)

            # Compute errors
            if use_error == 'abs':
                err = torch.abs(recon.detach() - target)
            else:  # 'se'
                err = (recon.detach() - target) ** 2

            # Squeeze if necessary
            if is_conv2d and err.dim() == 4 and err.shape[1] == 1:
                err = err.squeeze(1)

            # Store everything
            all_inputs.append(x.cpu())
            all_targets.append(target.cpu())
            all_recons.append(recon.cpu())
            all_errors.append(err.cpu())
            all_masks.append(mask.cpu())

    # Concatenate all
    all_inputs = torch.cat(all_inputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_recons = torch.cat(all_recons, dim=0)
    all_errors = torch.cat(all_errors, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    N_total = all_errors.shape[0]
    print(f"\n   ✓ Total sequences processed: {N_total}")

    # Use corruption mapping if available
    if corruption_mapping is not None:
        n_clean = corruption_mapping['n_clean']
        n_corrupted = corruption_mapping['n_corrupted']
        original_indices = corruption_mapping['original_indices']

        print(f"\n   🔗 Using corruption mapping for pairing:")
        print(f"      - Clean sequences: 0-{n_clean - 1}")
        print(f"      - Corrupted sequences: {n_clean}-{n_clean + n_corrupted - 1}")

        # Extract clean sequences that were corrupted (DERANDOMIZZAZIONE!)
        normal_data_paired = {
            'inputs': all_inputs[original_indices],
            'targets': all_targets[original_indices],
            'reconstructions': all_recons[original_indices],
            'errors': all_errors[original_indices],
            'errors_normalized': None
        }

        # Extract corresponding corrupted sequences
        anomaly_data_paired = {
            'inputs': all_inputs[n_clean:n_clean + n_corrupted],
            'targets': all_targets[n_clean:n_clean + n_corrupted],
            'reconstructions': all_recons[n_clean:n_clean + n_corrupted],
            'errors': all_errors[n_clean:n_clean + n_corrupted],
            'errors_normalized': None,
            'masks': all_masks[n_clean:n_clean + n_corrupted],
        }

        # Verify pairing
        idx_check = 0
        input_diff = torch.abs(normal_data_paired['inputs'][idx_check] -
                               anomaly_data_paired['inputs'][idx_check]).sum().item()
        target_diff_per_channel = torch.abs(normal_data_paired['targets'][idx_check] -
                                            anomaly_data_paired['targets'][idx_check]).sum(dim=1)
        n_diff_channels = (target_diff_per_channel > 1e-3).sum().item()

        print(f"\n      Pairing verification (pair {idx_check}):")
        print(f"      - Original index: {original_indices[idx_check]}")
        print(f"      - Input difference: {input_diff:.6f} (should be 0)")
        print(f"      - Channels with differences: {n_diff_channels} (should be 1)")

        normal_data = normal_data_paired
        anomaly_data = anomaly_data_paired

    else:
        # Fallback: assume block organization
        print(f"\n   ⚠️  No corruption mapping - using block split (may be incorrect!)")
        N_half = N_total // 2

        normal_data = {
            'inputs': all_inputs[:N_half],
            'targets': all_targets[:N_half],
            'reconstructions': all_recons[:N_half],
            'errors': all_errors[:N_half],
            'errors_normalized': None
        }

        anomaly_data = {
            'inputs': all_inputs[N_half:],
            'targets': all_targets[N_half:],
            'reconstructions': all_recons[N_half:],
            'errors': all_errors[N_half:],
            'errors_normalized': None,
            'masks': all_masks[N_half:],
        }

    # Compute mean errors
    normal_mean = normal_data['errors'].mean().item()
    anomaly_mean = anomaly_data['errors'].mean().item()

    print(f"\n   ✓ Final mean errors:")
    print(f"      - Normal:  {normal_mean:.6f}")
    print(f"      - Anomaly: {anomaly_mean:.6f}")
    print(
        f"      - Difference: {anomaly_mean - normal_mean:.6f} ({100 * (anomaly_mean - normal_mean) / normal_mean:.2f}%)")

    # Apply weighting_factor (normalization)
    normalization_factor = None
    if weighting_factor and normal_data['errors'].numel() > 0:
        print(f"\n🔧 Applying weighting_factor (normalization with median)...")

        normal_perm = normal_data['errors'].permute(0, 2, 1)  # [N, C, T] → [N, T, C]

        C = normal_perm.shape[2]
        flat_norm = normal_perm.reshape(-1, C).float()
        normalization_factor = torch.quantile(flat_norm, 0.5, dim=0)  # [C]

        print(f"   ✓ Normalization factor (median per feature):")
        print(f"      - Shape: {normalization_factor.shape}")
        print(f"      - Min:   {normalization_factor.min():.6f}")
        print(f"      - Max:   {normalization_factor.max():.6f}")
        print(f"      - Mean:  {normalization_factor.mean():.6f}")

        epsilon = float(epsilon)
        norm = normalization_factor.view(1, 1, C) + epsilon
        normal_data['errors_normalized'] = (normal_perm / norm).permute(0, 2, 1)

        if anomaly_data['errors'].numel() > 0:
            anomaly_perm = anomaly_data['errors'].permute(0, 2, 1)
            anomaly_data['errors_normalized'] = (anomaly_perm / norm).permute(0, 2, 1)

        print(f"   ✓ Errors normalized")

    print(f"\n   ✓ Normal sequences: {normal_data['inputs'].shape[0]}")
    print(f"   ✓ Anomalous sequences: {anomaly_data['inputs'].shape[0]}")

    return normal_data, anomaly_data, normalization_factor


def concatenate_sequences(data_dict, max_sequences=None, n_features=None, model_name=None):
    """
    Concatenate sequences along time dimension.

    Expected formats from model:
    - conv_ae1D: [B, C, T] where C=features (6), T=time (64)
    - conv_ae2D: [B, 1, C, T] where C=features (6), T=time (64)
    """
    targets = data_dict['targets']
    recons = data_dict['reconstructions']
    errors = data_dict['errors']

    if targets.numel() == 0:
        return None

    print(f"\n   📐 Initial shape: {targets.shape}")
    print(f"   📐 Model: {model_name}")
    print(f"   📐 Expected n_features: {n_features}")

    # ============================================
    # STEP 1: Remove extra dimensions if present
    # ============================================
    # conv_ae2D has shape [B, 1, C, T] - squeeze the channel dim
    if targets.dim() == 4 and targets.shape[1] == 1:
        print(f"   ✓ Squeezing dim 1: [B, 1, C, T] → [B, C, T]")
        targets = targets.squeeze(1)
        recons = recons.squeeze(1)
        errors = errors.squeeze(1)
    elif targets.dim() == 4:
        print(f"   ⚠️  4D tensor but dim 1 is not 1: shape={targets.shape}")

    # ============================================
    # STEP 2: Verify we have [B, C, T] format
    # ============================================
    # At this point, both conv_ae1D and conv_ae2D should be [B, C, T]
    # where C should equal n_features

    if targets.dim() != 3:
        raise ValueError(f"Expected 3D tensor [B, C, T], got shape {targets.shape}")

    N, dim1, dim2 = targets.shape

    print(f"   📐 Current 3D shape: [B={N}, dim1={dim1}, dim2={dim2}]")

    # Check which dimension matches n_features
    if dim1 == n_features:
        # Already [B, C, T] ✓
        C, T = dim1, dim2
        print(f"   ✓ Correct format: [B, C={C}, T={T}]")
    elif dim2 == n_features:
        # Wrong format: [B, T, C] - need to permute
        print(f"   ⚠️  Wrong format: [B, T={dim1}, C={dim2}]")
        print(f"   🔧 Permuting to [B, C, T]")
        targets = targets.permute(0, 2, 1)
        recons = recons.permute(0, 2, 1)
        errors = errors.permute(0, 2, 1)
        C, T = dim2, dim1
    else:
        # Neither dimension matches n_features!
        print(f"   ⚠️  WARNING: Neither dimension matches n_features={n_features}")
        print(f"      dim1={dim1}, dim2={dim2}")
        # Use heuristic: smaller dimension is usually channels
        if dim1 < dim2:
            C, T = dim1, dim2
            print(f"   → Assuming [B, C={C}, T={T}] (dim1 < dim2)")
        else:
            C, T = dim2, dim1
            targets = targets.permute(0, 2, 1)
            recons = recons.permute(0, 2, 1)
            errors = errors.permute(0, 2, 1)
            print(f"   → Permuted to [B, C={C}, T={T}] (dim1 >= dim2)")

    print(f"   ✓ Final verified shape: [B={N}, C={C}, T={T}]")

    # Limit sequences
    if max_sequences is not None and N > max_sequences:
        targets = targets[:max_sequences]
        recons = recons[:max_sequences]
        errors = errors[:max_sequences]
        N = max_sequences

    # ============================================
    # STEP 3: Reshape to [C, N*T] for plotting
    # ============================================
    # [B, C, T] → [C, B, T] → [C, B*T]
    targets_concat = targets.permute(1, 0, 2).reshape(C, N * T).numpy()
    recons_concat = recons.permute(1, 0, 2).reshape(C, N * T).numpy()
    errors_concat = errors.permute(1, 0, 2).reshape(C, N * T).numpy()

    # Handle normalized errors
    errors_norm_concat = None
    if data_dict.get('errors_normalized') is not None:
        errors_norm = data_dict['errors_normalized']

        # Apply same transformations
        if errors_norm.dim() == 4 and errors_norm.shape[1] == 1:
            errors_norm = errors_norm.squeeze(1)

        # Check format
        if errors_norm.shape[1] == n_features:
            pass  # Already [B, C, T]
        elif errors_norm.shape[2] == n_features:
            errors_norm = errors_norm.permute(0, 2, 1)  # [B, T, C] → [B, C, T]
        elif errors_norm.shape[1] < errors_norm.shape[2]:
            pass  # Assume [B, C, T]
        else:
            errors_norm = errors_norm.permute(0, 2, 1)

        if max_sequences is not None:
            errors_norm = errors_norm[:N]
        errors_norm_concat = errors_norm.permute(1, 0, 2).reshape(C, N * T).numpy()

    return {
        'targets': targets_concat,
        'reconstructions': recons_concat,
        'errors': errors_concat,
        'errors_normalized': errors_norm_concat,
        'n_sequences': N,
        'seq_length': T,
        'n_features': C
    }


def plot_block(block_idx, normal_concat, anomaly_concat, feature_names,
               output_dir, max_timesteps_per_plot=None, weighting_factor=False):
    """
    Create interactive Plotly plot for a block.
    Shows normal (top) and anomalous (bottom) concatenated sequences for each channel.
    Includes global error (averaged across channels) at the end.
    Adds vertical lines to separate sequences.
    """
    C = normal_concat['n_features']
    seq_length = normal_concat['seq_length']  # Get sequence length

    # Determine timestep range for this block
    if max_timesteps_per_plot is not None:
        start_t = block_idx * max_timesteps_per_plot
        end_t = start_t + max_timesteps_per_plot

        # Slice data
        normal_targets = normal_concat['targets'][:, start_t:end_t]
        normal_recons = normal_concat['reconstructions'][:, start_t:end_t]
        normal_errors = normal_concat['errors'][:, start_t:end_t]

        anom_targets = anomaly_concat['targets'][:, start_t:end_t]
        anom_recons = anomaly_concat['reconstructions'][:, start_t:end_t]
        anom_errors = anomaly_concat['errors'][:, start_t:end_t]

        T_normal = normal_targets.shape[1]
        T_anom = anom_targets.shape[1]
    else:
        normal_targets = normal_concat['targets']
        normal_recons = normal_concat['reconstructions']
        normal_errors = normal_concat['errors']

        anom_targets = anomaly_concat['targets']
        anom_recons = anomaly_concat['reconstructions']
        anom_errors = anomaly_concat['errors']

        T_normal = normal_targets.shape[1]
        T_anom = anom_targets.shape[1]
        start_t = 0

    # ============================================
    # COMPUTE GLOBAL ERROR (averaged over channels)
    # ============================================
    # Shape: [C, T] → mean over C → [T]
    normal_global_error = normal_errors.mean(axis=0)  # [T]
    anomaly_global_error = anom_errors.mean(axis=0)  # [T]

    # Compute 95th percentile threshold on normal global error
    threshold_95 = np.percentile(normal_global_error, 95)

    # ============================================
    # COMPUTE SHARED Y-AXIS LIMITS PER CHANNEL
    # ============================================
    value_ranges = []
    error_ranges = []

    for feat_idx in range(C):
        # Values
        all_values = np.concatenate([
            normal_targets[feat_idx],
            normal_recons[feat_idx],
            anom_targets[feat_idx],
            anom_recons[feat_idx]
        ])
        value_min = np.min(all_values)
        value_max = np.max(all_values)
        value_margin = (value_max - value_min) * 0.05
        value_ranges.append((value_min - value_margin, value_max + value_margin))

        # Errors
        all_errors = np.concatenate([
            normal_errors[feat_idx],
            anom_errors[feat_idx]
        ])
        error_min = np.min(all_errors)
        error_max = np.max(all_errors)
        error_margin = (error_max - error_min) * 0.05
        error_ranges.append((error_min - error_margin, error_max + error_margin))

    # Compute range for global error
    all_global_errors = np.concatenate([normal_global_error, anomaly_global_error])
    global_error_min = np.min(all_global_errors)
    global_error_max = np.max([all_global_errors.max(), threshold_95])
    global_error_margin = (global_error_max - global_error_min) * 0.05
    global_error_range = (global_error_min - global_error_margin, global_error_max + global_error_margin)

    # ============================================
    # CALCULATE SEQUENCE BOUNDARIES
    # ============================================
    # Compute where sequence boundaries are (every seq_length timesteps)
    # Relative to start_t
    sequence_boundaries = []
    current_boundary = seq_length
    while current_boundary < max(T_normal, T_anom):
        # Absolute timestep
        absolute_timestep = start_t + current_boundary
        sequence_boundaries.append(absolute_timestep)
        current_boundary += seq_length

    print(
        f"   📏 Sequence boundaries at timesteps: {sequence_boundaries[:5]}{'...' if len(sequence_boundaries) > 5 else ''}")

    # ============================================
    # CREATE SUBPLOTS: Channels + Global Error
    # ============================================
    n_channel_rows = 2 * C  # Normal + Anomaly per channel
    n_global_rows = 1  # One plot for global error (overlaid)
    n_rows = n_channel_rows + n_global_rows
    n_cols = 1

    subplot_titles = []
    for feat_name in feature_names:
        subplot_titles.append(f"{feat_name} - NORMAL")
        subplot_titles.append(f"{feat_name} - ANOMALY")

    # Add title for global error plot
    subplot_titles.append("GLOBAL ERROR (averaged across all channels)")

    # Specs: secondary_y for channel plots, no secondary for global
    specs = [[{"secondary_y": True}]] * n_channel_rows + [[{"secondary_y": False}]]

    # Row heights
    row_heights = [400] * n_channel_rows + [400]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.01,
        specs=specs,
        row_heights=row_heights
    )

    # Time steps
    time_normal = np.arange(start_t, start_t + T_normal)
    time_anom = np.arange(start_t, start_t + T_anom)

    # ============================================
    # PLOT CHANNELS
    # ============================================
    for feat_idx in range(C):
        row_normal = feat_idx * 2 + 1
        row_anom = feat_idx * 2 + 2

        # === NORMAL ===
        fig.add_trace(
            go.Scatter(
                x=time_normal, y=normal_targets[feat_idx],
                mode='lines', name='Target',
                line=dict(color='blue', width=1),
                showlegend=(feat_idx == 0), legendgroup='target'
            ),
            row=row_normal, col=1, secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=time_normal, y=normal_recons[feat_idx],
                mode='lines', name='Reconstruction',
                line=dict(color='cyan', width=1, dash='dash'),
                showlegend=(feat_idx == 0), legendgroup='recon'
            ),
            row=row_normal, col=1, secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=time_normal, y=normal_errors[feat_idx],
                mode='lines', name='Error',
                line=dict(color='orange', width=0.5),
                fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.2)',
                showlegend=(feat_idx == 0), legendgroup='error'
            ),
            row=row_normal, col=1, secondary_y=True
        )

        # === ANOMALY ===
        fig.add_trace(
            go.Scatter(
                x=time_anom, y=anom_targets[feat_idx],
                mode='lines', name='Target',
                line=dict(color='blue', width=1),
                showlegend=False, legendgroup='target'
            ),
            row=row_anom, col=1, secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=time_anom, y=anom_recons[feat_idx],
                mode='lines', name='Reconstruction',
                line=dict(color='cyan', width=1, dash='dash'),
                showlegend=False, legendgroup='recon'
            ),
            row=row_anom, col=1, secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                x=time_anom, y=anom_errors[feat_idx],
                mode='lines', name='Error',
                line=dict(color='orange', width=0.5),
                fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.2)',
                showlegend=False, legendgroup='error'
            ),
            row=row_anom, col=1, secondary_y=True
        )

        # ✅ ADD: Vertical lines for sequence boundaries (NORMAL)
        for boundary in sequence_boundaries:
            if start_t <= boundary < start_t + T_normal:
                fig.add_vline(
                    x=boundary,
                    line_dash="dot",
                    line_color="gray",
                    line_width=1,
                    opacity=0.5,
                    row=row_normal, col=1
                )

        # ✅ ADD: Vertical lines for sequence boundaries (ANOMALY)
        for boundary in sequence_boundaries:
            if start_t <= boundary < start_t + T_anom:
                fig.add_vline(
                    x=boundary,
                    line_dash="dot",
                    line_color="gray",
                    line_width=1,
                    opacity=0.5,
                    row=row_anom, col=1
                )

        # Update axes for channels
        value_range = value_ranges[feat_idx]
        error_range = error_ranges[feat_idx]

        fig.update_yaxes(title_text="Value", row=row_normal, col=1, secondary_y=False, range=value_range)
        fig.update_yaxes(title_text="Error", row=row_normal, col=1, secondary_y=True, range=error_range)
        fig.update_yaxes(title_text="Value", row=row_anom, col=1, secondary_y=False, range=value_range)
        fig.update_yaxes(title_text="Error", row=row_anom, col=1, secondary_y=True, range=error_range)

        if feat_idx == C - 1:
            fig.update_xaxes(title_text="Timestep", row=row_anom, col=1)

    # ============================================
    # PLOT GLOBAL ERROR (OVERLAID)
    # ============================================
    row_global = n_rows

    # Normal global error
    fig.add_trace(
        go.Scatter(
            x=time_normal,
            y=normal_global_error,
            mode='lines',
            name='Normal Global Error',
            line=dict(color='lightblue', width=2),
            fill='tozeroy',
            fillcolor='rgba(173, 216, 230, 0.3)',
            legendgroup='global'
        ),
        row=row_global, col=1
    )

    # Anomaly global error
    fig.add_trace(
        go.Scatter(
            x=time_anom,
            y=anomaly_global_error,
            mode='lines',
            name='Anomaly Global Error',
            line=dict(color='coral', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 127, 80, 0.3)',
            legendgroup='global'
        ),
        row=row_global, col=1
    )

    # ✅ ADD: Vertical lines for sequence boundaries (GLOBAL ERROR)
    for boundary in sequence_boundaries:
        if start_t <= boundary < start_t + max(T_normal, T_anom):
            fig.add_vline(
                x=boundary,
                line_dash="dot",
                line_color="gray",
                line_width=1,
                opacity=0.5,
                row=row_global, col=1
            )

    # Threshold line
    fig.add_hline(
        y=threshold_95,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"95th percentile = {threshold_95:.4f}",
        annotation_position="right",
        row=row_global, col=1
    )

    # Update axes for global error
    fig.update_xaxes(title_text="Timestep", row=row_global, col=1)
    fig.update_yaxes(
        title_text="Global Error (mean across channels)",
        row=row_global, col=1,
        range=global_error_range
    )

    # ============================================
    # UPDATE LAYOUT
    # ============================================
    n_seqs_normal = normal_concat['n_sequences']
    n_seqs_anom = anomaly_concat['n_sequences']

    # Compute statistics for title
    normal_above = (normal_global_error > threshold_95).sum()
    anomaly_above = (anomaly_global_error > threshold_95).sum()

    fig.update_layout(
        title=f"Block {block_idx} - Normal: {n_seqs_normal} seqs ({T_normal} timesteps) | Anomaly: {n_seqs_anom} seqs ({T_anom} timesteps)<br>"
              f"Seq length: {seq_length} | Boundaries every {seq_length} timesteps<br>"
              f"Global: Normal above threshold: {normal_above}/{T_normal} ({100 * normal_above / T_normal:.1f}%) | "
              f"Anomaly above threshold: {anomaly_above}/{T_anom} ({100 * anomaly_above / T_anom:.1f}%)",
        height=sum(row_heights),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom",
            y=1.01, xanchor="right", x=1
        ),
        hovermode='x unified'
    )

    # Save
    output_path = output_dir / f"block_{block_idx:04d}.html"
    fig.write_html(str(output_path))

    print(f"   ✓ Saved block {block_idx}: {output_path}")
    print(f"      Sequence length: {seq_length} | Boundaries added: {len(sequence_boundaries)}")
    print(
        f"      Global error - Normal mean: {normal_global_error.mean():.6f}, Anomaly mean: {anomaly_global_error.mean():.6f}")
    print(f"      Threshold (95th): {threshold_95:.6f}")
    print(f"      Above threshold: Normal {normal_above}/{T_normal} ({100 * normal_above / T_normal:.1f}%), "
          f"Anomaly {anomaly_above}/{T_anom} ({100 * anomaly_above / T_anom:.1f}%)")

    return output_path



def main():
    parser = argparse.ArgumentParser(description='Evaluate reconstruction errors from checkpoint')
    parser.add_argument('--config', type=str, default='./evaluation/evaluation.yaml',
                       help='Path to config YAML file (default: evaluation.yaml)')
    args = parser.parse_args()


    # Load config
    print(f"\n📋 Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    checkpoint_path = config['checkpoint_path']
    metric_dataset_path = config.get('metric_dataset_path', None)
    weighting_factor = config.get('weighting_factor', False)
    epsilon = config.get('epsilon', 1e-3)


    # Load checkpoint
    checkpoint, cfg, checkpoint_metric_path, scaler_params = load_checkpoint(checkpoint_path)
    max_sequences_per_block = int(cfg.dataset.get('seq_in_length_into_chunk', 200)/config.get('seq_in_length_factor', 200))

    print(f"   ✓ checkpoint_path: {checkpoint_path}")
    print(f"   ✓ weighting_factor: {weighting_factor}")
    print(f"   ✓ max_sequences_per_block: {max_sequences_per_block}")

    # Use metric_dataset_path from config, fallback to checkpoint
    if metric_dataset_path is None:
        metric_dataset_path = checkpoint_metric_path
        print(f"   → Using metric_dataset_path from checkpoint")


    if metric_dataset_path is None:
        raise ValueError("❌ metric_dataset_path not found in config or checkpoint!")

    # Load metric dataset
    metric_loader, metadata, metric_dataset = load_metric_dataset(
        metric_dataset_path,
        batch_size=cfg.opt.get('batch_size', 32)
    )

    # Add transform if missing
    if not hasattr(metric_dataset, 'transform') or metric_dataset.transform is None:
        from utils.load_dataset import get_transform
        metric_dataset.transform = get_transform(cfg)

    # Load model
    print(f"\n🤖 Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   ✓ Model loaded on {device}")

    # Compute reconstruction errors
    normal_data, anomaly_data, normalization_factor = compute_reconstruction_errors(
        model=model,
        dataloader=metric_loader,
        device=device,
        use_error=cfg.opt.get('use_error', 'abs'),
        weighting_factor=weighting_factor,
        epsilon=epsilon,
        metric_dataset_path=metric_dataset_path  #
    )
    # Get feature names
    feature_names = cfg.dataset.get('feats', None)
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(cfg.dataset.n_features)]

    # Concatenate sequences
    print(f"\n🔗 Concatenating sequences...")
    n_features = cfg.dataset.n_features
    normal_concat = concatenate_sequences(normal_data, max_sequences=None)
    anomaly_concat = concatenate_sequences(anomaly_data, max_sequences=None)

    if normal_concat is None or anomaly_concat is None:
        raise ValueError("❌ No data to plot!")

    print(f"   ✓ Normal: {normal_concat['n_sequences']} sequences × {normal_concat['seq_length']} timesteps = {normal_concat['targets'].shape[1]} total timesteps")
    print(f"   ✓ Anomaly: {anomaly_concat['n_sequences']} sequences × {anomaly_concat['seq_length']} timesteps = {anomaly_concat['targets'].shape[1]} total timesteps")

    # Create output directory
    output_dir = Path(checkpoint_path).parent / "reconstruction_plots"
    output_dir.mkdir(exist_ok=True)
    print(f"   → Output directory: {output_dir}")

    # Determine how many blocks needed
    seq_len = normal_concat['seq_length']
    max_timesteps_per_block = max_sequences_per_block * seq_len

    total_timesteps = max(normal_concat['targets'].shape[1], anomaly_concat['targets'].shape[1])
    n_blocks = math.ceil(total_timesteps / max_timesteps_per_block)

    print(f"\n📊 Creating {n_blocks} plot blocks (max {max_sequences_per_block} sequences per block)...")

    # Create plots
    for block_idx in tqdm(range(n_blocks), desc="Creating plots"):
        plot_block(
            block_idx=block_idx,
            normal_concat=normal_concat,
            anomaly_concat=anomaly_concat,
            feature_names=feature_names,
            output_dir=output_dir,
            max_timesteps_per_plot=max_timesteps_per_block,
            weighting_factor=weighting_factor
        )

    print(f"\n✅ Done! Plots saved to: {output_dir}")
    print(f"   → {n_blocks} block HTML files created")
    print(f"   → Open any block_XXXX.html file in a browser to view")

    # Print summary statistics
    print(f"\n📈 Summary Statistics:")
    print(f"   Normal sequences:")
    total_errors = normal_data['errors'].sum(dim=(1, 2))
    print(f"      - Mean total error: {total_errors.mean():.6f}")
    print(f"      - Std total error:  {total_errors.std():.6f}")
    print(f"      - Min total error:  {total_errors.min():.6f}")
    print(f"      - Max total error:  {total_errors.max():.6f}")

    print(f"\n   Anomalous sequences:")
    if anomaly_data['errors'].numel() > 0:
        total_errors_anom = anomaly_data['errors'].sum(dim=(1, 2))
        print(f"      - Mean total error: {total_errors_anom.mean():.6f}")
        print(f"      - Std total error:  {total_errors_anom.std():.6f}")
        print(f"      - Min total error:  {total_errors_anom.min():.6f}")
        print(f"      - Max total error:  {total_errors_anom.max():.6f}")

    if weighting_factor and normal_data['errors_normalized'] is not None:
        total_errors_norm = normal_data['errors_normalized'].sum(dim=(1, 2))
        print(f"\n   Normal sequences [Normalized]:")
        print(f"      - Mean total error: {total_errors_norm.mean():.6f}")
        print(f"      - Std total error:  {total_errors_norm.std():.6f}")
        print(f"      - Min total error:  {total_errors_norm.min():.6f}")
        print(f"      - Max total error:  {total_errors_norm.max():.6f}")

        if anomaly_data['errors_normalized'] is not None:
            total_errors_anom_norm = anomaly_data['errors_normalized'].sum(dim=(1, 2))
            print(f"\n   Anomalous sequences [Normalized]:")
            print(f"      - Mean total error: {total_errors_anom_norm.mean():.6f}")
            print(f"      - Std total error:  {total_errors_anom_norm.std():.6f}")
            print(f"      - Min total error:  {total_errors_anom_norm.min():.6f}")
            print(f"      - Max total error:  {total_errors_anom_norm.max():.6f}")


if __name__ == "__main__":
    main()
