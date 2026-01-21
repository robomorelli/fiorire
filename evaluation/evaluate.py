"""
Evaluate script: visualize reconstruction errors from a trained model.
Groups contiguous sequences into blocks and creates one HTML per block.
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

    # Create DataLoader
    metric_loader = DataLoader(
        metric_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return metric_loader, metadata, metric_dataset


def identify_contiguous_blocks(indices, seq_len, perc_overlap=0):
    """
    Identify contiguous blocks of sequences based on their indices.

    Args:
        indices: Array of starting indices for sequences
        seq_len: Sequence length
        perc_overlap: Percentage overlap between consecutive sequences

    Returns:
        blocks: List of lists, each containing indices of sequences in a contiguous block
    """
    if len(indices) == 0:
        return []

    # Calculate expected step between consecutive sequences
    if perc_overlap == 0:
        expected_step = seq_len
    else:
        expected_step = int(seq_len * (1 - perc_overlap))

    # Sort indices
    sorted_indices = np.sort(indices)

    # Identify blocks
    blocks = []
    current_block = [0]  # Start with first sequence index (in sorted array)

    for i in range(1, len(sorted_indices)):
        actual_step = sorted_indices[i] - sorted_indices[i-1]

        # Check if this sequence is contiguous with previous
        # Allow some tolerance (±20% of expected_step)
        tolerance = max(1, int(expected_step * 0.2))

        if abs(actual_step - expected_step) <= tolerance:
            # Contiguous - add to current block
            current_block.append(i)
        else:
            # Gap detected - start new block
            blocks.append(current_block)
            current_block = [i]

    # Add last block
    if current_block:
        blocks.append(current_block)

    print(f"\n🔍 Block Analysis:")
    print(f"   ✓ Total sequences: {len(indices)}")
    print(f"   ✓ Expected step: {expected_step}")
    print(f"   ✓ Identified blocks: {len(blocks)}")

    # Print block details
    for block_idx, block in enumerate(blocks):
        block_indices = sorted_indices[block]
        print(f"   → Block {block_idx}: {len(block)} sequences "
              f"(indices {block_indices[0]} to {block_indices[-1]})")

    return blocks, sorted_indices


def compute_reconstruction_errors(model, dataloader, device='cpu', use_error='abs',
                                   weighting_factor=False, epsilon=1e-3):
    """
    Compute reconstruction errors for all sequences.
    Separates normal and anomalous sequences.
    """
    model.eval()
    model.to(device)

    normal_inputs, normal_targets, normal_recons, normal_errors = [], [], [], []
    anom_inputs, anom_targets, anom_recons, anom_errors, anom_masks = [], [], [], [], []

    print(f"\n🔍 Computing reconstructions (weighting_factor={weighting_factor})...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            x, target, mask, *_ = batch
            x = x.to(device)
            target = target.to(device)

            # Forward pass
            recon = model(x)

            # Compute errors
            if use_error == 'abs':
                err = torch.abs(target - recon)
            else:  # 'se'
                err = (target - recon) ** 2

            # Separate normal and anomalous
            is_anom = mask.view(mask.size(0), -1).sum(dim=1) > 0
            is_norm = ~is_anom

            # Store normals
            if is_norm.any():
                normal_inputs.append(x[is_norm].cpu())
                normal_targets.append(target[is_norm].cpu())
                normal_recons.append(recon[is_norm].cpu())
                normal_errors.append(err[is_norm].cpu())

            # Store anomalies
            if is_anom.any():
                anom_inputs.append(x[is_anom].cpu())
                anom_targets.append(target[is_anom].cpu())
                anom_recons.append(recon[is_anom].cpu())
                anom_errors.append(err[is_anom].cpu())
                anom_masks.append(mask[is_anom].cpu())

    # Concatenate
    normal_errors_tensor = torch.cat(normal_errors, dim=0) if normal_errors else torch.tensor([])
    anomaly_errors_tensor = torch.cat(anom_errors, dim=0) if anom_errors else torch.tensor([])

    # Apply weighting_factor (normalization)
    normalization_factor = None
    normal_errors_normalized = None
    anomaly_errors_normalized = None

    if weighting_factor and normal_errors_tensor.numel() > 0:
        print(f"\n🔧 Applying weighting_factor (normalization with median)...")

        normal_perm = normal_errors_tensor.permute(0, 2, 1)  # [N, C, T] → [N, T, C]

        C = normal_perm.shape[2]
        flat_norm = normal_perm.reshape(-1, C).float()
        normalization_factor = torch.quantile(flat_norm, 0.5, dim=0)  # [C]

        print(f"   ✓ Normalization factor (median per feature):")
        print(f"      - Shape: {normalization_factor.shape}")
        print(f"      - Min:   {normalization_factor.min():.6f}")
        print(f"      - Max:   {normalization_factor.max():.6f}")
        print(f"      - Mean:  {normalization_factor.mean():.6f}")

        norm = normalization_factor.view(1, 1, C) + epsilon
        normal_errors_normalized = (normal_perm / norm).permute(0, 2, 1)

        if anomaly_errors_tensor.numel() > 0:
            anomaly_perm = anomaly_errors_tensor.permute(0, 2, 1)
            anomaly_errors_normalized = (anomaly_perm / norm).permute(0, 2, 1)

        print(f"   ✓ Errors normalized")

    # Build return dictionaries
    normal_data = {
        'inputs': torch.cat(normal_inputs, dim=0) if normal_inputs else torch.tensor([]),
        'targets': torch.cat(normal_targets, dim=0) if normal_targets else torch.tensor([]),
        'reconstructions': torch.cat(normal_recons, dim=0) if normal_recons else torch.tensor([]),
        'errors': normal_errors_tensor,
        'errors_normalized': normal_errors_normalized
    }

    anomaly_data = {
        'inputs': torch.cat(anom_inputs, dim=0) if anom_inputs else torch.tensor([]),
        'targets': torch.cat(anom_targets, dim=0) if anom_targets else torch.tensor([]),
        'reconstructions': torch.cat(anom_recons, dim=0) if anom_recons else torch.tensor([]),
        'errors': anomaly_errors_tensor,
        'errors_normalized': anomaly_errors_normalized,
        'masks': torch.cat(anom_masks, dim=0) if anom_masks else torch.tensor([]),
    }

    print(f"   ✓ Normal sequences: {normal_data['inputs'].shape[0]}")
    print(f"   ✓ Anomalous sequences: {anomaly_data['inputs'].shape[0]}")

    return normal_data, anomaly_data, normalization_factor


def plot_contiguous_block(block_idx, block_seq_indices, targets, recons, errors, errors_normalized,
                          sorted_indices, output_dir, feature_names=None, weighting_factor=False):
    """
    Create interactive Plotly plot for a contiguous block of sequences.
    Shows all sequences in the block stacked vertically.

    Args:
        block_idx: Block index
        block_seq_indices: Indices of sequences in this block (positions in sorted array)
        targets: All target sequences [N, C, T]
        recons: All reconstructed sequences [N, C, T]
        errors: All error sequences [N, C, T]
        errors_normalized: Normalized errors (if weighting_factor=True)
        sorted_indices: Sorted array of dataset indices
        output_dir: Output directory
        feature_names: List of feature names
        weighting_factor: If True, show normalized errors
    """
    n_sequences = len(block_seq_indices)

    # Get actual dataset indices for this block
    dataset_indices = sorted_indices[block_seq_indices]

    # Extract data for all sequences in block
    block_targets = targets[block_seq_indices]  # [n_seq, C, T]
    block_recons = recons[block_seq_indices]
    block_errors = errors[block_seq_indices]

    if errors_normalized is not None:
        block_errors_normalized = errors_normalized[block_seq_indices]
    else:
        block_errors_normalized = None

    # Handle shape
    if block_targets.dim() == 4:  # [n_seq, 1, C, T]
        block_targets = block_targets.squeeze(1)
        block_recons = block_recons.squeeze(1)
        block_errors = block_errors.squeeze(1)
        if block_errors_normalized is not None:
            block_errors_normalized = block_errors_normalized.squeeze(1)

    # Transpose to [n_seq, T, C]
    if block_targets.shape[1] < block_targets.shape[2]:
        block_targets = block_targets.permute(0, 2, 1)
        block_recons = block_recons.permute(0, 2, 1)
        block_errors = block_errors.permute(0, 2, 1)
        if block_errors_normalized is not None:
            block_errors_normalized = block_errors_normalized.permute(0, 2, 1)

    n_seq, T, C = block_targets.shape

    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(C)]

    # Create subplots: one row per sequence, showing all features + error
    n_rows = n_sequences
    n_cols = C + (2 if weighting_factor else 1)  # Features + error plots

    subplot_titles = []
    for seq_idx in range(n_sequences):
        ds_idx = dataset_indices[seq_idx]
        for feat_idx in range(C):
            subplot_titles.append(f"Seq {seq_idx} (idx={ds_idx}): {feature_names[feat_idx]}")
        subplot_titles.append(f"Seq {seq_idx}: Total Error (Raw)")
        if weighting_factor:
            subplot_titles.append(f"Seq {seq_idx}: Total Error (Norm)")

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.02,
        horizontal_spacing=0.05
    )

    time_steps = np.arange(T)

    # Plot each sequence
    for seq_idx in range(n_sequences):
        row = seq_idx + 1

        # Plot each feature
        for feat_idx in range(C):
            col = feat_idx + 1

            # Input (target)
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=block_targets[seq_idx, :, feat_idx],
                    mode='lines',
                    name='Input',
                    line=dict(color='blue', width=1.5),
                    showlegend=(seq_idx == 0 and feat_idx == 0),
                    legendgroup='input'
                ),
                row=row, col=col
            )

            # Reconstruction
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=block_recons[seq_idx, :, feat_idx],
                    mode='lines',
                    name='Reconstruction',
                    line=dict(color='red', width=1.5, dash='dash'),
                    showlegend=(seq_idx == 0 and feat_idx == 0),
                    legendgroup='recon'
                ),
                row=row, col=col
            )

        # Plot total error (raw)
        total_error = block_errors[seq_idx].sum(axis=1)  # Sum over features
        col = C + 1

        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=total_error,
                mode='lines',
                name='Error (Raw)',
                line=dict(color='orange', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(255, 165, 0, 0.2)',
                showlegend=(seq_idx == 0),
                legendgroup='error_raw'
            ),
            row=row, col=col
        )

        # Plot total error (normalized) if enabled
        if weighting_factor and block_errors_normalized is not None:
            total_error_norm = block_errors_normalized[seq_idx].sum(axis=1)
            col = C + 2

            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=total_error_norm,
                    mode='lines',
                    name='Error (Norm)',
                    line=dict(color='purple', width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(128, 0, 128, 0.2)',
                    showlegend=(seq_idx == 0),
                    legendgroup='error_norm'
                ),
                row=row, col=col
            )

    # Update layout
    fig.update_layout(
        title=f"Block {block_idx} - {n_sequences} Contiguous Sequences (indices {dataset_indices[0]}-{dataset_indices[-1]})",
        height=250 * n_rows,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )

    # Update axes
    for row in range(1, n_rows + 1):
        for col in range(1, n_cols + 1):
            if col <= C:
                fig.update_yaxes(title_text="Value", row=row, col=col)
            else:
                fig.update_yaxes(title_text="Error", row=row, col=col)

            if row == n_rows:
                fig.update_xaxes(title_text="Time Step", row=row, col=col)

    # Save
    output_path = output_dir / f"block_{block_idx:04d}.html"
    fig.write_html(str(output_path))

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Evaluate reconstruction errors from checkpoint')
    parser.add_argument('--config', type=str, default='evaluation.yaml',
                       help='Path to config YAML file (default: evaluation.yaml)')
    args = parser.parse_args()

    # Load config
    print(f"\n📋 Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    checkpoint_path = config['checkpoint_path']
    perc_plot = config.get('perc_plot', 0.05)
    metric_dataset_path = config.get('metric_dataset_path', None)
    weighting_factor = config.get('weighting_factor', False)
    epsilon = config.get('epsilon', 1e-3)

    print(f"   ✓ checkpoint_path: {checkpoint_path}")
    print(f"   ✓ perc_plot: {perc_plot}")
    print(f"   ✓ weighting_factor: {weighting_factor}")

    # Load checkpoint
    checkpoint, cfg, checkpoint_metric_path, scaler_params = load_checkpoint(checkpoint_path)

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

    # Extract indices from dataset
    if hasattr(metric_dataset, 'indices'):
        dataset_indices = np.array(metric_dataset.indices)
    else:
        # Fallback: assume sequential indices
        dataset_indices = np.arange(len(metric_dataset))
        print(f"   ⚠️  Dataset has no 'indices' attribute, using sequential indices")

    # Get sequence parameters
    seq_len = cfg.dataset.seq_in_length
    perc_overlap = cfg.dataset.get('perc_overlap', 0)

    print(f"\n🔍 Sequence parameters:")
    print(f"   ✓ seq_len: {seq_len}")
    print(f"   ✓ perc_overlap: {perc_overlap}")

    # Identify contiguous blocks
    blocks, sorted_indices = identify_contiguous_blocks(
        dataset_indices,
        seq_len=seq_len,
        perc_overlap=perc_overlap
    )

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
        epsilon=epsilon
    )

    # Get feature names
    feature_names = cfg.dataset.get('feats', None)
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(cfg.dataset.n_features)]

    # Determine how many blocks to plot
    n_blocks = len(blocks)
    n_blocks_to_plot = max(1, math.ceil(n_blocks * perc_plot))

    print(f"\n📊 Creating plots for {n_blocks_to_plot}/{n_blocks} blocks ({perc_plot*100:.1f}%)")

    # Create output directory
    output_dir = Path(checkpoint_path).parent / "reconstruction_plots"
    output_dir.mkdir(exist_ok=True)
    print(f"   → Output directory: {output_dir}")

    # Sample blocks to plot
    np.random.seed(42)
    if n_blocks_to_plot < n_blocks:
        blocks_to_plot_indices = np.random.choice(n_blocks, size=n_blocks_to_plot, replace=False)
        blocks_to_plot_indices = np.sort(blocks_to_plot_indices)
    else:
        blocks_to_plot_indices = np.arange(n_blocks)

    # Create plots
    for block_plot_idx, block_idx in enumerate(tqdm(blocks_to_plot_indices, desc="Creating block plots")):
        block = blocks[block_idx]

        # Create plot for this block
        output_path = plot_contiguous_block(
            block_idx=block_idx,
            block_seq_indices=block,
            targets=normal_data['targets'].numpy(),
            recons=normal_data['reconstructions'].numpy(),
            errors=normal_data['errors'].numpy(),
            errors_normalized=normal_data['errors_normalized'].numpy() if normal_data['errors_normalized'] is not None else None,
            sorted_indices=sorted_indices,
            output_dir=output_dir,
            feature_names=feature_names,
            weighting_factor=weighting_factor
        )

    print(f"\n✅ Done! Plots saved to: {output_dir}")
    print(f"   → {n_blocks_to_plot} block HTML files created")
    print(f"   → Open any block_XXXX.html file in a browser to view")

    # Print summary statistics
    print(f"\n📈 Summary Statistics:")
    print(f"   Normal sequences:")
    total_errors = normal_data['errors'].sum(dim=(1, 2))
    print(f"      - Mean total error: {total_errors.mean():.6f}")
    print(f"      - Std total error:  {total_errors.std():.6f}")
    print(f"      - Min total error:  {total_errors.min():.6f}")
    print(f"      - Max total error:  {total_errors.max():.6f}")

    if weighting_factor and normal_data['errors_normalized'] is not None:
        total_errors_norm = normal_data['errors_normalized'].sum(dim=(1, 2))
        print(f"      [Normalized]")
        print(f"      - Mean total error: {total_errors_norm.mean():.6f}")
        print(f"      - Std total error:  {total_errors_norm.std():.6f}")
        print(f"      - Min total error:  {total_errors_norm.min():.6f}")
        print(f"      - Max total error:  {total_errors_norm.max():.6f}")


if __name__ == "__main__":
    main()