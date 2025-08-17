#!/usr/bin/env python3
"""
Coordinate Check Script for μP (Maximal Update Parameterization)

This script runs coordinate checks to verify that μP is implemented correctly
by testing models with different widths and ensuring activations remain stable.

Usage:
    python coords_check.py --widths 0.5,1.0,2.0,4.0 --steps 10
"""

import argparse
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from config_utils import load_config
from data.dataset_utils import create_train_loader
from nn.transfomer.model.gpt_model import GPTModel
from scaling.csv_logging import CSVLogWrapper


def get_batch_from_loader(data_loader, device):
    batch = next(iter(data_loader))
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        input_batch, target_batch = batch
        return input_batch.to(device), target_batch.to(device)
    
    return batch.to(device), None


def setup_coord_check_hooks(model, coord_check_dict):
    def hook(layer_output, key):
        with torch.no_grad():
            if isinstance(layer_output, torch.Tensor):
                coord_check_dict[key].append(layer_output.abs().mean().item())
            elif isinstance(layer_output, (list, tuple)) and len(layer_output) > 0:
                coord_check_dict[key].append(layer_output[0].abs().mean().item())

    def create_hook(key):
        return lambda module, module_input, output: hook(output, key)

    handles = []
    for module_name, module in model.named_modules():
        if any(pattern in module_name for pattern in ['embed', 'wte']):
            handles.append(module.register_forward_hook(create_hook('embedding')))
        elif 'attn' in module_name and not any(skip in module_name for skip in ['proj', 'fc']):
            handles.append(module.register_forward_hook(create_hook('attention')))
        elif 'mlp' in module_name or 'feed_forward' in module_name:
            handles.append(module.register_forward_hook(create_hook('mlp')))
        elif 'head' in module_name or module_name.endswith('lm_head'):
            handles.append(module.register_forward_hook(create_hook('output')))

    return handles


def run_coord_check(base_config, widths, num_steps=10, device='cuda'):
    """
    Run coordinate check across different model widths

    Args:
        base_config: Base configuration dictionary
        widths: List of width multipliers to test
        num_steps: Number of forward passes to run
        device: Device to run on
    """
    results = defaultdict(lambda: defaultdict(list))

    train_loader, _ = create_train_loader(base_config, distributed=False)

    for width_mult in widths:
        print(f"\nTesting width multiplier: {width_mult}")

        config = base_config.copy()
        config['mup'] = True
        config['mup_width_multiplier'] = width_mult

        if 'd_model' in config:
            base_width = config['d_model']
            config['d_model'] = int(base_width * width_mult)
        if 'hidden_size' in config:
            base_hidden = config['hidden_size']
            config['hidden_size'] = int(base_hidden * width_mult)

        model = GPTModel(config).to(device)
        model.eval()

        coord_check_dict = defaultdict(list)
        handles = setup_coord_check_hooks(model, coord_check_dict)

        with torch.no_grad():
            for step in range(num_steps):
                try:
                    input_batch, target_batch = get_batch_from_loader(train_loader, device)
                    if target_batch is not None:
                        _ = model(input_batch, target_batch)
                    else:
                        _ = model(input_batch)

                    if step % 5 == 0:
                        print(f"  Step {step}/{num_steps}")

                except Exception as e:
                    print(f"Warning: Forward pass failed for width {width_mult}: {e}")
                    break

        for handle in handles:
            handle.remove()

        for key, values in coord_check_dict.items():
            if values:
                results[key][width_mult] = np.mean(values)

        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    return dict(results)


def plot_results(results, save_path='coords_check_results.png'):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    layer_types = ['embedding', 'attention', 'mlp', 'output']

    for i, layer_type in enumerate(layer_types):
        ax = axes[i]

        if layer_type in results:
            widths = sorted(results[layer_type].keys())
            values = [results[layer_type][w] for w in widths]

            ax.loglog(widths, values, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Width Multiplier')
            ax.set_ylabel('Mean Absolute Activation')
            ax.set_title(f'{layer_type.title()} Layer Activations')
            ax.grid(True, alpha=0.3)

            if values:
                ideal_value = np.median(values)
                ax.axhline(y=ideal_value, color='red', linestyle='--', alpha=0.7,
                          label=f'Ideal (constant at {ideal_value:.3f})')
                ax.legend()
        else:
            ax.text(0.5, 0.5, f'No data for\n{layer_type}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{layer_type.title()} Layer (No Data)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Results saved to {save_path}")


def log_results_to_csv(results, out_dir='coords_check_output'):
    logger = CSVLogWrapper(out_dir=out_dir, config={'coord_check': True})

    all_widths = set()
    for layer_data in results.values():
        all_widths.update(layer_data.keys())
    widths = sorted(all_widths)

    for width in widths:
        log_data = {'width_multiplier': width}
        for layer_type, layer_data in results.items():
            if width in layer_data:
                log_data[f'{layer_type}_activation'] = layer_data[width]
        logger.log(log_data)
        logger.step()

    logger.close()
    print(f"CSV logs saved to {out_dir}/log.csv")


def save_results(results, save_path='coords_check_data.pkl'):
    """Save results to pickle file"""
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Data saved to {save_path}")


def print_summary(results):
    """Print a summary of the coordination check results"""
    print("\n" + "="*60)
    print("COORDINATION CHECK SUMMARY")
    print("="*60)
    
    for layer_type, width_data in results.items():
        if not width_data:
            continue
            
        print(f"\n{layer_type.upper()} LAYER:")
        widths = sorted(width_data.keys())
        values = [width_data[w] for w in widths]

        cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else float('inf')



def main():
    parser = argparse.ArgumentParser(description='Run μP coordination check')
    parser.add_argument('--widths', type=str, default='0.5,1.0,2.0,4.0',
                       help='Comma-separated width multipliers to test')
    parser.add_argument('--steps', type=int, default=10,
                       help='Number of forward passes per width')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--save_plot', type=str, default='coords_check_results.png',
                       help='Path to save plot')
    parser.add_argument('--save_data', type=str, default='coords_check_data.pkl',
                       help='Path to save data')
    parser.add_argument('--csv_dir', type=str, default='coords_check_output',
                       help='Directory to save CSV logs')

    args = parser.parse_args()

    widths = [float(w.strip()) for w in args.widths.split(',')]

    try:
        base_config = load_config()
    except Exception as e:
        print(f"Failed to load config: {e}")
        print("Using default configuration...")
        base_config = {
            'batch_size': 8,
            'max_seq_length': 128,
            'd_model': 256,
            'hidden_size': 256,
            'n_head': 8,
            'n_layer': 6,
            'vocab_size': 50257,
            'mup': True,
            'train_data': 'data/sample.txt',  
        }

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    print(f"Running coordination check with widths: {widths}")
    print(f"Steps per width: {args.steps}")
    print(f"Device: {args.device}")

    results = run_coord_check(base_config, widths, args.steps, args.device)

    print_summary(results)

    if results:
        save_results(results, args.save_data)
        log_results_to_csv(results, args.csv_dir)
        plot_results(results, args.save_plot)
    else:
        print("No results to save - check your model and data configuration")


if __name__ == '__main__':
    main()