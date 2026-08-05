import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

import os
from datetime import datetime



def load_data():

    train_x = np.load(f'dataset/x_train.npy') 
    train_y = np.load(f'dataset/y_train.npy')
    val_x = np.load(f'dataset/x_val.npy')   
    val_y = np.load(f'dataset/y_val.npy')
    test_x = np.load(f'dataset/x_test.npy')  
    test_y = np.load(f'dataset/y_test.npy')

    print ("Len train: ", len(train_x))
    print ("Len val: ", len(val_x))
    print ("Len test: ", len(test_x))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 2. Plot the first image (Original) on the first axis (axes[0])
    axes[0].imshow(train_x[0], cmap='gray')
    axes[0].set_title("Train")
    axes[0].axis('off')

    # 3. Plot the second image (Processed) on the second axis (axes[1])
    # Note: if using the MLP version, remember to reshape it back to (32,32)
    axes[1].imshow(test_x[0], cmap='gray') 
    axes[1].set_title("Test")
    axes[1].axis('off')

    # 4. Show the entire figure once at the end
    plt.tight_layout()
    #plt.show()

    # 2. Convert to PyTorch Tensors AND normalize pixels to [0.0, 1.0]
    # Labels must be torch.long for CrossEntropyLoss
    train_x_t = torch.tensor(train_x, dtype=torch.float32) / 255.0
    train_y_t = torch.tensor(train_y, dtype=torch.long)

    #print (train_x_t.shape)

    val_x_t   = torch.tensor(val_x, dtype=torch.float32) / 255.0
    val_y_t   = torch.tensor(val_y, dtype=torch.long)

    test_x_t  = torch.tensor(test_x, dtype=torch.float32) / 255.0
    test_y_t  = torch.tensor(test_y, dtype=torch.long)

    # 3. Create Datasets
    train_dataset = TensorDataset(train_x_t, train_y_t)
    val_dataset   = TensorDataset(val_x_t, val_y_t)
    test_dataset  = TensorDataset(test_x_t, test_y_t)

    # 4. Create DataLoaders
    batch_size = 128

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print (f"Batch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader


import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def plot_stats(train_loss_hists,  
               val_loss_hists,
               test_acc_hists,
               mean_spike_rate_per_layer_hists,
               grad_norm_per_layer_hists,
               layer_indices,
               run_names,    
               num_inputs,
               num_hidden,
               n_hidden_layers,
               num_steps,
               beta,
               activation_function,
               slope,
               lr,
               betas):

    # 1. Create the "figures" directory
    os.makedirs("figures", exist_ok=True)

    # 2. Timestamp and filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"figures/stats_{num_inputs}_{num_hidden}_{n_hidden_layers}_{num_steps}_{activation_function}_{beta}_{slope}_{lr}_{betas}_{timestamp}.png"
    
    hp_text = (
        f"Input dim: {num_inputs} | Hidden dim: {num_hidden} | Number of hidden layers: {n_hidden_layers} |  Steps: {num_steps}  |  "
        f"Beta: {beta}  | Surrogate activation func: {activation_function} | Slope: {slope}  |  LR: {lr}  |  Adam betas: {betas}"
    )

    # 3. Define colors for the 4 different runs/decoding methods
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']

    # Create a figure with 4 vertically stacked subplots, sharing the X-axis (Iterations)
    fig, (ax_loss, ax_acc, ax_spike, ax_grad) = plt.subplots(nrows=4, ncols=1, figsize=(12, 16), sharex=True)

    # Define line styles for layers in bottom plots: L0 = solid, L1 = dashed
    line_styles = ['-', '--']

    # Plotting loop for the runs
    for i in range(len(train_loss_hists)):
        color = colors[i % len(colors)]
        run_label = run_names[i]
        
        # --- SUBPLOT 1: Train Loss & Val Loss ---
        ax_loss.plot(train_loss_hists[i], color=color, 
                     linestyle='-', linewidth=1.5, 
                     label=f'{run_label} train loss')
        
        ax_loss.plot(val_loss_hists[i], color=color, 
                     linestyle='--', linewidth=1.5, 
                     label=f'{run_label} val loss')

        # --- SUBPLOT 2: Test Accuracy ---
        ax_acc.plot(test_acc_hists[i], color=color, 
                    linestyle='-', linewidth=2, 
                    label=f'{run_label} acc')

        # --- SUBPLOT 3: Mean Spike Rates per Layer ---
        if isinstance(mean_spike_rate_per_layer_hists, list) and len(mean_spike_rate_per_layer_hists) == len(train_loss_hists):
            layer_rates = mean_spike_rate_per_layer_hists[i]
        else:
            layer_rates = mean_spike_rate_per_layer_hists  
        
        if isinstance(layer_rates, (list, np.ndarray)) and len(layer_rates) > 0:
            if isinstance(layer_rates[0], (list, np.ndarray, tuple)):
                layer_streams = list(zip(*layer_rates))
                for layer_idx, rates in enumerate(layer_streams):
                    ls = line_styles[layer_idx % len(line_styles)]
                    ax_spike.plot(rates, color=color, linestyle=ls, alpha=0.8, linewidth=1.5,
                                  label=f'{run_label} (L{layer_indices[layer_idx]})')
            else:
                ax_spike.plot(layer_rates, color=color, linestyle='-', alpha=0.8,
                              label=f'{run_label}')

        # --- SUBPLOT 4: Gradient Norms per Layer ---
        if isinstance(grad_norm_per_layer_hists, list) and len(grad_norm_per_layer_hists) == len(train_loss_hists):
            grad_norms = grad_norm_per_layer_hists[i]
        else:
            grad_norms = grad_norm_per_layer_hists  
        
        if isinstance(grad_norms, (list, np.ndarray)) and len(grad_norms) > 0:
            if isinstance(grad_norms[0], (list, np.ndarray, tuple)):
                grad_streams = list(zip(*grad_norms))
                for layer_idx, norms in enumerate(grad_streams):
                    ls = line_styles[layer_idx % len(line_styles)]
                    ax_grad.plot(norms, color=color, linestyle=ls, alpha=0.8, linewidth=1.5,
                                 label=f'{run_label} (L{layer_indices[layer_idx]})')
            else:
                ax_grad.plot(grad_norms, color=color, linestyle='-', alpha=0.8,
                             label=f'{run_label}')

    # --- Formatting Subplot 1 (Losses) ---
    ax_loss.set_ylabel('Cross Entropy Loss', fontsize=12)
    ax_loss.set_title("Comparison of Readouts: Losses, Val Accuracy, Layer Spike Rates, and Gradient Norms", fontsize=13)
    ax_loss.grid(True, linestyle='--', alpha=0.5)
    ax_loss.legend(loc='lower left', fontsize=6, ncol=2)

    # --- Formatting Subplot 2 (Accuracy) ---
    ax_acc.set_ylabel('Val Accuracy', fontsize=12)
    ax_acc.grid(True, linestyle='--', alpha=0.5)
    ax_acc.legend(loc='upper left', fontsize=6, ncol=2)

    # --- Formatting Subplot 3 (Spike Rates) ---
    ax_spike.set_ylabel('Mean Spike Rate', fontsize=12)
    ax_spike.grid(True, linestyle='--', alpha=0.5)
    ax_spike.legend(loc='upper left', fontsize=6, ncol=2)

    # --- Formatting Subplot 4 (Gradient Norms) ---
    ax_grad.set_xlabel('Iteration', fontsize=12)
    ax_grad.set_ylabel('Gradient Norm (Frobenius)', fontsize=12)
    ax_grad.grid(True, linestyle='--', alpha=0.5)
    ax_grad.legend(loc='upper left', fontsize=6, ncol=2)

    # Make room at the bottom of the figure for the hyperparameter text box
    fig.subplots_adjust(bottom=0.10)

    # Hyperparams text box placed centrally at the bottom
    fig.text(0.5, 0.01, hp_text, 
            fontsize=8, 
            verticalalignment='bottom', 
            horizontalalignment='center', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(filename)
    print(f"Saved comparison results to: {filename}")
    # plt.show()

def plot_stats1(train_loss_hists,  
               val_loss_hists,
               test_acc_hists,
               mean_spike_rate_per_layer_hists,
               grad_norm_per_layer_hists,
               layer_indices,
               run_names,    
               num_inputs,
               num_hidden,
               n_hidden_layers,
               num_steps,
               beta,
               activation_function,
               slope,
               lr,
               betas):

    # 1. Create the "figures" directory
    os.makedirs("figures", exist_ok=True)

    # 2. Timestamp and filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"figures/stats_{num_inputs}_{num_hidden}_{n_hidden_layers}_{num_steps}_{activation_function}_{beta}_{slope}_{lr}_{betas}_{timestamp}.png"
    
    hp_text = (
        f"Input dim: {num_inputs} | Hidden dim: {num_hidden} | Number of hidden layers: {n_hidden_layers} |  Steps: {num_steps}  |  "
        f"Beta: {beta}  | Surrogate activation func: {activation_function} | Slope: {slope}  |  LR: {lr}  |  Adam betas: {betas}"
    )

    # 3. Define colors for the 4 different runs/decoding methods
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']

    # Create a figure with 3 vertically stacked subplots, sharing the X-axis (Iterations)
    fig, (ax1, ax3, ax4) = plt.subplots(nrows=3, ncols=1, figsize=(12, 14), sharex=True)

    # Define line styles for layers in bottom plots: L0 = solid, L1 = dashed
    line_styles = ['-', '--']

    # Plotting loop for the runs
    for i in range(len(train_loss_hists)):
        color = colors[i % len(colors)]
        run_label = run_names[i]
        
        # --- TOP PLOT: Train Loss, Val Loss & Test Accuracy ---
        # Train loss: Solid line
        ax1.plot(train_loss_hists[i], color=color, 
                 linestyle='-', linewidth=1.5, 
                 label=f'{run_label} train loss')
        
        # Val loss: Dotted line (neither solid nor dashed)
        ax1.plot(val_loss_hists[i], color=color, 
                 linestyle=':', linewidth=1.5, 
                 label=f'{run_label} val loss')
        
        if i == 0:
            ax2 = ax1.twinx()

        # Test Accuracy: Dashed line (left as is)
        ax2.plot(test_acc_hists[i], color=color, 
                linestyle='--', linewidth=2, 
                label=f'{run_label} acc')

        # --- MIDDLE PLOT: Mean Spike Rates per Layer ---
        if isinstance(mean_spike_rate_per_layer_hists, list) and len(mean_spike_rate_per_layer_hists) == len(train_loss_hists):
            layer_rates = mean_spike_rate_per_layer_hists[i]
        else:
            layer_rates = mean_spike_rate_per_layer_hists  
        
        if isinstance(layer_rates, (list, np.ndarray)) and len(layer_rates) > 0:
            if isinstance(layer_rates[0], (list, np.ndarray, tuple)):
                layer_streams = list(zip(*layer_rates))
                for layer_idx, rates in enumerate(layer_streams):
                    ls = line_styles[layer_idx % len(line_styles)]
                    ax3.plot(rates, color=color, linestyle=ls, alpha=0.8, linewidth=1.5,
                             label=f'{run_label} (L{layer_indices[layer_idx]})')
            else:
                ax3.plot(layer_rates, color=color, linestyle='-', alpha=0.8,
                         label=f'{run_label}')

        # --- BOTTOM PLOT: Gradient Norms per Layer ---
        if isinstance(grad_norm_per_layer_hists, list) and len(grad_norm_per_layer_hists) == len(train_loss_hists):
            grad_norms = grad_norm_per_layer_hists[i]
        else:
            grad_norms = grad_norm_per_layer_hists  
        
        if isinstance(grad_norms, (list, np.ndarray)) and len(grad_norms) > 0:
            if isinstance(grad_norms[0], (list, np.ndarray, tuple)):
                grad_streams = list(zip(*grad_norms))
                for layer_idx, norms in enumerate(grad_streams):
                    ls = line_styles[layer_idx % len(line_styles)]
                    ax4.plot(norms, color=color, linestyle=ls, alpha=0.8, linewidth=1.5,
                             label=f'{run_label} (L{layer_indices[layer_idx]})')
            else:
                ax4.plot(grad_norms, color=color, linestyle='-', alpha=0.8,
                         label=f'{run_label}')

    # --- Formatting Top Subplot ---
    ax1.set_ylabel('Loss', color='black', fontsize=12)
    ax2.set_ylabel('Test Accuracy', color='black', fontsize=12)
    ax1.set_title("Comparison of Readouts: Train and Val Losses, Test Accuracy, Layer Spike Rates, and Gradient Norms", fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Using ncol=2 so the expanded legend fits cleanly
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=6, ncol=2)

    # --- Formatting Middle Subplot (Spike Rates) ---
    ax3.set_ylabel('Mean Spike Rate', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper left', fontsize=6, ncol=2)

    # --- Formatting Bottom Subplot (Gradient Norms) ---
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Gradient Norm (Frobenius)', fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend(loc='upper left', fontsize=6, ncol=2)

    # Make room at the bottom of the figure for the hyperparameter text box
    fig.subplots_adjust(bottom=0.12)

    # Hyperparams text box placed centrally at the bottom
    fig.text(0.5, 0.01, hp_text, 
            fontsize=8, 
            verticalalignment='bottom', 
            horizontalalignment='center', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(filename)
    print(f"Saved comparison results to: {filename}")
    # plt.show()

