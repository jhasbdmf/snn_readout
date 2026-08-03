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
import matplotlib.pyplot as plt
from datetime import datetime

def plot_stats(train_loss_hists,  
               test_acc_hists,
               run_names,    
               num_inputs,
               num_hidden,
               num_steps,
               beta,
               slope,
               lr,
               betas):

    # 1. Create the "figures" directory
    os.makedirs("figures", exist_ok=True)

    # 2. Timestamp and filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"figures/stats_{num_inputs}_{num_hidden}_{num_steps}_{beta}_{slope}_{lr}_{betas}_{timestamp}.png"
    
    hp_text = (
        f"Input dim: {num_inputs}  |  Hidden dim: {num_hidden}  |  Steps: {num_steps}  |  "
        f"Beta: {beta}  |  Slope: {slope}  |  LR: {lr}  |  Adam betas: {betas}"
    )

    # 3. Define colors for the 4 different runs
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plotting loop
    for i in range(len(train_loss_hists)):
        color = colors[i % len(colors)]
        
        # Plot Loss (Solid lines)
        ax1.plot(train_loss_hists[i], color=color, 
                 linestyle='-', linewidth=1.5, 
                 label=f'{run_names[i]} loss')
        
        # Plot Accuracy on the second axis (Dashed lines)
        if i == 0:
            ax2 = ax1.twinx()

        ax2.plot(test_acc_hists[i], color=color, 
                linestyle='--', linewidth=2, 
                label=f'{run_names[i]} acc')

    # Formatting Axes
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', color='black', fontsize=12)
    ax2.set_ylabel('Test Accuracy', color='black', fontsize=12)
    
    # Combine legends and place in the upper-right (now free!)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=8)

    plt.title("Comparison of 4 Runs: Training Loss vs Test Accuracy", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Make room at the bottom of the figure for the hyperparameter text box
    fig.subplots_adjust(bottom=0.18)

    # Hyperparams text box placed centrally at the bottom
    fig.text(0.5, 0.02, hp_text, 
            fontsize=9, 
            verticalalignment='bottom', 
            horizontalalignment='center', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(filename)
    print(f"Saved comparison results to: {filename}")
    # plt.show()
