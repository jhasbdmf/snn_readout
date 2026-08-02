#Choosing the Readout: How Output Decoding Shapes a Spiking Network

#Keywords: feedforward networks, surrogate gradient

#When a spiking network is trained end-to-end with surrogate gradients, the classifier's decision has to be 
#extracted from a stream of discrete spikes. There are several common ways to do this: attach a spiking 
#output layer and count its spikes, or read out the membrane potential of non-spiking output units via their 
#maximum over time, their temporal mean, or their value at the last timestep. Each choice defines a 
#different loss surface and routes gradients back through the network differently, even when the underlying 
#architecture and task are identical.

#In this project, investigate how the choice of readout affects a spiking network trained with surrogate 
#gradient descent. Compare a spiking readout against membrane-potential readouts using max-over-time, 
#mean-over-time, and last-timestep decoding on a simple classification or temporal prediction task. 
#Quantify effects on task performance, mean firing activity in the hidden layers, and the scale and stability 
#of gradients flowing back to earlier layers, and relate the differences to how each readout aggregates 
#information over time.

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

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
    plt.show()

    # 2. Convert to PyTorch Tensors AND normalize pixels to [0.0, 1.0]
    # Labels must be torch.long for CrossEntropyLoss
    train_x_t = torch.tensor(train_x, dtype=torch.float32) / 255.0
    train_y_t = torch.tensor(train_y, dtype=torch.long)

    val_x_t   = torch.tensor(val_x, dtype=torch.float32) / 255.0
    val_y_t   = torch.tensor(val_y, dtype=torch.long)

    test_x_t  = torch.tensor(test_x, dtype=torch.float32) / 255.0
    test_y_t  = torch.tensor(test_y, dtype=torch.long)

    # 3. Create Datasets
    train_dataset = TensorDataset(train_x_t, train_y_t)
    val_dataset   = TensorDataset(val_x_t, val_y_t)
    test_dataset  = TensorDataset(test_x_t, test_y_t)

    # 4. Create DataLoaders
    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print (f"Batch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    return train_dataset, val_loader, test_loader

load_data()
