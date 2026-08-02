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
import torch.nn as nn

#from torchvision import datasets, transforms


import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikegen
from snntorch import spikeplot as splt
from snntorch import utils






# Reproducibility + device
torch.manual_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

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
    batch_size = 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print (f"Batch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader

def forward_pass(net, spike_data):
    """Run the network over time on a pre-encoded spike train.

    spike_data: [num_steps, batch, 1, 28, 28] (from spikegen.rate)
    Returns stacked output spikes: [num_steps, batch, num_outputs]
    """
    spk_rec = []
    utils.reset(net)                       # clear membrane states of every Leaky
    for step in range(spike_data.size(0)):
        spk_out, _ = net(spike_data[step])
        spk_rec.append(spk_out)
    return torch.stack(spk_rec)


def batch_accuracy(loader, net, num_steps):
    """Accuracy over a whole DataLoader using a rate code (spike counts)."""
    net.eval()
    total, acc = 0, 0
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            spike_data = spikegen.rate(data, num_steps=num_steps)
            spk_rec = forward_pass(net, spike_data)
            acc   += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)
    return acc / total



train_loader, val_loader, test_loader = load_data()

# Network + simulation parameters
num_inputs  = 64 * 64
num_hidden  = 1000
num_outputs = 6

num_steps = 25          # timesteps of the rate-coded input (raise for Colab GPU)
beta      = 0.9         # snn.Leaky MEMBRANE decay (NOT the surrogate steepness!)
spike_grad = surrogate.fast_sigmoid(slope=25)   # slope = the lecture's beta

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_inputs, num_hidden),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Linear(num_hidden, num_outputs),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
    # TODO: fill in the layers


).to(device)

print(net)


# TODO: add cross-entropy on output spike counts (rate code)
#loss_fn   = nn.CrossEntropyLoss()
loss_fn   = SF.ce_rate_loss()        
optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, betas=(0.9, 0.999))
num_epochs = 1

loss_hist, test_acc_hist = [], []
counter = 0

for epoch in range(num_epochs):
    for data, targets in train_loader:
        #print("CHECK THIS SHAPE:", data.shape)
        data, targets = data.to(device), targets.to(device)

        # 1) TODO: rate-encode the images into a spike train with spikegen.rate
        spike_data = spikegen.rate(data, num_steps)

        # 2) TODO: run the forward pass over time (use forward_pass)
        net.train()
        #spk_rec = net(spike_data)
        spk_rec = forward_pass(net, spike_data)

        # 3) TODO: compute the loss on the output spikes with loss_fn
        loss_val = loss_fn(spk_rec, targets)

        # 4) backward pass (surrogate gradients kick in here) + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        loss_hist.append(loss_val.item())

        if counter % 50 == 0:
            test_acc = batch_accuracy(test_loader, net, num_steps)
            test_acc_hist.append(test_acc.item())
            print(f"Iteration {counter:4d} | loss {loss_val.item():.3f} "
                  f"| test acc {test_acc * 100:.2f}%")
        counter += 1

# Final test accuracy and a plot of the training loss
final_acc = batch_accuracy(test_loader, net, num_steps)
print(f"Final test set accuracy: {final_acc * 100:.2f}%")

plt.figure(figsize=(8, 4))
plt.plot(loss_hist)
plt.title("Training loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()