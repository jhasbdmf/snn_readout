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


from utils import load_data
from itertools import islice

import os
from datetime import datetime



# Reproducibility + device
torch.manual_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)

def forward_pass(net, spike_data):
    """Run the network over time on a pre-encoded spike train.

    spike_data: [num_steps, batch, 1, 28, 28] (from spikegen.rate)
    Returns stacked output spikes: [num_steps, batch, num_outputs]
    """
    spk_rec = []
    mem_rec = []
    utils.reset(net)                       # clear membrane states of every Leaky
    for step in range(spike_data.size(0)):
        spk_out, mem_out = net(spike_data[step])
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)
    #return torch.stack(spk_rec)


def batch_accuracy(loader, net, num_steps):
    """Accuracy over a whole DataLoader using a rate code (spike counts)."""
    net.eval()
    total, correct_spike_rate, correct_max_mem, correct_mean_mem, correct_last_mem  = 0, 0, 0, 0, 0
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            spike_data = spikegen.rate(data, num_steps=num_steps)
            spk_rec, mem_rec = forward_pass(net, spike_data)
            correct_spike_rate   += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

            #max membrane potential decoding
            predictions_max_mem = torch.argmax(torch.max(mem_rec, dim=0).values, dim=1)
            #mean membrane potential decoding
            predictions_mean_mem = torch.argmax(torch.mean(mem_rec, dim=0), dim=1)

            #last time step membrane potential decoding
            predictions_last_mem = torch.argmax(mem_rec[-1], dim=1)

            correct_max_mem += (predictions_max_mem == targets).sum().item()
            correct_mean_mem += (predictions_mean_mem == targets).sum().item()
            correct_last_mem += (predictions_last_mem == targets).sum().item()

            #print (predictions_max_mem.shape)

            #print (mem_rec.shape)

            acc_spike_rate = correct_spike_rate / total
            acc_max_mem = correct_max_mem / total
            acc_mean_mem = correct_mean_mem / total
            acc_last_mem = correct_last_mem / total


    return acc_spike_rate, acc_max_mem, acc_mean_mem, acc_last_mem

def train_snn (net,
               num_epochs,
               train_loader,
               val_loader,
               test_loader,
               decoding_method="spike_rate", 
               lr=1e-5,
               betas=(0.9, 0.999)):
        
    # TODO: add cross-entropy on output spike counts (rate code)
    #loss_fn   = nn.CrossEntropyLoss()
    #loss_fn_rate_encoding   = SF.ce_rate_loss()
    #loss_cross_entropy = nn.CrossEntropyLoss()        
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=betas)
   

    loss_hist, test_acc_hist = [], []
    counter = 0

    for epoch in range(num_epochs):
        #for data, targets in train_loader:
        for data, targets in islice(train_loader, 128):
            
            #print("CHECK THIS SHAPE:", data.shape)
            data, targets = data.to(device), targets.to(device)

            # 1) rate-encode the images into a spike train with spikegen.rate
            spike_data = spikegen.rate(data, num_steps)

            # 2) run the forward pass over time 
            net.train()
            #spk_rec = net(spike_data)
            spk_rec, mem_rec = forward_pass(net, spike_data)

            # 3) compute the loss on the output spikes with loss_fn
            if decoding_method=="spike_rate":
                loss = SF.ce_rate_loss()(spk_rec, targets)
            elif decoding_method=="max_membrane_potential":
                logits = torch.max(mem_rec, dim=0).values
                loss = nn.CrossEntropyLoss()(logits, targets)
            elif decoding_method=="mean_membrane_potential":
                logits = torch.mean(mem_rec, dim=0)
                loss = nn.CrossEntropyLoss()(logits, targets)
            elif decoding_method=="last_membrane_potential":
                logits = mem_rec[-1]
                loss = nn.CrossEntropyLoss()(logits, targets)

            # 4) backward pass (surrogate gradients kick in here) + weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_hist.append(loss.item())

            if counter % 50 == 0:
                test_acc_spike_rate, test_acc_max_mem, test_acc_mean_mem, test_acc_last_mem = batch_accuracy(test_loader, net, num_steps)
                test_acc_hist.append(test_acc_spike_rate.item())
                print(f"Iteration {counter:4d} | loss {loss.item():.3f} "
                    f"| test acc spk rate {test_acc_spike_rate * 100:.2f}% "
                    f"| test acc max mem {test_acc_max_mem * 100:.2f}%"
                    f"| test acc mean mem {test_acc_mean_mem * 100:.2f}%"
                    f"| test acc last mem {test_acc_last_mem * 100:.2f}%"
                )
            counter += 1

    # Final test accuracy and a plot of the training loss
    final_acc, _, _ , _ = batch_accuracy(test_loader, net, num_steps)



    print(f"Final test set accuracy: {final_acc * 100:.2f}%")

    # 1. Create the "figures" directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)

    # 2. Create the timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"figures/loss_curve_{timestamp}.png"

    hp_text = (
        f"Input dimension: {num_inputs}\n"
        f"Hidden: {num_hidden}\n"
        f"Outputs: {num_outputs}\n"
        f"Steps: {num_steps}\n"
        f"snn.Leaky MEMBRANE decay beta: {beta}\n"
        f"Slope: 25\n"
        f"LR: 1e-5\n"
        f"Adam betas: (0.9, 0.999)"
    )

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(loss_hist, color='blue', linewidth=1.5)

    plt.title("Training Loss", fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add hyperparameters as a text box in the upper right corner
    plt.text(0.95, 0.95, hp_text, 
            transform=plt.gca().transAxes, 
            fontsize=10, 
            verticalalignment='top', 
            horizontalalignment='right', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Save the figure
    plt.savefig(filename)
    print(f"Figure saved to: {filename}")

    # Show the plot
    plt.show()


    #plt.figure(figsize=(8, 4))
    #plt.plot(loss_hist)
    #plt.title("Training loss")
    #plt.xlabel("Iteration")
    #plt.ylabel("Loss")
    #plt.show()



train_loader, val_loader, test_loader = load_data()

# Network + simulation parameters
num_inputs  = 64 * 64
num_hidden  = 512
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

train_snn(net=net,
          num_epochs=1,
          train_loader=train_loader,
          val_loader=val_loader,
          test_loader=test_loader,
          decoding_method="last_membrane_potential"
        )

