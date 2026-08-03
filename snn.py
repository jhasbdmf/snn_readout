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


from utils import load_data, plot_stats
from itertools import islice

# Reproducibility + device
torch.manual_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", device)




def forward_pass1(net, spike_data):
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


def forward_pass(net, spike_data):
    utils.reset(net)
    
    final_spk_rec = []
    # Dictionaries to hold lists over time
    # Key: layer_index, Value: list of tensors over time
    inter_spk_rec = {i: [] for i in range(len(net.layers) // 2)} 
    inter_mem_rec = {i: [] for i in range(len(net.layers) // 2)}

    for step in range(spike_data.size(0)):
        # Unpack the 3 return values
        spk_out, all_spikes, all_mems = net(spike_data[step])
        
        final_spk_rec.append(spk_out)
        
        # Store intermediate spikes and membranes for this time step
        for i in range(len(all_spikes)):
            inter_spk_rec[i].append(all_spikes[i])
            inter_mem_rec[i].append(all_mems[i])

    # Stack into tensors: [num_steps, batch, dim]
    final_spikes = torch.stack(final_spk_rec)
    
    for i in inter_spk_rec:
        inter_spk_rec[i] = torch.stack(inter_spk_rec[i])
        inter_mem_rec[i] = torch.stack(inter_mem_rec[i])

    last_layer_idx = len(inter_mem_rec) - 1

    
    #print ("a", inter_spk_rec[0].shape)

    mean_firing_rate_per_layer = []
    for i in inter_spk_rec:
        mean_firing_rate_per_layer.append(inter_spk_rec[i].float().mean().item())

    #print (mean_firing_rate_per_layer)

    #return final_spikes, inter_spk_rec, inter_mem_rec[last_layer_idx]
    return final_spikes, mean_firing_rate_per_layer, inter_mem_rec[last_layer_idx]


def batch_accuracy(loader, net, num_steps):
    """Accuracy over a whole DataLoader using a rate code (spike counts)."""
    net.eval()
    total, correct_spike_rate, correct_max_mem, correct_mean_mem, correct_last_mem  = 0, 0, 0, 0, 0
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            spike_data = spikegen.rate(data, num_steps=num_steps)
            #spk_rec, mem_rec = forward_pass(net, spike_data)

            #print ("************")
            spk_rec, _, mem_rec = forward_pass(net, spike_data)

            #print ("lkasjdflkjasdf", mem_rec.shape)
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
               num_steps,
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
   

    train_loss_hist, test_acc_hist = [], []
    counter = 0
    #all_mem_rec = torch.empty(0,num_steps)
    #print (all_mem_rec.shape)
    #all_mem_rec_list = []
    mean_spike_rate_per_layer_hist = []

    for epoch in range(num_epochs):
        #for data, targets in train_loader:
        for data, targets in islice(train_loader, 3):
            
            #print("CHECK THIS SHAPE:", data.shape)
            data, targets = data.to(device), targets.to(device)

            # 1) rate-encode the images into a spike train with spikegen.rate
            spike_data = spikegen.rate(data, num_steps)

            # 2) run the forward pass over time 
            net.train()
            #spk_rec = net(spike_data)
            #spk_rec, mem_rec = forward_pass(net, spike_data)
            #print ("____")
            spk_rec, mean_spike_rate , mem_rec = forward_pass(net, spike_data)
            mean_spike_rate_per_layer_hist.append(mean_spike_rate)
            #all_mem_rec_list.append(mem_rec)

            #print ("asdasd", type(mem_rec))
            #print ("asdasdasd", mem_rec.shape)

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

            train_loss_hist.append(loss.item())

            test_acc_spike_rate, test_acc_max_mem, test_acc_mean_mem, test_acc_last_mem = batch_accuracy(test_loader, net, num_steps)

            if decoding_method=="spike_rate":
                test_acc_hist.append(test_acc_spike_rate)
            elif decoding_method=="max_membrane_potential":
                test_acc_hist.append(test_acc_max_mem)
            elif decoding_method=="mean_membrane_potential":
                test_acc_hist.append(test_acc_mean_mem)
            elif decoding_method=="last_membrane_potential":
                test_acc_hist.append(test_acc_last_mem)

            #if counter % 50 == 0:
                #test_acc_spike_rate, test_acc_max_mem, test_acc_mean_mem, test_acc_last_mem = batch_accuracy(test_loader, net, num_steps)
                #test_acc_hist.append(test_acc_spike_rate.item())
            print(f"Batch {counter:4d} | loss {loss.item():.3f} "
                f"| test acc spk rate {test_acc_spike_rate * 100:.2f}% "
                f"| test acc max mem {test_acc_max_mem * 100:.2f}%"
                f"| test acc mean mem {test_acc_mean_mem * 100:.2f}%"
                f"| test acc last mem {test_acc_last_mem * 100:.2f}%"
            )
            counter += 1

    # Final test accuracy and a plot of the training loss
    final_acc, _, _ , _ = batch_accuracy(test_loader, net, num_steps)

    print(f"Final test set accuracy: {final_acc * 100:.2f}%")

    #all_mem_rec_tensor = torch.stack(all_mem_rec_list)
    #print ("kjashkjdflsadf", all_mem_rec_tensor.shape)

    return train_loss_hist, test_acc_hist, mean_spike_rate_per_layer_hist

class LIF_SNN(nn.Module):
    def __init__(self, input_dim=64*64, hidden_dim=128, n_hidden=1, spike_grad=surrogate.fast_sigmoid(slope=25), beta=0.9, out_dim=6):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.out_dim = out_dim
        self.beta = beta

        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()


        if self.n_hidden == 0:
            self.layers.append(nn.Linear(self.input_dim, self.out_dim))
            self.layers.append (snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True))

        else:
            self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            self.layers.append (snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True))
            for i in range(n_hidden-1):
                self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                self.layers.append(snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True))
                
            self.layers.append(nn.Linear(self.hidden_dim, self.out_dim))
            self.layers.append (snn.Leaky(beta=self.beta, spike_grad=spike_grad, init_hidden=True))

    def forward(self, x):

        all_spikes = []
        all_mem_pot = []
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
            
            if isinstance(layer, snn.Leaky):
                all_mem_pot.append(layer.mem)
                all_spikes.append(x)
        
        return x, all_spikes, all_mem_pot


    
        

train_loader, val_loader, test_loader = load_data()

# Network + simulation parameters
num_inputs  = 64 * 64
num_hidden  = 64
n_hidden = 1
num_outputs = 6

num_steps = 25          # timesteps of the rate-coded input (raise for Colab GPU)
beta      = 0.9
slope = 25         # snn.Leaky MEMBRANE decay (NOT the surrogate steepness!)
spike_grad = surrogate.fast_sigmoid(slope=slope)   # slope = the lecture's beta

lr=1e-5
betas = (0.9, 0.999)

readouts = ["spike_rate", "max_membrane_potential", "mean_membrane_potential", "last_membrane_potential"]


train_loss_hists = []
test_acc_hists = []




baseline_net = LIF_SNN(hidden_dim=num_hidden, n_hidden=n_hidden)
initial_state = baseline_net.state_dict()


for readout in readouts:

    """
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_inputs, num_hidden),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(num_hidden, num_outputs),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
    ).to(device)
    """
    #print(net)

    net = LIF_SNN(hidden_dim=num_hidden, n_hidden=n_hidden).to(device)
    
    # 3. Load the exact same initial weights
    net.load_state_dict(initial_state)
    

    train_loss_hist, test_acc_hist, _ = train_snn(
            net=net,
            num_steps=num_steps,
            num_epochs=1,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            decoding_method=readout,
            lr=lr,
            betas=betas
            )

    train_loss_hists.append(train_loss_hist)
    test_acc_hists.append(test_acc_hist)

    


plot_stats(train_loss_hists=train_loss_hists,
            test_acc_hists=test_acc_hists,
            #run_names=run_names,
            run_names=readouts,
            num_inputs=num_inputs,
            num_hidden=num_hidden,
            num_steps=num_steps,
            beta=beta,
            slope=slope,
            lr=lr,
            betas=betas,
            )


