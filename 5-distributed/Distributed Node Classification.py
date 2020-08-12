#!/usr/bin/env python
# coding: utf-8

# # Distributed GraphSage for Node Classification
# 
# The tutorial shows distributed training on GraphSage for node classification. We reuse code from the mini-batch training examples. The model implementation and sampling for single-machine training and distributed training is exactly the same.

# In[ ]:


import dgl
import torch as th
import argparse
import numpy as np
from torch.utils.data import DataLoader


# ## Define hyperparameters

# To help us convert this Notebook to a training script easily, let's list all hyperparameters we want to tune. When we convert the notebook into a training script, we can specify the hyperparameters with arguments of the training script.
# 
# `standalone` controls whether to call Pytorch distributed training components.

# In[ ]:


ip_config = None
conf_path = 'standalone_data/ogbn-products.json'
num_epochs = 10
num_hidden = 128
num_layers = 2
batch_size = 1000
batch_size_eval = 100000
dropout = 0.5
lr = 0.001
standalone = True
num_workers = 0


# Define the arguments for the training script.
# 
# **Note**: `argparse` doesn't work in the Jupyter Notebook. When running in the Notebook environment, we should skip executing the code block.

# In[ ]:


parser = argparse.ArgumentParser(description='GCN')
parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
parser.add_argument('--conf_path', type=str, help='The path to the partition config file')
parser.add_argument('--num-epochs', type=int, default=10)
parser.add_argument('--num-hidden', type=int, default=128)
parser.add_argument('--num-layers', type=int, default=2)
parser.add_argument('--batch-size', type=int, default=1000)
parser.add_argument('--batch-size-eval', type=int, default=100000)
parser.add_argument('--standalone', action='store_true')
parser.add_argument('--local_rank', type=int, help='the rank for distributed training in Pytorch')
parser.add_argument('--num-workers', type=int, default=0, help='The number of worker processes for sampling.')
args = parser.parse_args()

ip_config = args.ip_config
conf_path = args.conf_path
num_epochs = args.num_epochs
num_hidden = args.num_hidden
num_layers = args.num_layers
batch_size = args.batch_size
batch_size_eval = args.batch_size_eval
standalone = args.standalone
num_workers = args.num_workers


# ## Initialize network communication
# 
# Before creating any components for distributed training, we need to initialize the network communication for both Pytorch and DGL.
# 
# Initialize RPC for network communication in DGL. When the process runs in the server mode, `init_rpc` will not return. Instead, it executes DGL servers.

# In[ ]:


dgl.distributed.initialize(ip_config, num_workers=num_workers)


# Initialize distributed training in Pytorch.

# In[ ]:


if not standalone:
    th.distributed.init_process_group(backend='gloo')


# ## Create DistGraph

# When creating a DistGraph object, it will load the input graph or connected to the servers that load the input graph, depending on its execution mode.
# 
# *Note*: the input graph has to be partitioned by the partition notebook first.

# In[ ]:


g = dgl.distributed.DistGraph('ogbn-products', part_config=conf_path)
print('#nodes:', g.number_of_nodes())
print('#edges:', g.number_of_edges())


# Get the nodes in the training, validation and testing set, which the current process is responsible for.

# In[ ]:


train_nid = dgl.distributed.node_split(g.ndata['train_mask'])
valid_nid = dgl.distributed.node_split(g.ndata['val_mask'])
test_nid = dgl.distributed.node_split(g.ndata['test_mask'])
print('train set:', len(train_nid))
print('valid set:', len(valid_nid))
print('test set:', len(test_nid))


# To define a model to classify nodes, we need to know how many unique labels there are in the dataset. The operation below actually fetch the labels of all nodes in the graph and run `unique` on the labels. This operation can be relatively expensive. If a user knows how many labels there are in the dataset, he/she can just pass the number of unique labels as an argument in the training script.

# In[ ]:


labels = g.ndata['labels'][0:g.number_of_nodes()]
uniq_labels = th.unique(labels)
num_labels = len(uniq_labels)
print('#labels:', num_labels)


# ## Define the model
# 
# The code of defining the GraphSage model is copied from the mini-batch training example.

# In[ ]:


import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        
    def forward(self, blocks, x):
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            x = layer(block, x)
            if l != self.n_layers - 1:
                x = F.relu(x)
        return x


# Create the model and use Adam as the optimizer.

# In[ ]:


import torch.optim as optim

model = SAGE(g.ndata['features'].shape[1], num_hidden, num_labels, num_layers)
loss_fcn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# To enable distributed training in Pytroch, we need to convert the model into a distributed model.

# In[ ]:


if not standalone:
    model = th.nn.parallel.DistributedDataParallel(model)


# # Sampling

# The same sampling code for a single-process training also works for distributed training.

# In[ ]:


sampler = dgl.dataloading.MultiLayerNeighborSampler([25,10])
train_dataloader = dgl.dataloading.NodeDataLoader(
    g, train_nid, sampler,
    batch_size=1024,
    shuffle=True,
    drop_last=False
)
valid_dataloader = dgl.dataloading.NodeDataLoader(
    g, valid_nid, sampler,
    batch_size=1024,
    shuffle=False,
    drop_last=False
)


# # Training loop

# The training loop is also the same as the mini-batch training in a single machine.
# 
# We recommend users to compute the validation score in a mini-batch fashion with neighbor sampling. This is the most cost-effective way of computing validation scores in the distributed training. Although the score could be a little lower than the actual one, it should be sufficient for us to select the right model.

# In[ ]:


import time
import sklearn.metrics

start = time.time()
for epoch in range(num_epochs):
    # Loop over the dataloader to sample the computation dependency graph as a list of blocks.
    start = time.time()
    losses = []
    for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
        # Load the input features as well as output labels
        batch_inputs = g.ndata['features'][input_nodes]
        batch_labels = g.ndata['labels'][seeds]

        # Compute loss and prediction
        batch_pred = model(blocks, batch_inputs)
        loss = loss_fcn(batch_pred, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.detach().cpu().numpy())

        # Aggregate gradients in multiple nodes.
        if not standalone:
            for param in model.parameters():
                if param.requires_grad and param.grad is not None:
                    th.distributed.all_reduce(param.grad.data,
                                              op=th.distributed.ReduceOp.SUM)
                    param.grad.data /= dgl.distributed.get_num_client()

        optimizer.step()
        break
    print('Epoch {}: training takes {:.3f} seconds, loss={:.3f}'.format(epoch, time.time() - start, np.mean(losses)))

    # validation
    predictions = []
    labels = []
    start = time.time()
    with th.no_grad():
        for step, (input_nodes, seeds, blocks) in enumerate(valid_dataloader):
            inputs = g.ndata['features'][input_nodes]
            labels.append(g.ndata['labels'][seeds].numpy())
            predictions.append(model(blocks, inputs).argmax(1).numpy())
            break
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        accuracy = sklearn.metrics.accuracy_score(labels, predictions)
        print('Epoch {}: validation takes {:.3f} seconds, Validation Accuracy {}'.format(epoch, time.time() - start, accuracy))


# # Inference
# 
# For offline inference, there are two ways:
# * We can compute the classification accuracy of nodes in the test set with a mini-batch fashion. In this case, we should still use neighbor sampling to reduce the computation overhead. This is cost effective if the test set is small. However, in practice, the nodes where we need to compute the scores are usually much more than the labeled nodes in the training set. In this case, the mini-batch inference is not recommended.
# * We can perform the inference on the full graph. In this case, we compute the node embeddings of all nodes in the graph. To perform full graph inference efficiently, we compute the intermediate node embeddings on all nodes layer by layer. In the end, we will compute the final embeddings of all nodes in the graph. After having the final node embeddings, we compute the accuracy on nodes in the test set.
# 
# The code below shows how the full graph inference is implemented in a distributed fashion.
# 
# First, we split nodes that need to compute embeddings. Since this is full graph inference, all nodes need to compute embeddings, so we generate a boolean array of the size equal to the number of nodes in the graph and all elements are True. `node_split` returns the nodes that the local process is responsible for.

# In[ ]:


nodes = dgl.distributed.node_split(np.ones(g.number_of_nodes(), dtype=bool), g.get_partition_book())


# Because we compute the node embeddings in a layer-by-layer fashion, we need a sampler that samples one-hop neighborhood. We can use a relatively large batch size to increase computation efficiency.

# In[ ]:


# The sampler generates a mini-batch for 1-layer GraphSage. Thus, we can use very large batch size.
sampler = dgl.dataloading.MultiLayerNeighborSampler([None])
test_dataloader = dgl.dataloading.NodeDataLoader(
    g, nodes, sampler,
    batch_size=10000,
    shuffle=False,
    drop_last=False
)


# The distributed Pytorch model has slightly different interface. We can access the original model object from its `module`.

# In[ ]:


layers = model.layers if standalone else model.module.layers


# Here is the code to compute node embeddings one layer at time. It first computes the intermediate node embeddings of all nodes before moving to the next layer. The intermediate embeddings are stored in `DistTensor`.
# 
# In distributed inference, we have to put a barrier between every layer because different processes may perform computation at a different rate. After computing the node embeddings of one layer, all processes need to synchronize to ensure that the embeddings of all nodes are ready before moving to the next layer. Otherwise, some process that run faster may end up reading embeddings that haven't been computed yet.

# In[ ]:

class NeighborSampler(object):
    def __init__(self, g, fanouts, sample_neighbors):
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors

    def sample_blocks(self, seeds):
        seeds = th.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = self.sample_neighbors(self.g, seeds, fanout, replace=True)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
        return blocks

'''
nodes = dgl.distributed.node_split(np.arange(g.number_of_nodes()), g.get_partition_book())
x = g.ndata['features']
y = dgl.distributed.DistTensor((g.number_of_nodes(), num_hidden), th.float32, 'h', persistent=True)
for l, layer in enumerate(layers):
    if l == len(layers) - 1:
        y = dgl.distributed.DistTensor((g.number_of_nodes(), num_labels),
                                       th.float32, 'h_last', persistent=True)

    sampler = NeighborSampler(g, [-1], dgl.distributed.sample_neighbors)
    print('|V|={}, eval batch size: {}'.format(g.number_of_nodes(), batch_size))
    # Create PyTorch DataLoader for constructing blocks
    dataloader = DataLoader(
                dataset=nodes,
                batch_size=batch_size,
                collate_fn=sampler.sample_blocks,
                shuffle=False,
                drop_last=False)

    for blocks in dataloader:
        block = blocks[0]
        input_nodes = block.srcdata[dgl.NID]
        output_nodes = block.dstdata[dgl.NID]
        h = x[input_nodes]
        h_dst = h[:block.number_of_dst_nodes()]
        h = layer(block, (h, h_dst))
        if l != len(layers) - 1:
            h = F.relu(h)
        y[output_nodes] = h.cpu()

    x = y
    g.barrier()
print('inference complete')
'''


start = time.time()
x = g.ndata['features']
# We create a distributed tensor to store the intermediate node embeddings.
y = dgl.distributed.DistTensor((g.number_of_nodes(), num_hidden), th.float32, name='h1')
layer = layers[0]
print('test1')
batch = 0
for input_nodes, seeds, blocks in test_dataloader:
    print('test2:', batch)
    block = blocks[0]
    h = x[input_nodes]
    with th.no_grad():
        h = layer(block, h)
        h = F.relu(h)
        y[seeds] = h
    print('test3:', batch)
    batch += 1
print('test4')

x = y
g.barrier()
layer = layers[1]
# We create another tensor to store the final node embeddings.
y = dgl.distributed.DistTensor((g.number_of_nodes(), num_labels), th.float32, name='h2', persistent=True)
print('test5')
for input_nodes, seeds, blocks in test_dataloader:
    block = blocks[0]
    h = x[input_nodes]
    with th.no_grad():
        y[seeds] = layer(block, h)
end = time.time()


# Once we have the node embeddings of all nodes, we can predict the labels of the nodes in the test set.

# In[ ]:


# Compute the accuracy of nodes in the test set.
predictions = y[test_nid].argmax(1).numpy()
labels = g.ndata['labels'][test_nid]
accuracy = sklearn.metrics.accuracy_score(labels, predictions)
print('Test takes {:.3f} seconds, acc={:.3f}'.format(end - start, accuracy))

