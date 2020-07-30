This section extends the mini-batch training shown in the previous section to distributed setting. It contains the parts below:

**Part 1**: introduction of the basic concept of distributed training and the components provided by DGL
([slides](https://github.com/dglai/KDD20-Hands-on-Tutorial/raw/master/5-distributed/distributed%20training%20tutorial.pptx)).
This introduction assumes the readers have read the previous section on mini-batch training.

**Part 2**: Jupyter Notebook to partition the OGB product graph
([notebook](https://github.com/dglai/KDD20-Hands-on-Tutorial/blob/master/5-distributed/partition.ipynb)).

**Part 3**: Jupyter Notebook to demonstrate the basic operations on distributed components of DGL
([notebook](https://github.com/dglai/KDD20-Hands-on-Tutorial/blob/master/5-distributed/basic.ipynb))

**Part 4**: Jupyter Notebook for distributed GraphSage for node classification
([notebook](https://github.com/dglai/KDD20-Hands-on-Tutorial/blob/master/5-distributed/Distributed%20Node%20Classification.ipynb))

**Part 5**: Jupyter Notebook for distributed GraphSage with node embeddings for node classification
([notebook](https://github.com/dglai/KDD20-Hands-on-Tutorial/blob/master/5-distributed/Distributed%20Node%20Classification-emb.ipynb))

**Part 6**: Convert the Notebooks to training scripts and launch the training script on a cluster

Step 1: Convert the Notebooks into training scripts:
```bash
jupyter nbconvert --to script Distributed\ Node\ Classification.ipynb
jupyter nbconvert --to script Distributed\ Node\ Classification-emb.ipynb
```

Step 2: Set up ssh passwordless access between machines in the cluster.

Step 3: Download the launch script:
```bash
wget https://raw.githubusercontent.com/dmlc/dgl/master/tools/launch.py
```

Step 4: Train the model:
Standalone mode:
```bash
python3 -m torch.distributed.launch  Distributed\ Node\ Classification.py --ip_config ip_config.txt --num-epochs 10 --batch-size 5000 --num-hidden 512 --conf_path standalone_data/ogbn-products.json
```

Distributed mode:
```bash
python3 launch.py --workspace ~/workspace/KDD20-Hands-on-Tutorial/5-distributed --num_client 4 --conf_path 4part_data/ogbn-products.json --ip_config ip_config.txt "python3 Distributed\ Node\ Classification.py --ip_config ip_config.txt --num-epochs 10 --batch-size 5000 --num-hidden 512 --conf_path standalone_data/ogbn-products.json"

python3 launch.py --workspace ~/workspace/KDD20-Hands-on-Tutorial/5-distributed --num_client 4 --conf_path 4part_data/ogbn-products.json --ip_config ip_config.txt "python3 Distributed\ Node\ Classification-emb.py --ip_config ip_config.txt --num-epochs 1 --batch-size 5000 --num-hidden 512 --conf_path standalone_data/ogbn-products.json"
```

