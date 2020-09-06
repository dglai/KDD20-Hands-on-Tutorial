# Scalable Graph Neural Networks with Deep Graph Library

Learning from graph and relational data plays a major role in many applications including social network analysis, marketing, e-commerce, information retrieval, knowledge modeling, medical and biological sciences, engineering, and others. In the last few years, Graph Neural Networks (GNNs) have emerged as a promising new supervised learning framework capable of bringing the power of deep representation learning to graph and relational data. This ever-growing body of research has shown that GNNs achieve state-of-the-art performance for problems such as link prediction, fraud detection, target-ligand binding activity prediction, knowledge-graph completion, and product recommendations. In practice, many of the real-world graphs are very large. It is urgent to have scalable solutions to train GNN on large graphs efficiently.

The objective of this tutorial is twofold. First, it will provide an overview of the theory behind GNNs, discuss the types of problems that GNNs are well suited for, and introduce some of the most widely used GNN model architectures and problems/applications that are designed to solve. Second, it will introduce the Deep Graph Library (DGL), a scalable GNN framework that simplifies the development of efficient GNN-based training and inference programs at a large scale. To make things concrete, the tutorial will cover state-of-the-art training methods to scale GNN to large graphs and provide hands-on sessions to show how to use DGL to perform scalable training in different settings (multi-GPU training and distributed training). This hands-on part will start with basic graph applications (e.g., node classification and link prediction) to set up the context and move on to train GNNs on large graphs. It will provide tutorials to demonstrate how to apply the techniques in DGL to train GNNs for real-world applications.

Presenters: Da Zheng, Minjie Wang, Quan Gan, Zheng Zhang, George Karypis

Agenda
---

| Time | Session | Material | Presenter |
|:----:|:-------:|:--------:|:---------:|
| 9:00-10:00 | Overview of Graph Neural Networks | [slides](https://github.com/dglai/KDD20-Hands-on-Tutorial/blob/master/GNN_overview.pptx) | George Karypis |
| 10:00-10:30 | Overview of Deep Graph Library (DGL) | [slides](https://github.com/zheng-da/dgl-tutorial-full/blob/master/dgl_api/dgl-www-zz.pptx) | Zheng Zhang |
| 10:30-11:00 | Virtual Coffee Break | | |
| 11:00-12:00 | (Hands-on) GNN models for basic graph tasks | [notebook](https://github.com/dglai/KDD20-Hands-on-Tutorial/tree/master/3-basics) | Minjie Wang |
| 12:00-13:00 | Virtual Lunch Break | | |
| 13:00-14:30 | (Hands-on) GNN mini-batch training | [notebook](https://github.com/dglai/KDD20-Hands-on-Tutorial/tree/master/4-large%20graph) | Quan Gan |
| 14:30-16:00 | (Hands-on) GNN distributed training | [notebook](https://github.com/dglai/KDD20-Hands-on-Tutorial/tree/master/5-distributed) | Da Zheng |

Section Content
---

* **Section 1: Overview of Graph Neural Networks.** This section describes how graph
  neural networks operate, their underlying theory, and their advantages over alternative
  graph learning approaches. In addition, it describes various learning problems on graphs
  and shows how GNNs can be used to solve them.
* **Section 2: Overview of Deep Graph Library (DGL).** This section describes the different
  abstractions and APIs that DGL provides, which are designed to simplify the implementation
  of GNN models, and explains how DGL interfaces with MXNet, Pytorch, and TensorFlow.
  It then proceeds to introduce DGL’s message-passing API that can be used to develop
  arbitrarily complex GNNs and the pre-defined GNN nn modules that it provides.
* **Section 3: GNN models for basic graph tasks.** This section demonstrates how to use
  GNNs to solve four key graph learning tasks: node classification, link prediction, graph
  classification, and network embedding pre-training. It will show how GraphSage, a popular
  GNN model, can be implemented with DGL’s nn module and show how the node embeddings
  computed by GraphSage can be used in different types of downstream tasks. In addition,
  it will demonstrate the implementation of a customized GNN model with DGL’s message passing
  interface.
* **Section 4: GNN training on large graphs.** This section uses some of the models described
  in Section 3 to demonstrate mini-batch training, multi-GPU training, and distributed
  training in DGL. It starts by describing how the concept of mini-batch training applies to
  GNNs and how mini-batch computations can be sped up by using various sampling techniques.
  It then proceeds to illustrate how one such sampling technique, called neighbor sampling,
  can be implemented in DGL using a Jupyter notebook. This notebook is then extended to show
  multi-GPU training and distributed training.
* **Section 5: Distributed GNN training.** This section extends the mini-batch training
  in Section 4 to the distributed setting. It describes the concept of distributed GNN training
  and the facilities provided by DGL for distributed training. It illustrates the concept with
  code in the Jupyter Notebook and demo.

## Community

Join our [Slack channel "KDD20-tutorial"](https://join.slack.com/t/deep-graph-library/shared_invite/zt-eb4ict1g-xcg3PhZAFAB8p6dtKuP6xQ) for discussion.
