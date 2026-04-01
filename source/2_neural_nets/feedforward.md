## Feedforward Networks


Feedforward networks are the simplest type of neural network. They are called "feedforward" because the input flows through the network in a single direction, from the input layer to the output layer.

### Main idea

A feedforward network is a type of neural network that is composed of a sequence of layers. Each layer is a linear transformation of the input, followed by a non-linear activation function.

### Architecture

A feedforward network is composed of a sequence of layers. Each layer is a linear transformation of the input, followed by a non-linear activation function.

The input layer is the first layer, and the output layer is the last layer.

```{figure} ../_static/imgs/neural_nets/feedforward1.PNG
:width: 80%
:alt: Feedforward network architecture

Feedforward network architecture.
```


#### Feedforward network vs. general networks

```{figure} ../_static/imgs/neural_nets/feedforward2.PNG
:width: 80%
:alt: Feedforward network architecture vs. general networks

Feedforward network architecture vs. general networks.
```

Feedforward network: inputs propagate strictly layer-to-layer, no cycles or lateral connections.

General neural network: may include lateral (within-layer) and feedback (between-layer) connections, creating cycles in the computation graph.

*In a feedforward network, the graph is acyclic―data always moves forward from input through hidden layers to output. In contrast, general networks can have arbitrary connections, such as lateral (connections between neurons in the same layer) and feedback loops (connections from later to earlier layers), allowing for more complex behaviors but introducing cycles.*


