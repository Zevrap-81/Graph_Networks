# Graph_Networks

This is a Thesis Project *Development of Graph Neural Network architecture for quality prediction of Deep-drawing processes*. 

The aim is to learn mesh-based data from deep-drawing simulations using data-driven graph neural network models. These models are given physics information (in the form of contact/collision) from simulation along with process parameters. Please refer to my [report](https://drive.google.com/file/d/1SENZZaMOzfNtqS37IhtRGa6_48r_atRY/view) for more information.

### Repository structure
```
└── examples
│   │   train_mesh_graph_net.py
│   │   ...
└── graph_networks
|   │   common.py
|   │   parameters.py
│   │   ...
│   └── datageneration
│   |   │   ...
│   └── models
│       │   mesh_graph_net.py
│       │   mesh_net.py
│       │   simple_ann.py
│       │   ...
└── saved_data
    └── MM_DD_hh_mm
        └── checkpoints
        └── logs
        └── visualizations
```

## Installation
Clone this repo
```
$ git clone https://github.com/Zevrap-81/Graph_Networks.git
$ cd Graph_networks
```
Create a virtual environment and activate it (activation command is different for linux)
```
$ python -m venv env
$ .\env\Scripts\activate
```
Install dependencies
```
$ pip install -r requirements.txt
```
Aditionally install my trainer module 
```
$ git clone https://github.com/Zevrap-81/Trainer.git
$ cd Trainer
$ pip install -e .
$ cd ..
```
Finally install graph_networks 
```
$ pip install -e .
```
