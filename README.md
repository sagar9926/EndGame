# Self Driving Car Simulation | Reinforcement Leearning using Twin Delayed DDPG

## Problem Statement : Create a self driving car simulation game using pygame and run T3D on top of it.

## Code Base Overview:


First I tried to create custom gym environment using pygame to train the model in colab but the exectuion time was very high thus created two files as follows :

* AI.py : Contains the TD3 Implementation including Actor-Critic models and Replay buffer
* Car_Train.py : Training the TDR for self driving car simulation
* Car_Inference : Inference code
* Train.ipynb : Training code to run on colab

Training was performed on colab and Inference was done using weights on local machine
