# Self Driving Car Simulation | Reinforcement Learning using Twin Delayed DDPG

## Problem Statement : Create a self driving car simulation game using pygame and run T3D on top of it.

## Code Base Overview:


First I tried to create custom gym environment using pygame to train the model in colab but the exectuion time was very high thus created two files as follows :

* AI.py : Contains the TD3 Implementation including Actor-Critic models and Replay buffer
* Car_Train.py : Training the TDR for self driving car simulation
* Car_Inference : Inference code
* Train.ipynb : Training code to run on colab

Training was performed on colab and Inference was done using weights on local machine

## Steps followed to solve the problem :

__Step 1__: Create a basic car game using pygame. The car moves at a pre initialised speed, only the steering of the car is handled by using left and right arrow keys. Later this steering angle would be the value predicted by the model.

__Step 2__: Add the following functions in Car game :

* __Step()__ : This command will take an action at each step. The action is specified as its parameter. This function returns four parameters, namely observation, reward, done and info.
      * Observation :

* __reset()__ : This command will reset the environment. It returns an initial observation.

