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

* __Step()__ : This command will take an action at each step. The action is specified as its parameter. This function returns four parameters, namely state, reward, done.

  * __State__ : An environment-specific object representing your observation of the environment.This comprises of three variables:
  
         * 28x28 pixels around the car
         
         * Distance of car from the destination 
         
         * Orientation (the angle between velocity vector and displacement vector in reference to position of the car) 
  
  * __Reward__ : Amount of reward achieved by the previous action.
  
  Action                                 | Reward Amount
  -------------------------------------- | -------------
  Moving on road(Living Penalty)         | - 1.5
  Moving on road and towards Destination | + 0.75
  Moving on sand                         | -5 
  Moving near boundary                   | -10
  
  * __done__ : A boolean value stating whether it’s time to reset the environment again.

* __reset()__ : This command will reset the environment. It returns an initial observation.

__Step 3__ : Buid the Actor-Critic models : Model takes input a 28x28 image patch , orientation and distance from destination. First part of model is __Encoder__. This extracts the features from the input 28x28 image patch and encodes them into latent vector of size 16. In second part of model The other two state parameters are concatenated with this latent vector and the passed through a series of fully conected layers.

### Single convolution Block :
![image](ConvolutionBlock.png)

### Network Architecture :


