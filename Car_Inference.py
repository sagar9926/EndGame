# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 07:05:10 2020

@author: Admin
"""


#OldLaptop
# Importing the libraries
import numpy as np
import os
import time
import torch
import math 
import matplotlib.pyplot as plt

import os
import pygame
from math import sin, radians, degrees, copysign
from pygame.math import Vector2


from PIL import Image as PILImage
from AIUpdatedV2 import ReplayBuffer, TD3
from scipy.ndimage import rotate


seed = 0  # Random seed number
# Set seed for consistency
torch.manual_seed(seed)
np.random.seed(seed)
save_models = True  # Boolean checker whether or not to save the pre-trained model

file_name = "%s_%s_%s" % ("TD3", "CarApp", str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")
if not os.path.exists("./results"):
  os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"):
  os.makedirs("./pytorch_models")


prev_reward = 0
origin_x = 743
origin_y = 380
coordinates = [[590,380],[730,90],[1130,405],[120,300],[1110,630]]
first_update = True # Setting the first update
last_distance = 0   # Initializing the last distance

def init():
    global sand
    global dest_x
    global dest_y
    global origin_x
    global origin_y
    global first_update
    sand = np.zeros((longueur, largeur))
    img = PILImage.open("mask.png").convert('L')
    sand = np.asarray(img) / 255
    dest_x, dest_y = coordinates[np.random.randint(0, 5)]
    first_update = False


class Car:
    cropsize = 28
    padsize = 28
    rotation = 0
    state_img_patch = np.zeros([1,int(cropsize),int(cropsize)])
    
    def __init__(self, x=origin_x, y=origin_y, angle=0.0):
        # Positon vector for Car initialised with the initial location of car on map
        self.position = Vector2(743,358) 
        
        # Velocity Vector of car
        self.velocity = Vector2(0.0, 0.0)
        
        # Steering angle 
        self.angle = angle
        
        # 28x28 patch to be fed into model
        self.state_img_patch = np.zeros([1,int(28),int(28)])
        
        # 56x56 patch to display the patch on the game screen
        self.state_img_patch2 = np.zeros([1,int(56),int(56)])

    def move(self, rotation):
        global episode_num
        global padsize
        global cropsize
        
        # In pygame Top left corner is the origin thus inverting Y axis by subtracting it from Sand image height
        #storing the car location
        tempx,tempy = self.position.x , 660 - self.position.y
        
        # Updating the car location
        self.position = Vector2(*self.velocity) + self.position
        self.rotation = rotation
        
        # Updating the steering angle
        self.angle = self.angle + self.rotation
        
        #Creating Numpy array for sand
        Sand = np.copy(sand)
        
        #Padding the sand image
        Sand = np.pad(Sand,self.padsize,constant_values=1.0,mode = 'constant')
        
        # Cropping a patch of size 56x56 at the current location of car on sand map
        Sand = Sand[int(tempx) - self.cropsize + self.padsize:int(tempx) + self.cropsize + self.padsize,
                   int(tempy) - self.cropsize + self.padsize:int(tempy) + self.cropsize + self.padsize]
        
        # Orienting the cropped patch along the orientation of velocity vector of car
        Sand = rotate(Sand, angle=90-(self.angle-90), reshape= False, order=1, mode='constant',  cval=1.0)
        
        #setting two pixels as black and white respectively to locate the car on cropped sand patch
        Sand[int(self.padsize)-5:int(self.padsize), int(self.padsize) - 2:int(self.padsize) + 3 ] = 0
        Sand[int(self.padsize):int(self.padsize) + 5, int(self.padsize) - 2:int(self.padsize) + 3] = 1
        
        self.state_img_patch2=Sand
        
        # Second smaller crop
        y,x = Sand.shape
        startx = x//2 - (self.cropsize//2)
        starty = y//2 - (self.cropsize//2)
        Sand = Sand[starty:starty+self.cropsize,startx:startx+self.cropsize]
        
        self.state_img_patch=Sand
        
        
        # Resizing the cropped image patch to 28x28
        #self.state_img_patch = self.state_img_patch[::2, ::2]
        self.state_img_patch2 = np.expand_dims(self.state_img_patch2, 0)
        self.state_img_patch = np.expand_dims(self.state_img_patch, 0)
        
        return 90-(self.angle-90)


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Car tutorial")
        width = 1200
        height = 660
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False
        self.car = Car()
        #self.bg_img = pygame.image.load('MASK1.png')
        #self.bg_img = pygame.transform.scale(self.bg_img,(1200,660))
        self.city_img = pygame.image.load('citymap.png')
        self.city_img = pygame.transform.scale(self.city_img,(1200,690))
        
        self.car_img = pygame.image.load('car.png')
        self.car_img = pygame.transform.scale(self.car_img,(20,15))
        
        self.dest_img = pygame.image.load('destination.jpg')
        self.dest_img = pygame.transform.scale(self.dest_img,(20,20))
        
        global total_timesteps
    


    def reset(self):
        global last_distance
        global origin_x
        global origin_y
        self.car.position.x = origin_x
        self.car.position.y  = 660 - origin_y
        x_dist = dest_x - self.car.position.x
        y_dist = dest_y - (660 - self.car.position.y)
        
        # Calculating the angle between the velocity vector and distance displacement vector in degrees
        Int_angle = -(180 / math.pi) * math.atan2(
            self.car.velocity[0] * y_dist+ self.car.velocity[1] * x_dist,
            self.car.velocity[0]* x_dist - self.car.velocity[1] * y_dist)
        
        # converting it into radians
        orientation = Int_angle/180
        
        # Calculate the distance of car current position w.r.t Destination
        self.distance = np.sqrt((self.car.position.x- dest_x) ** 2 + ( 660 - self.car.position.y - dest_y) ** 2)
        
        state = [self.car.state_img_patch , orientation, -orientation, last_distance - self.distance]
        return state


    def step(self,action):
        global dest_x
        global dest_y
        global origin_x
        global origin_y
        global done
        global last_distance
        global distance_travelled

        rotation = action.item()
        tAngle = self.car.move(rotation)
        self.Game_Screen(dest_x,dest_y,tAngle)
        self.distance = np.sqrt((self.car.position.x - dest_x) ** 2 + ( 660 - self.car.position.y - dest_y) ** 2)
        x_dist = dest_x - self.car.position.x
        y_dist = dest_y - 660 + self.car.position.y
        Int_angle = -(180 / math.pi) * math.atan2(
            self.car.velocity[0] * y_dist+ self.car.velocity[1] * x_dist,
            self.car.velocity[0]* x_dist - self.car.velocity[1] * y_dist)
        orientation = Int_angle / 180.
        
        # State vector 
        state = [self.car.state_img_patch, orientation, -orientation, last_distance-self.distance]
        
        #Penalty for moving on sand
        if sand[int(self.car.position.x), int(660 - self.car.position.y)] > 0:
            self.car.velocity = Vector2(1, 0).rotate(self.car.angle)
            prev_reward = -5 #-1

        else:  # Living Penalty
            self.car.velocity = Vector2(2.5, 0).rotate(self.car.angle)
            prev_reward = -1.5 
            # Reward for moving on road and towards destination
            if self.distance < last_distance:
                prev_reward = 1.5 
                
        # Boundary Conditions
        if self.car.position.x < 5:
        
            self.car.position.x = 5
            prev_reward = -10 #-1
            
        if self.car.position.x > self.width - 5:
            self.car.position.x= self.width - 5
            prev_reward = -10    #-1
            
        if self.car.position.y < 5:
            self.car.position.y = 5
            prev_reward = -10   #-1
            
        if self.car.position.y > self.height - 5:
            self.car.position.y = self.height - 5
            prev_reward = -10    #-1
            
        # Reward fro reaching destination
        if self.distance < 30:
            origin_x = dest_x
            origin_y = dest_y
            dest_x,dest_y= coordinates[np.random.randint(0,5)]
            prev_reward = 100
            done = True
        print(prev_reward)
        last_distance = self.distance
        return state, prev_reward, done



    def evaluate_policy(self, policy, eval_episodes=10):
        avg_reward = 0.
        for _ in range(eval_episodes):
            obs = self.reset() # ToDo reset env
            done = False
            while not done:
                action = policy.select_action(np.array(obs))
                obs,reward,done = self.step(action)
                avg_reward += reward
        avg_reward /= eval_episodes
        print("---------------------------------------")
        print("Average Reward over the Evaluation Step: %f" % (avg_reward))
        print("---------------------------------------")
        return avg_reward



    def update(self, dt):
        global first_update
        global dest_x
        global dest_y
        global longueur
        global largeur
        global prev_reward
        global reward
        global policy
        global done
        global episode_reward
        global replay_buffer
        global obs
        global new_obs
        global evaluations

        global episode_num
        global total_timesteps
        global timesteps_since_eval
        global episode_num
        global max_timesteps
        global max_episode_steps
        global episode_timesteps
        global distance_travelled
        self.width = 1200
        self.height = 660
        longueur = self.width
        largeur = self.height
        if first_update:
            init()
            evaluations = [self.evaluate_policy(policy)]
            distance_travelled=0
            done = True
            obs = self.reset()
        
        if episode_reward<-2500: # if total accumulated reward becomes more negetive than -2500 then the episode is completed
            done=True
        if total_timesteps < max_timesteps:
            if done:
                print("done-reached")
                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps != 0:
                    print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num,
                                                                                  episode_reward))
                    policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip,
                                 policy_freq)

                # We evaluate the episode and we save the policy
                if timesteps_since_eval >= eval_freq:
                    print("Reached-")
                    timesteps_since_eval %= eval_freq
                    evaluations.append(self.evaluate_policy(policy))
                    policy.save(file_name, directory="./pytorch_modelsOldLaptop")
                    np.save("./resultsOldLaptop/%s" % (file_name), evaluations)

                # When the training step is done, we reset the state of the environment
                obs = self.reset()

                # Set the Done to False
                done = False

                # Set rewards and episode timesteps to zero
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Before 10000 timesteps, we play random actions
            if total_timesteps < start_timesteps:
                action = np.random.uniform(low=-5, high=5, size=(1,))
            else:  # After start_timesteps, we switch to the model
                action = policy.select_action(np.array(obs))
                # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                if expl_noise != 0:
                    action = (action + np.random.normal(0, expl_noise, size=1)).clip(
                        -5, 5)

            # The agent performs the action in the environment, then reaches the next state and receives the reward
            new_obs,reward, done = self.step(action)

            # We check if the episode is done
            done_bool = 0 if episode_timesteps + 1 == max_episode_steps else float(
                done)

            # We increase the total reward
            episode_reward += reward
            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add((obs, new_obs, action, reward, done_bool))
            #if total_timesteps%10==1:
              #print(" ".join([str(total_timesteps), str(obs[1:]), str(new_obs[1:]), str(action), str(reward), str(done_bool)]))
            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1
            # Saving model at every 5000 iterations
            if total_timesteps%5000==1:
                print("Saving Model %s" % (file_name))
                policy.save("%s" % (file_name), directory="./pytorch_modelsOldLaptop")
                np.save("./resultsOldLaptop/%s" % (file_name), evaluations)
        else:
            action = policy.select_action(np.array(obs))
            new_obs,reward, done = self.step(action)
            obs = new_obs
            total_timesteps += 1
            if total_timesteps%1000==1:
                print(total_timesteps)


    def Game_Screen(self,dest_x,dest_y,tAngle):
        
        # Creating an image to display the state patch going into model 
        display_cam = np.copy(self.car.state_img_patch2.squeeze())
        display_cam2 = np.zeros([int(56),int(56),3])
        display_cam2[:,:,0] = display_cam*255
        display_cam2[:,:,1] = display_cam*255
        display_cam2[:,:,2] = display_cam*255
        display_cam3 = pygame.surfarray.make_surface(display_cam2)
        
        # Steering the car
        rot_img = pygame.transform.rotate(self.car_img,tAngle)
        # display City_map
        self.screen.blit(self.city_img, self.city_img.get_rect())
        # display car
        self.screen.blit(rot_img,self.car.position)
        # display destination
        self.screen.blit(self.dest_img,(dest_x, 660 - dest_y))
        # display the 28x28  state patch
        self.screen.blit(display_cam3,(400,100)) 
        pygame.display.flip()




# Initializing Global Variables
start_timesteps = 1e4  # 1e4 Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 1e3  #5e3 How often the evaluation step is performed (after how many timesteps)
max_timesteps = 5e6  #5e5 Total number of iterations/timesteps

expl_noise = 0.1  # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100  # Size of the batch
discount = 0.99  # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005  # Target network update rate
policy_noise = 0.2  # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5  # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2  #
total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0

episode_reward=0
distance_travelled=0
max_episode_steps = 1000
done = True # Episode over
load_model=True # Inference. Set to false for training from scratch

state_dim = 4
action_dim = 1
max_action = 5

replay_buffer = ReplayBuffer()

policy = TD3(state_dim, action_dim, max_action)

obs=np.array([])
new_obs=np.array([])
evaluations=[]

print("------------load model")
total_timesteps = max_timesteps
policy.load("%s" % (file_name), directory="./pytorch_models")
        

parent = Game()
start_ticks=pygame.time.get_ticks()

#parent.run()
while True:
    pygame.event.get()
    parent.update(1/60)
    time.sleep(1/60) 


