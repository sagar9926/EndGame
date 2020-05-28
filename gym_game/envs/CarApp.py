# -*- coding: utf-8 -*-
"""
Created on Thu May 28 08:05:19 2020

@author: Sagar.Agrawal
"""


import pygame
from pygame.math import Vector2
from pygame.locals import *
import random
import math
from scipy.ndimage import rotate
import sys
import numpy as np
last_reward = 0
Screen_width = 1200
Screen_height = 554
car_width = 10
car_height = 20
cropsize= 28
padsize = 28

class Car:
    
    MAX_FORWARD_SPEED = 1
    MAX_REVERSE_SPEED = 1
    ACCELERATsION = 0.1
    TURN_SPEED = 10
    

    def __init__(self,image,position):
        self.position = position
        self.speed = self.direction = 0
        self.k_left = self.k_right = self.k_down = self.k_up = 0
        self.src_image = image
        self.center = [self.position[0] + 0.5*car_width, self.position[1] + 0.5*car_height]
        self.current_distance = 0
        self.previous_distance = 0
        self.destination_x = 0 
        self.destination_y = 0
        
        
    def Destination_point(self):
        #Random destination on road
        map_image = pygame.image.load('C:\\Users\\sagar.agrawal\\Downloads\\EndGame\\framework_tutorial-master\\map1.png').convert()
        while True:
            self.destination_x = random.randrange(0,Screen_width - 10)
            self.destination_y = random.randrange(0,Screen_height - 10)
    
            if map_image.get_at((self.destination_x,self.destination_y)) == (0, 0, 0, 255):break
            else:continue
        
        

    def update(self):
        #print(dt)
        self.speed += (self.k_up + self.k_down)
        if self.speed > self.MAX_FORWARD_SPEED:
            self.speed = self.MAX_FORWARD_SPEED
        if self.speed < -self.MAX_REVERSE_SPEED:
            self.speed = -self.MAX_REVERSE_SPEED
        self.direction += (self.k_right + self.k_left)
        if self.direction > 360 :
            self.direction -= 360 
        elif self.direction < -360:
            self.direction +=360
            
        x, y = (self.position)
        rad = self.direction * math.pi / 180
        x += -self.speed*math.sin(rad)
        y += -self.speed*math.cos(rad)
        self.position = (x, y)
        self.image = pygame.transform.rotate(self.src_image, self.direction)
        self.rect = self.image.get_rect()
        self.rect.center = self.position
        self.center = [self.position[0] + 0.5*car_width, self.position[1] + 0.5*car_height]
        self.previous_distance = np.sqrt((self.position[0] - self.destination_x) ** 2 + (self.position[1] - self.destination_y) ** 2)
        
        if((self.position[0] > (self.destination_x-30) and self.position[0] < (self.destination_x + 30)) and (self.position[1] > (self.destination_y-30) and self.position[1] < (self.destination_y+30))):
                self.Destination_point()
                self.current_distance = np.sqrt((self.position[0] - self.destination_x) ** 2 + (self.position[1] - self.destination_y) ** 2)
                
        

class Game:
    
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Car tutorial")
        global car_height
        global car_width
        width = Screen_width
        height = Screen_height
        self.screen = pygame.display.set_mode((width, height))
        #Adding Map to our game
        self.map_image = pygame.image.load('map.png').convert()
        self.map_rect = self.map_image.get_rect()
        self.dest_image =pygame.image.load('destination.jpg').convert()
        self.dest_image = pygame.transform.scale(self.dest_image,(20,40))
        self.car_image_path = "car.png"
        self.car_image = pygame.image.load(self.car_image_path)
        self.car_image = pygame.transform.scale(self.car_image, (car_width, car_height))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False
        self.win_condition = None
        self.car = Car(self.car_image,(10,10))
        
    def is_done(self):
        if((self.car.position[0] > (self.car.destination_x-30) and self.car.position[0] < (self.car.destination_x + 30)) and (self.car.position[1] > (self.car.destination_y-30) and self.car.position[1] < (self.car.destination_y+30))):
            return True
        return False  
        
        
    def observe(self):
        # Get sand image
        sand = pygame.surfarray.array3d(self.map_image)
        tempsand = np.copy(sand)
        tempsand = tempsand.swapaxes(0,1)
        
        tempsand = np.pad(tempsand,((int(padsize/2),int(padsize/2)), (int(padsize/2),int(padsize/2)), (0, 0)), constant_values=255)
        carx_sand = self.car.position[0] + padsize/2
        cary_sand = self.car.position[1] + padsize/2
        center_x = carx_sand + car_width/2
        center_y = cary_sand + car_height/2

        #first small crop
        startx, starty = int(center_x-(cropsize)), int(center_y-(cropsize))
        crop1 = tempsand[starty:starty+cropsize*2, startx:startx+cropsize*2,:]
        #rotate
        crop1 = rotate(crop1, -1*self.car.direction, mode='constant', cval=255, reshape=False, prefilter=False)
        
        #again final crop
        startx, starty = int(crop1.shape[0]//2-cropsize//2), int(crop1.shape[0]//2-cropsize//2)

        im = crop1[starty:starty+cropsize, startx:startx+cropsize]
        return(pygame.surfarray.make_surface(im))
        
    def evaluate(self):
        global last_reward
        global Screen_height
        global Screen_width
        
        # Destination Reward
        if self.car.previous_distance < self.car.current_distance:
            last_reward += 0.001
        elif self.car.previous_distance > self.car.current_distance:
            last_reward -= 0.001
            
        
        #Boundary Penalty
        if self.car.position[0] < 5:
            last_reward = -10 #-1
            pos = list(self.car.position)
            pos[0] = 5
            self.car.position = tuple(pos)
            
        if self.car.position[0] > Screen_width - 15:
            last_reward = -10    #-1
            pos = list(self.car.position)
            pos[0] = Screen_width - 15
            self.car.position = tuple(pos)
            
        if self.car.position[1] < 5:
            last_reward = -10   #-1
            pos = list(self.car.position)
            pos[1] = 5
            self.car.position = tuple(pos)
            
        if self.car.position[1] > Screen_height - 15:
            last_reward = -10    #-1
            pos = list(self.car.position)
            pos[1] = Screen_height - 15
            self.car.position = tuple(pos)
            

    def action(self,action):
        
        self.car.k_right = action[0] * -5  #Steering
        self.car.k_left = action[0] * 5
        self.car.k_up = action[1] * 2      #Acceleration
        self.car.k_down = action[1] * -2 
        
        self.car.update()

            # Drawing
            #self.screen.fill((0, 0, 0))
            
    def view(self):
        self.screen.blit(self.map_image, self.map_rect)
        self.screen.blit(self.dest_image, (self.car.destination_x,self.car.destination_y))
        self.screen.blit(self.car.image,car.position)
        self.screen.blit(self.observe(),(700,400))
           
        pygame.display.flip()
        self.clock.tick(self.ticks)
