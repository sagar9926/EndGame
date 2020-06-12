import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

state_dim = 4
latent_dim = state_dim - 1 + 16


class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_img, batch_next_img, batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], [], [], []
    for i in ind:
      state, next_state, action, reward, done = self.storage[i]
      batch_img.append(np.array(state[0], copy=False))
      batch_next_img.append(np.array(next_state[0], copy=False))
      batch_states.append(np.array(state[1:], copy=False))
      batch_next_states.append(np.array(next_state[1:], copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_img), np.array(batch_next_img), np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        self.encoder = nn.ModuleList([
                
                #Convolution Block 1
                nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),# output_size = 26x26x6
                nn.ReLU(),
                nn.BatchNorm2d(8),
                nn.Dropout2d(0.1),
                
                #Convolution Block 2
                nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),# output_size = 24x24x12
                nn.ReLU(),
                nn.BatchNorm2d(10),
                nn.Dropout2d(0.1),
           
                #Convolution Block 3
                #nn.Conv2d(in_channels=12, out_channels=18, kernel_size=(3, 3), padding=0, bias=False),# output_size = 22x22x18
                #nn.ReLU(),
                #nn.BatchNorm2d(18),
                #nn.Dropout2d(0.1),

          
                #Transition Block
                nn.Conv2d(in_channels=10, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),# output_size = 22x22x8
                nn.ReLU(),
                nn.MaxPool2d(2, 2),# output_size = 11x11x6
                
                #Convolution Block 4
                nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),# output_size = 9x9x16
                nn.ReLU(),
                nn.BatchNorm2d(10),
                nn.Dropout2d(0.1),
                
                #Convolution Block 5
                nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),# output_size = 7x7x24
                nn.ReLU(),
                nn.BatchNorm2d(12),
                nn.Dropout2d(0.1),
                      
          
                #Convolution Block 6
                #nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),# output_size = 5x5x32
                #nn.ReLU(),
                #nn.BatchNorm2d(12),
                #nn.Dropout2d(0.1),
                
                #Convolution Block 6
                nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),# output_size = 5x5x16
                
                nn.AdaptiveAvgPool2d((1,1)),
                ])
    
    
        self.linear = torch.nn.ModuleList([ 
                nn.Linear(latent_dim, 400),
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Linear(300, action_dim),
                ])
        

    def forward(self, img,state):
        x = img
        for layer in self.encoder:
            x = layer(x)
        # print(x.size())
        x = x.view(-1, 16)
        x = torch.cat([x, state], 1)
        for layer in self.linear:
            x = layer(x)
            # print(x.size())
        x = self.max_action * torch.tanh(x)
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        self.encoder_C1 = nn.ModuleList([
                
                #Convolution Block 1
                nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),# output_size = 26x26x6
                nn.ReLU(),
                nn.BatchNorm2d(8),
                nn.Dropout2d(0.1),
                
                #Convolution Block 2
                nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),# output_size = 24x24x12
                nn.ReLU(),
                nn.BatchNorm2d(10),
                nn.Dropout2d(0.1),
          
                #Convolution Block 3
                #nn.Conv2d(in_channels=12, out_channels=18, kernel_size=(3, 3), padding=0, bias=False),# output_size = 22x22x18
                #nn.ReLU(),
                #nn.BatchNorm2d(18),
                #nn.Dropout2d(0.1),

          
                #Transition Block
                nn.Conv2d(in_channels=10, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),# output_size = 22x22x8
                nn.ReLU(),
                nn.MaxPool2d(2, 2),# output_size = 11x11x8
                
                #Convolution Block 4
                nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),# output_size = 9x9x16
                nn.ReLU(),
                nn.BatchNorm2d(10),
                nn.Dropout2d(0.1),
                
                #Convolution Block 5
                nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),# output_size = 7x7x24
                nn.ReLU(),
                nn.BatchNorm2d(12),
                nn.Dropout2d(0.1),
                      
          
                #Convolution Block 6
                #nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),# output_size = 5x5x32
                #nn.ReLU(),
                #nn.BatchNorm2d(12),
                #nn.Dropout2d(0.1),
                
                #Convolution Block 6
                nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),# output_size = 5x5x16
                
                nn.AdaptiveAvgPool2d((1,1)),
                ])
    
    
        self.linear_C1 = torch.nn.ModuleList([ 
                nn.Linear(latent_dim + action_dim, 400),
                nn.ReLU(),
                nn.Linear(400, 300), 
                nn.ReLU(),
                nn.Linear(300, 1),
                ])
        # Defining the second Critic neural network
        self.encoder_C2 = nn.ModuleList([
                
                #Convolution Block 1
                nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),# output_size = 26x26x6
                nn.ReLU(),
                nn.BatchNorm2d(8),
                nn.Dropout2d(0.1),
                
                #Convolution Block 2
                nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),# output_size = 24x24x12
                nn.ReLU(),
                nn.BatchNorm2d(10),
                nn.Dropout2d(0.1),
          
                #Convolution Block 3
                #nn.Conv2d(in_channels=12, out_channels=18, kernel_size=(3, 3), padding=0, bias=False),# output_size = 22x22x18
                #nn.ReLU(),
                #nn.BatchNorm2d(18),
                #nn.Dropout2d(0.1),

          
                #Transition Block
                nn.Conv2d(in_channels=10, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),# output_size = 22x22x8
                nn.ReLU(),
                nn.MaxPool2d(2, 2),# output_size = 11x11x8
                
                #Convolution Block 4
                nn.Conv2d(in_channels=8, out_channels=10,kernel_size=(3, 3), padding=0, bias=False),# output_size = 9x9x16
                nn.ReLU(),
                nn.BatchNorm2d(10),
                nn.Dropout2d(0.1),
                
                #Convolution Block 5
                nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),# output_size = 7x7x24
                nn.ReLU(),
                nn.BatchNorm2d(12),
                nn.Dropout2d(0.1),
                      
          
                #Convolution Block 6
                #nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),# output_size = 5x5x32
                #nn.ReLU(),
                #nn.BatchNorm2d(12),
                #nn.Dropout2d(0.1),
                
                #Convolution Block 6
                nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),# output_size = 5x5x16
                
                nn.AdaptiveAvgPool2d((1,1)),
                ])
    
    
        self.linear_C2 = torch.nn.ModuleList([ 
                nn.Linear(latent_dim + action_dim, 400), 
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Linear(300,1),
                ])
        
    def forward(self, img,state,u):
        x1 = img
        # Forward-Propagation on the first Critic Neural Network
        for layer in self.encoder_C1:
            x1 = layer(x1)
        # print(x.size())
        x1 = x1.view(-1, 16)
        x1u = torch.cat([x1, state,u], 1)
        for layer in self.linear_C1:
            x1u = layer(x1u)
            # print(x.size())
        
        # Forward-Propagation on the second Critic Neural Network
        x2 = img
        for layer in self.encoder_C2:
            x2 = layer(x2)
        # print(x.size())
        x2 = x2.view(-1, 16)
        x2u = torch.cat([x2, state,u], 1)
        for layer in self.linear_C2:
            x2u = layer(x2u)
            # print(x.size())
        return x1u, x2u

    def Q1(self, img, state, u):
        x1 = img
        for layer in self.encoder_C1:
            x1 = layer(x1)
        # print(x.size())
        x1 = x1.view(-1, 16)
        x1u = torch.cat([x1, state,u], 1)
        for layer in self.linear_C1:
            x1u = layer(x1u)
            # print(x.size())
        return x1u


# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Building the whole Training Process into a class

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action

    def select_action(self, state):
        image=np.expand_dims(state[0],1)
        # print(image.shape)
        state = np.array(state[1:], dtype=np.float)
        state = np.expand_dims(state,0)
        # print("Before",state.shape,state.dtype)
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        image = torch.Tensor(image).to(device)
        # print("After", state.shape)
        # state = torch.Tensor(list(state)).to(device)
        return self.actor(image,state).cpu().data.numpy().flatten()
        # return self.actor(state).gpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
            batch_img, batch_next_img, batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(
                batch_size)
            img = torch.Tensor(batch_img).to(device)
            next_img = torch.Tensor(batch_next_img).to(device)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)
            # print("action", action.shape)

            # Step 5: From the next state s’, the Actor target plays the next action a’
            next_action = self.actor_target(next_img, next_state)
            # print("next_action", next_action.shape)
            # print("next_state", next_state.shape)

            # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
            target_Q1, target_Q2 = self.critic_target(next_img, next_state, next_action)

            # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
            target_Q = torch.min(target_Q1, target_Q2)

            # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
            current_Q1, current_Q2 = self.critic(img, state, action)

            # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
            if it % policy_freq == 0:
                actor_loss = -self.critic.Q1(img, state, self.actor(img, state)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor (1).pth' % (directory, filename),map_location = 'cpu'))
        self.critic.load_state_dict(torch.load('%s/%s_critic (1).pth' % (directory, filename),map_location = 'cpu'))


