import os
import numpy as np
import random
import torch
from torch import nn, optim
import torch.nn.functional as F


class Linear_Net(nn.Module):

  def __init__(self, hidden_size, input_size=11, output_size=3):
    super().__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.linear2 = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    x = self.linear1(x)
    x = F.relu(
        x
    )  # Apply relu so only positive results kept & negative results set to 0. This mitigates vanishing gradient problem
    x = self.linear2(x)
    return x

  def save(self, file_name="model.path"):
    model_folder_path = "./model"
    if not os.path.exists(model_folder_path):
      os.makedirs(model_folder_path)
    file_name = os.path.join(model_folder_path, file_name)
    torch.save(self.state_dict(), file_name)

 
class NeuralNetTrainer:
    def __init__(self, model, learn_rate):
        self.model = model
        self.learn_rate = learn_rate
        self.optimizer = optim.Adam(model.parameters(), lr=self.learn_rate)
        self.criterion = nn.CrossEntropyLoss() # Uses cross-entropy loss instead of Q-Learning

    def heuristic(self, state: list[int]) -> list[int]:
      possible_action = [1 if state[i] == 0 else 0 for i in range(3)]
      possible_action_count = sum(possible_action)
      if possible_action_count > 0:
          directions = state[3:7]
          food_directions = state[7:]
          # direction = [left, right, up, down] with 1-hot encoding to denote current direction
          current_direction = np.argmax(directions)
          if food_directions[current_direction] == 1:
            return [1,0,0]
          else:
            food_directions = food_directions[2:] if current_direction >= 1 else food_directions[:2]
            food_direction = np.argmax(food_directions)
            if current_direction == 0: # Current direction=left
              return [0,1,0] if food_direction == 0 else [0,0,1] # food_direction == 0 means food is up
            elif current_direction == 1: # Current direction=right
              return [0,0,1] if food_direction == 0 else [0,1,0]
            elif current_direction == 2: # Current direction=up
              return [0,0,1] if food_direction == 0 else [0,1,0] # food_direction == 0 means food is left
            else: # Current direction=down
              return [0,1,0] if food_direction == 0 else [0,0,1]
                    
      elif possible_action_count == 1:
          return possible_action # Return only action that won't result in game over
      else:
          random.seed(42)
          random_index = random.randint(0, 2)
          decided_action = [0,0,0]
          decided_action[random_index] = 1
          return decided_action # Default go random direction if danger on all sides 
            
    def train_step(self, state, action, reward, next_state, done):
      # Current state & next move as input, then trains the network to predict the next move.
        no_tensor = state
        state = np.array(state)
        state = torch.tensor(state, dtype=torch.float)
        next_state = np.array(next_state) # THIS IS TO SPEED UP TORCH CONVERSION
        next_state = torch.tensor(next_state, dtype=torch.long)

        # If only 1 state, convert it to format (n, argument)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done, )
        
        # 1. Predict next move
        prediction = self.model(state)

        # 2. Use heuristic for each state in the batch
        targets = []
        for s in state:
            target = self.heuristic(s.numpy())  # s is a tensor, convert to numpy for heuristic
            targets.append(target)
        
        # Convert targets to tensor (from list of tensors)
        targets = torch.stack([torch.tensor(t, dtype=torch.float) for t in targets])

        # 3. Find the most likely class from target
        target_index = torch.argmax(targets, dim=1)  # Get the index of the highest value in each target
        target_class_index = target_index.long()  # Ensure target class is long dtype for cross-entropy loss

        loss = self.criterion(prediction, target_class_index)
        # 2. Backpropagate and update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class QTrainer:
  def __init__(self, model, learn_rate, gamma):
    self.model = model
    self.learn_rate = learn_rate
    self.gamma = gamma
    self.optimizer = optim.Adam(model.parameters(), lr=self.learn_rate)
    self.criterion = nn.MSELoss()

  def train_step(self, state, action, reward, next_state, done):
    # If multiple states, actions, rewards passed in, the format is already in (n, element) format
    state = np.array(state) # THIS IS TO SPEED UP TORCH CONVERSION
    state = torch.tensor(state, dtype=torch.float)
    next_state = np.array(next_state) # THIS IS TO SPEED UP TORCH CONVERSION
    next_state = torch.tensor(next_state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.long)
    reward = torch.tensor(reward, dtype=torch.float)

    # If only 1 state, convert it to format (n, argument)
    if len(state.shape) == 1:
      state = torch.unsqueeze(state, 0)
      action = torch.unsqueeze(action, 0)
      reward = torch.unsqueeze(reward, 0)
      next_state = torch.unsqueeze(next_state, 0)
      done = (done, )

    # 1. Predict Q value with current state
    prediction = self.model(state)
    target = prediction.clone()
    for i in range(len(done)):
      new_Q = reward[i]
      if not done[i]:
        new_Q += self.gamma * torch.max(self.model(next_state[i]))
      target[i][torch.argmax(action).item()] = new_Q
    # 2. new_Q = r + y * max(next_Qvalue_prediciton) -> Only do if not done
    self.optimizer.zero_grad()
    loss = self.criterion(target, prediction)
    loss.backward()  # Apply backward propagation.

    self.optimizer.step()
