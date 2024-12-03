# For implementing neural network for DQN
from collections import deque  # Stores agent's memory
import numpy as np
from math import cos
import random
import torch
from ai_snake_game import SnakeGameAI, Point, Direction
from model import Linear_Net, NeuralNetTrainer, QTrainer
from plotter import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LEARN_RATE = 0.001
BLOCK_SIZE = 20


class Agent:

  def __init__(self):
    self.n_games = 0
    self.epsilon = 0  # Controls randomness for exploration
    self.gamma = 0.9  # Discount rate (how much weight placed on future reward? Will it impact the agent alot?). Must < 1
    self.memory = deque(
        maxlen=MAX_MEMORY)  #popleft() of memory if len(memory) > MAX_MEMORY
    self.model = Linear_Net(
        input_size=11, hidden_size=256, output_size=3
    )  # Input must be 11 bc there are 11 conditions represented in 1 state. Output must be 3 bc it'll be mapped to an action [0,0,0] later
    self.trainer = QTrainer(self.model, LEARN_RATE, self.gamma)
    #self.trainer = NeuralNetTrainer(self.model, LEARN_RATE)
  def get_state(self, game):
    # Only 1 element == True, rest == False.
    current_dir_left = game.direction == Direction.LEFT
    current_dir_right = game.direction == Direction.RIGHT
    current_dir_up = game.direction == Direction.UP
    current_dir_down = game.direction == Direction.DOWN

    head = game.snake[0]
    left_pt = Point(head.x - 20, head.y)
    right_pt = Point(head.x + 20, head.y)
    up_pt = Point(head.x, head.y - 20)
    down_pt = Point(head.x, head.y + 20)

    danger_forward = ((current_dir_left and game.is_collision(left_pt))
                      or (current_dir_right and game.is_collision(right_pt))
                      or (current_dir_up and game.is_collision(up_pt))
                      or (current_dir_down and game.is_collision(down_pt)))
    danger_right = ((current_dir_left and game.is_collision(up_pt))
                    or (current_dir_right and game.is_collision(down_pt))
                    or (current_dir_up and game.is_collision(right_pt))
                    or (current_dir_down and game.is_collision(left_pt)))
    danger_left = ((current_dir_left and game.is_collision(down_pt))
                   or (current_dir_right and game.is_collision(up_pt))
                   or (current_dir_up and game.is_collision(left_pt))
                   or (current_dir_down and game.is_collision(right_pt)))

    state = [
        danger_forward,
        danger_right,
        danger_left,
        current_dir_left,
        current_dir_right,
        current_dir_up,
        current_dir_down,
        game.food.x < game.head.x,  # food left
        game.food.x > game.head.x,  # food right
        game.food.y < game.head.y,  # food up
        game.food.y > game.head.y  # food down
    ]
    return np.array(state, dtype=int)

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state,
                        done))  # popleft() if MAX_MEMORY reached

  def train_short_memory(self, state, action, reward, next_state, done):
    self.trainer.train_step(state, action, reward, next_state, done)

  def train_long_memory(self):
    if len(self.memory) > BATCH_SIZE:
      mini_sample = random.sample(self.memory, BATCH_SIZE)  # Returns tuples
    else:
      mini_sample = self.memory
    states, actions, rewards, next_states, dones = zip(*mini_sample)
    self.trainer.train_step(states, actions, rewards, next_states, dones)

  def get_action(self, state):
    """
    # Neural Net implementation
    decided_action = [0, 0, 0]
    state0 = torch.tensor(state, dtype=torch.float)
    prediction = self.model(state0)  # Can be array of float elements
    action_index = torch.argmax(prediction).item()  # Returns max element's index in prediction
    decided_action[action_index] = 1
    return decided_action
  
    """
    # Deep Q Learning implementation
    self.epsilon = 80 - self.n_games
    # self.epsilon = 0.4 * (cos(self.n_games/160) * np.pi) + 0.5
    decided_action = [0, 0, 0]
    if random.randint(0, 200) < self.epsilon:  # Random Exploration
      random_action_index = random.randint(0, 2)
      decided_action[random_action_index] = 1
    else:  # Exploitation
      state0 = torch.tensor(state, dtype=torch.float)
      prediction = self.model(state0)  # Can be array of float elements
      action_index = torch.argmax(prediction).item()  # Returns max element's index in prediction
      decided_action[action_index] = 1
    return decided_action
    
    
    

def train():
  plot_scores = []
  plot_mean_scores = []
  total_score = 0
  highest_record = 0
  agent = Agent()
  game = SnakeGameAI()
  while True:
    old_state = agent.get_state(game)
    decided_action = agent.get_action(old_state)
    reward, done, score = game.take_action(decided_action)
    new_state = agent.get_state(game)

    # Train short memory
    agent.train_short_memory(old_state, decided_action, reward, new_state,
                             done)
    agent.remember(old_state, decided_action, reward, new_state, done)
    if done:
      game.reset_game()
      agent.n_games += 1
      agent.train_long_memory()  # Experience replay. Trains on all previous moves to improve itself.
      if score > highest_record:
        highest_record = score
        agent.model.save()
        print(f"Game count: {agent.n_games}. Score: {score}\nHighest record: {highest_record}")

        plot_scores.append(score)
        total_score += score 
        mean_score = total_score/ agent.n_games
        plot_mean_scores.append(mean_score)
        plot(plot_scores, plot_mean_scores)
if __name__ == "__main__":
  train()
