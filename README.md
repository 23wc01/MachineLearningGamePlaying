# GamePlayingAI

Artificial Intelligence 7750 - Graduate final project
Relevant papers
* [Neural Learning of Heuristic Functions for General Game Playing](https://core.ac.uk/download/pdf/302082154.pdf)
* [Neural network architecture for a snake game](https://www.researchgate.net/figure/Neural-network-architecture-for-a-snake-game_fig3_334998694)
* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602v1)
* [A Deep Q-Learning based approach applied to the Snake game](https://www.researchgate.net/publication/351884746_A_Deep_Q-Learning_based_approach_applied_to_the_Snake_game)

# **Demo video**
![image](https://github.com/user-attachments/assets/e4fae0dd-c99f-4a8d-9d61-700e561aa2ea)

![[DQN vs Neural network with heuristic to play Snake](https://www.youtube.com/watch?v=rDVHLXzJPvQ)

# **Summary**

Agent will compare the usage of Neural Network with heuristic vs Deep-Q-Network (DQN) learning to increasingly improve itself on playing a Snake game.

# **Environment**
## **Actions**

Represent snake's 3 possible actions using one-hot encoding, with 1 = action to do and 0 = action to not do.

`[1,0,0]` = forward (continues in current direction)

`[0,1,0]` = turn right

`[0,0,1]` = turn left

## **State**
state = Represents 11 conditions using one-hot encoding, with 1 = condition met and 0 = condition unmet.
* If danger (snake collides with its own body or game window boundary) is forward, right, and or left of the snake.
* If current direction of snake is going left, right, up, or down.
* If mice is left, right, up, and or down of snake (can have 2 combos if it's diagonal).
  ~~~
  [danger_forward, danger_right_turn, danger_left_turn,
   going_left, going_right, going_up, going_down,
   mice_left, mice_right, mice_up, mice_down]
  ~~~
Ex: `state = [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]` = Danger to left of 
snake, snake moving downward, and mice (food) is to right & up of snake.

## **Model**
![image](https://github.com/user-attachments/assets/a9813a5c-0292-440a-b956-29154d6450c3)

# **Neural Network with heuristic**
Uses **heuristic** function to determine target action to take:
1. decided_action = Direction(s) where there's no danger
  2. decided_action = If mice in same direction snake is heading towards, return "go forward" action
  3. decided_action = If mice in direction that snake can turn towards, return that direction
4. decided_action = If no previous conditions matched/danger everywhere just return random action

![image](https://github.com/user-attachments/assets/df71717f-6d7b-4fa0-8a41-e2d725ea3ef7)


# **DQN**
## **Reward & Penalty**

* eat_mice = +10
* game_over = -10
* idle_steps_after_long_time = -10 (idle/useless steps limit porportional to length of snake*100)

## **Q learning**
Uses Bellman equation to calculate new Q values
![image](https://github.com/user-attachments/assets/34793a16-18c2-453c-9a69-3659f2b89d56)

## **Gamma & Epsilon**

`epsilon = 80-m` Random exploration if `randint(0, 200) < epsilon`, else do exploitation

`gamma = 0.9` Results fair better when gamma, aka discount factor, set closer to 1 (aka values future rewards almost as much as current rewards)


# **Results comparison**
Both experiments ran for 10 minutes.

**Conclusion**
As can be seen, whereas Neural Network with heuristic approach improves in performance quickly, as time passes the increase in performance plateaus. In DQN, the performance doesn't improve as quickly initially, but as time goes on there is a clear and continuous increase in performance and no signs of plateauing yet even after 10 minutes. 
## Neural Network with heuristic

![image](https://github.com/user-attachments/assets/322cffba-b4f3-4279-afe7-ab3bbdd21ba1)

![image](https://github.com/user-attachments/assets/83143a06-0c80-4bb8-a41c-3e6e9560c76a)


## DQN

![image](https://github.com/user-attachments/assets/6a2c34d9-f87b-4f84-b065-c0357a2b650b)

![image](https://github.com/user-attachments/assets/61e78de9-d84e-4ae7-94b9-4fe8a075d32e)


# **References**

* [chatgpt to debug heuristic function](https://chatgpt.com)
* [Python + PyTorch + Pygame Reinforcement Learning – Train an AI to Play Snake](https://www.youtube.com/watch?v=L8ypSXwyBds)

