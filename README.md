# GamePlayingAI
Artificial Intelligence 7750 - Graduate final project

# **About the Game**
Agent will compare the usage of Neural Network vs Deep-Q-Network (DQN) learning to increasingly improve itself on playing a Snake game where

## **State**

* If danger (snake collides with its own body or game window boundary) is forward, right, or left of the snake.
* If current direction of snake is left, right, up, down.
* If mice is left, right, up, down of snake (can have 2 combos if it's diagonal).
Ex: state = [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0] = Danger to left of 
snake, snake moving downward, and mice (food) is to right & up of snake.

# **Neural Network**
* Uses heuristic function to determine target action to take.
# **DQN**
**Reward & Penalty**

* eat_mice = +10
* game_over = -10
* idle_steps_after_long_time = -10 (idle/useless steps limit porportional to length of snake*100)


**Actions**

[1,0,0] = forward (continues in current direction)

[0,1,0] = turn right

[0,0,1] = turn left
