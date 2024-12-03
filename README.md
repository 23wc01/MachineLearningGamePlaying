# GamePlayingAI
Artificial Intelligence 7750 - Graduate final project

# **About the Game**
Agent will compare the usage of Neural Network vs Deep-Q-Network (DQN) learning to increasingly improve itself on playing a Snake game where

## **Actions**
3 possible actions
[1,0,0] = forward (continues in current direction)

[0,1,0] = turn right

[0,0,1] = turn left

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
![image](https://github.com/user-attachments/assets/ca04816d-ebd8-4d1f-950a-601413f017dd)

# **Neural Network**
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



