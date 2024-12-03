# For implementing snake game (environment)
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# rgb colors
GAMEBOARD_COLOR = (255, 255, 255)  # Gameboard is white
SCOREBOARD_COLOR = (0, 0, 0)  # Scoreboard displays in black text
BROWN = (150, 75, 0)  # Mice (food) is brown color
SNAKE_SPOTS_COLOR = (0, 255, 150)  # Snake spots are aquamarine color
SNAKE_BODY_COLOR = (0, 255, 0)  # Snake body is green color

BLOCK_SIZE = 20
SPEED = 80

# Define Point tuple (essentially an x-y coordinate value, where .x retrieves 1st tuple element and .y retrieves 2nd tuple element)
Point = namedtuple('Point', 'x, y')


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


pygame.init()
font = pygame.font.SysFont('arial', 25)


class SnakeGameAI:

    def __init__(self, w=640*2, h=480*2):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset_game()

    def reset_game(self):
        # init game state
        self.direction = Direction.RIGHT
        # Initialize snake position
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        # Reset score
        self.score = 0
        # Initialize new food position
        self.food = None
        self._place_food()
        self.steps_taken = 0  # Reset # of steps taken

    def _place_food(self):
        """
        Place mice at random Point(x,y) in game screen
        """
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            #self.score += 10
            self._place_food()

    def take_action(self, action):
        # 0. Increment # of steps
        self.steps_taken += 1
        # 1. collect user input
        for event in pygame.event.get():
            # User input can only stop AI from playing game
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.steps_taken > 100 * len(self.snake):
            reward -= 10
            game_over = True
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:  # Snake head at same coordinate as food (i.e. snake is "eating" food)
            reward += 10
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(GAMEBOARD_COLOR)

        for pt in self.snake:
            pygame.draw.rect(self.display, SNAKE_SPOTS_COLOR,
                             pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, SNAKE_BODY_COLOR,
                             pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(
            self.display, BROWN,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, SCOREBOARD_COLOR)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        """
        [1,0,0] = forward
        [0,1,0] = right turn
        [0,0,1] = left turn
        """
        clockwise = [
            Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP
        ]
        index = clockwise.index(
            self.direction
        )  # Find element in clockwise that self.direction is equal to. Get the element's index
        if np.array_equal(action, [1, 0, 0]):
            new_direction = clockwise[index]  # Continue in same direction
        elif np.array_equal(action, [0, 1, 0]):  # Turn right
            new_index = (index + 1)  # Represents r->d->l->u
            new_index = new_index % 4  # Represents u->r wraparound
            new_direction = clockwise[new_index]  # Turn right
        else:
            new_index = (index - 1)  # r<-d<-l<-u and u<-r (-1 case)
            new_direction = clockwise[new_index]
        self.direction = new_direction

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
