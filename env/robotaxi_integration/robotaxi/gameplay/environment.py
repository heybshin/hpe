import pprint
import random
import time
from collections import deque
import numpy as np
import pandas as pd

from .entities import Snake, Field, CellType, SnakeAction, SnakeDirection, ALL_SNAKE_ACTIONS, SNAKE_GROW, WALL_WARP, \
    Point

PUNISH_WALL = False
PLAY_SOUND = False

RANDOM_BORN_VEHICLE = [
    [Point(8, 1), SnakeDirection.EAST],
    [Point(18, 1), SnakeDirection.EAST],
    [Point(16, 21), SnakeDirection.WEST],
    [Point(6, 1), SnakeDirection.EAST],
    [Point(7, 1), SnakeDirection.EAST],
    [Point(17, 1), SnakeDirection.EAST],
    [Point(21, 16), SnakeDirection.NORTH],
    [Point(17, 21), SnakeDirection.WEST],
    [Point(4, 21), SnakeDirection.WEST],
    [Point(5, 21), SnakeDirection.WEST],
    [Point(6, 21), SnakeDirection.WEST],
    [Point(1, 5), SnakeDirection.SOUTH],
    [Point(1, 6), SnakeDirection.SOUTH],
    [Point(1, 7), SnakeDirection.SOUTH],
]

OTHER_BORN_VEHICLE = [
    [Point(6, 1), SnakeDirection.EAST],
    [Point(7, 1), SnakeDirection.EAST],
    [Point(17, 21), SnakeDirection.WEST],
    [Point(4, 21), SnakeDirection.WEST],
    [Point(5, 21), SnakeDirection.WEST],
    [Point(6, 21), SnakeDirection.WEST],
    [Point(1, 5), SnakeDirection.SOUTH],
    [Point(1, 6), SnakeDirection.SOUTH],
    [Point(1, 7), SnakeDirection.SOUTH],
]

RANDOM_BORN_PEDESTRIAN = [
    [Point(3, 1), SnakeDirection.EAST],
    [Point(9, 1), SnakeDirection.EAST],
    [Point(15, 1), SnakeDirection.EAST],
    [Point(19, 1), SnakeDirection.EAST],
    [Point(3, 21), SnakeDirection.WEST],
    [Point(9, 21), SnakeDirection.WEST],
    [Point(15, 21), SnakeDirection.WEST],
    [Point(19, 21), SnakeDirection.WEST],
    [Point(21, 14), SnakeDirection.NORTH],
    [Point(21, 17), SnakeDirection.NORTH],
    [Point(1, 4), SnakeDirection.SOUTH],
    [Point(1, 10), SnakeDirection.SOUTH],
    [Point(1, 14), SnakeDirection.SOUTH],
    [Point(1, 17), SnakeDirection.SOUTH],
]

PICKUP_BORN_PEDESTRIAN = [
    # [Point(3, 3), SnakeDirection.EAST],
    # [Point(2, 4), SnakeDirection.SOUTH],
    # [Point(3, 11), SnakeDirection.WEST],
    # [Point(2, 10), SnakeDirection.SOUTH],
    # [Point(3, 13), SnakeDirection.EAST],
    # [Point(2, 14), SnakeDirection.SOUTH],
    # [Point(3, 18), SnakeDirection.WEST],
    # [Point(2, 17), SnakeDirection.SOUTH],

    [Point(9, 3), SnakeDirection.EAST],
    [Point(10, 4), SnakeDirection.NORTH],
    [Point(9, 11), SnakeDirection.WEST],
    [Point(10, 10), SnakeDirection.NORTH],
    [Point(9, 13), SnakeDirection.EAST],
    [Point(10, 14), SnakeDirection.NORTH],
    [Point(9, 18), SnakeDirection.WEST],
    [Point(10, 17), SnakeDirection.NORTH],

    [Point(15, 3), SnakeDirection.EAST],
    [Point(14, 4), SnakeDirection.SOUTH],
    [Point(15, 11), SnakeDirection.WEST],
    [Point(14, 10), SnakeDirection.SOUTH],
    [Point(15, 13), SnakeDirection.EAST],
    [Point(14, 14), SnakeDirection.SOUTH],
    [Point(15, 18), SnakeDirection.WEST],
    [Point(14, 17), SnakeDirection.SOUTH],

    # [Point(19, 3), SnakeDirection.EAST],
    # [Point(19, 5), SnakeDirection.WEST],
    [Point(19, 11), SnakeDirection.WEST],
    [Point(19, 9), SnakeDirection.EAST],
    [Point(19, 13), SnakeDirection.EAST],
    [Point(20, 14), SnakeDirection.NORTH],
    # [Point(19, 18), SnakeDirection.WEST],
    # [Point(20, 17), SnakeDirection.NORTH],
]

PICKUP_CAR_POS = [
    [Point(8, 1), SnakeDirection.EAST],
    [Point(8, 1), SnakeDirection.EAST],
    [Point(17, 1), SnakeDirection.EAST],
    [Point(16, 21), SnakeDirection.WEST],
    [Point(18, 1), SnakeDirection.EAST],
    [Point(21, 16), SnakeDirection.NORTH],
    [Point(21, 16), SnakeDirection.NORTH],
]
class Environment(object):
    """
    Represents the RL environment for the Snake game that implements the game logic,
    provides rewards for the agent and keeps track of game statistics.
    """

    def __init__(self, config, stationary=False, collaboration=False, verbose=1, participant=None, mode=None):
        """
        Create a new Snake RL environment.
        
        Args:
            config (dict): level configuration, typically found in JSON configs.  
            verbose (int): verbosity level:
                0 = do not write any debug information;
                1 = write a CSV file containing the statistics for every episode;
                2 = same as 1, but also write a full log file containing the state of each timestep.
        """
        self.config = config
        self.initial_config = config['field']
        self.initial_config_pedestrian = config['field_pedestrian']
        self.field = Field(level_map=self.initial_config)
        self.field_pedestrian = Field(level_map=self.initial_config_pedestrian)
        self.vehicle = None
        self.collaborator = None
        self.collaboration = collaboration
        self.in_pit = False
        self.original_direction = SnakeDirection.NORTH

        self.initial_snake_length = config['initial_snake_length']
        self.rewards = config['rewards']
        self.max_step_limit = config.get('max_step_limit', 100)
        self.punch = False
        self.punch_collaborator = False
        self.is_game_over = False
        self.stationary = stationary
        self.good_fruit_revealed = False
        self.bad_fruit_revealed = False
        self.lava_revealed = False
        self.good_fruit_num = 2
        self.bad_fruit_num = 2 if not self.collaboration else 0
        self.lava_num = 2

        self.timestep_index = 0
        self.current_action = None
        self.current_action_collaborator = None
        self.stats = EpisodeStatistics()
        self.stats_collaborator = EpisodeStatistics()
        self.verbose = verbose
        self.debug_file = None
        self.stats_file = None
        self.stats_file_collaborator = None
        self.debug_file_collaborator = None
        self.participant = participant
        self.random_place_vehicle = config.get('random_place_snake')  # 0 = fixed position, 1 = random position
        self.num_initial_vehicles = self.config.get("num_initial_vehicles", 1)
        self.num_initial_pedestrians = self.config.get("num_initial_pedestrians", 1)
        if self.num_initial_vehicles > 1: self.vehicles = []
        if self.num_initial_pedestrians > 0: self.pedestrians = []
        self.mode = mode

    def update_field(self, config):
        self.initial_config = config['field']
        self.field = Field(level_map=self.initial_config)
    
    def new_episode_with_field(self, field):

        self.field = field
        self.field.create_level(init_cells=False)
        self.stats.reset()
        self.timestep_index = 0        
        self.vehicle = Snake(self.field.find_snake_head(), length=self.initial_snake_length, body_coord=self.field.find_snake_body())

        self.good_event, self.bad_fruit, self.lava, self.pit = self.field.get_initial_items()
        self.current_action = None
        self.punch = False
        self.punch_wall_pos = [0,0]
        self.is_game_over = False
        self.good_miss_ct = 0 
        self.bad_miss_ct = 0 
        self.lava_miss_ct = 0 
        result = TimestepResult(
            observation=self.get_observation(),
            reward=0,
            is_episode_end=self.is_game_over
        )

        self.record_timestep_stats(result)
        return result
       
    def seed(self, value):
        """ Initialize the random state of the environment to make results reproducible. """
        random.seed(value)
        np.random.seed(value)

    @property
    def observation_shape(self):
        """ Get the shape of the state observed at each timestep. """
        return self.field.size, self.field.size

    @property
    def num_actions(self):
        """ Get the number of actions the agent can take. """
        return len(ALL_SNAKE_ACTIONS)

    def new_agent_vehicle(self, born_head_idx=None):
        if born_head_idx is None:
            if self.mode == 'surveillance':
                born_index = np.random.choice(len(RANDOM_BORN_VEHICLE))
                vehicle_head = RANDOM_BORN_VEHICLE[born_index][0]
            elif self.mode == 'navigation':
                born_index = np.random.choice(len(OTHER_BORN_VEHICLE))
                vehicle_head = OTHER_BORN_VEHICLE[born_index][0]
            vehicle_head = Point(vehicle_head.y, vehicle_head.x)
            for vehicle in self.vehicles:
                if vehicle_head == vehicle.head:
                    return
            vehicle = Snake(vehicle_head, length=self.initial_snake_length)
            if self.mode == 'surveillance':
                vehicle.direction = RANDOM_BORN_VEHICLE[born_index][1]
            elif self.mode == 'navigation':
                vehicle.direction = OTHER_BORN_VEHICLE[born_index][1]

            if self.mode == 'navigation':
                vehicle.icon_type = np.random.randint(1, 4)
            elif self.mode == 'surveillance':
                vehicle.icon_type = np.random.randint(4)

            vehicle.last_head = vehicle_head
            self.field.place_snake(vehicle)

            self.vehicles.append(vehicle)
        else:
            vehicle_head = PICKUP_CAR_POS[born_head_idx][0]
            vehicle_head = Point(vehicle_head.y, vehicle_head.x)
            for vehicle in self.vehicles:
                if vehicle_head == vehicle.head:
                    return False
            vehicle = Snake(vehicle_head, length=self.initial_snake_length)
            vehicle.icon_type = 0
            vehicle.last_head = vehicle_head
            self.field.place_snake(vehicle)
            vehicle.direction = PICKUP_CAR_POS[born_head_idx][1]
            self.vehicles.append(vehicle)
            return True
    def new_agent_pedestrian(self):
        if self.mode == 'navigation':
            pesdestrian_born_pos = PICKUP_BORN_PEDESTRIAN
        elif self.mode == 'surveillance':
            pesdestrian_born_pos = RANDOM_BORN_PEDESTRIAN

        born_index = np.random.choice(len(pesdestrian_born_pos))
        pedestrian_head = pesdestrian_born_pos[born_index][0]
        pedestrian_head = Point(pedestrian_head.y, pedestrian_head.x)
        pedestrian = Snake(pedestrian_head, length=self.initial_snake_length)
        pedestrian.direction = pesdestrian_born_pos[born_index][1]
        pedestrian.last_head = pedestrian_head
        self.field_pedestrian.place_snake(pedestrian)
        self.pedestrians.append(pedestrian)


    def new_episode(self):
        """ Reset the environment and begin a new episode. """
        self.vehicles = []
        self.pedestrians = []
        self.seed(random.randint(12345, 99999))
        self.field = Field(level_map=self.initial_config)
        self.field.create_level()
        self.field_pedestrian = Field(level_map=self.initial_config_pedestrian)
        self.field_pedestrian.create_level()

        self.stats.reset()
        self.timestep_index = 0

        # if self.random_place_snake:
        #     idx = np.random.choice(len(RANDOM_BORN))
        #     snake_head = RANDOM_BORN[idx][0]
        #     snake_head = Point(snake_head.y, snake_head.x)
        #     self.snake = Snake(snake_head, length=self.initial_snake_length)
        #     self.snake.direction = RANDOM_BORN[idx][1]
        # else:
        #     self.snake = Snake(self.field.find_snake_head(), length=self.initial_snake_length)
        # self.field.place_snake(self.snake)

        # for agent_index in range(self.num_agents):
        if self.random_place_vehicle:
            for i in range(self.num_initial_vehicles):
                self.new_agent_vehicle()

            for i in range(self.num_initial_pedestrians):
                self.new_agent_pedestrian()
        else:
            self.vehicle = Snake(self.field.find_snake_head(), length=self.initial_snake_length)
            self.field.place_snake(self.vehicle)

        self.current_action = None
        self.punch = False
        self.punch_collaborator = False
        self.punch_wall_pos = [0,0]
        self.punch_wall_pos_collaborator = [0,0]
        self.is_game_over = False
        self.good_miss_ct = 0 
        self.bad_miss_ct = 0 
        self.lava_miss_ct = 0 

        result = TimestepResult(
            observation=self.get_observation(),
            reward=0,
            is_episode_end=self.is_game_over
        )

        self.record_timestep_stats(result)
        if self.collaboration:
            self.record_timestep_stats_collaborator(result)
        return result

    def record_timestep_stats(self, result, agent_mode=0):
        """ Record environment statistics according to the verbosity level. """
        timestamp = time.strftime('%Y%m%d-%H%M%S')

        # Write CSV header for the stats file.
        if self.verbose >= 1 and self.stats_file is None:
            if self.participant is None: self.stats_file = open(f'robotaxi_integration/csv/autocar_{self.field.size}_{timestamp}.csv', 'w')
            else: self.stats_file = open(f'robotaxi_integration/csv/autocar_{self.field.size}_{self.participant}_{timestamp}.csv', 'w')
            stats_csv_header_line = self.stats.to_dataframe()[:0].to_csv(index=None)
            # print(stats_csv_header_line, file=self.stats_file, end='', flush=True)

        # Create a blank debug log file.
        if self.verbose >= 2 and self.debug_file is None:
            if self.participant is None: self.debug_file = open(f'robotaxi_integration/log/autocar_{self.field.size}_{timestamp}.log', 'w')
            else: self.debug_file = open(f'robotaxi_integration/log/autocar_{self.field.size}_{self.participant}_{timestamp}.log', 'w')
            # print('max_step_limit:'+str(self.max_step_limit)+'\n', file=self.debug_file)

        self.stats.record_timestep(self.current_action, result)
        self.stats.timesteps_survived = self.timestep_index

        if self.verbose >= 2:
            # print(str(result) +'punch:' + str(self.punch) +'\npwall_pos:' + str(self.punch_wall_pos) +'\ndirection:(' + str(self.vehicle.direction[0]) + ',' + str(self.vehicle.direction[1]) + ')\nAgent:' + str(agent_mode) + '\n', file=self.debug_file)
            pass
        # Log episode stats if the appropriate verbosity level is set.
        if result.is_episode_end:
            if self.verbose >= 1:
                stats_csv_line = self.stats.to_dataframe().to_csv(header=False, index=None)
                # print(stats_csv_line, file=self.stats_file, end='', flush=True)
            if self.verbose >= 2:
                # print(self.stats, file=self.debug_file)
                pass

    def get_observation(self):
        """ Observe the state of the environment. """
        return np.copy(self.field._cells)

    def get_observation_pedestrians(self):
        """ Observe the state of the environment. """
        return np.copy(self.field_pedestrian._cells)

    def choose_action(self, action):
        """ Choose the action that will be taken at the next timestep. """
        self.current_action = action
        if action == SnakeAction.TURN_LEFT:
            self.vehicle.turn_left()
        elif action == SnakeAction.TURN_RIGHT:
            self.vehicle.turn_right()

    def choose_action_multiagents(self, action, agent, agent_index):
        """ Choose the action that will be taken at the next timestep. """
        self.current_action = action
        if action == SnakeAction.TURN_LEFT:
            agent[agent_index].turn_left()
        elif action == SnakeAction.TURN_RIGHT:
            agent[agent_index].turn_right()

    def timestep_vehicle(self, agent_mode=0, agent_index=0, rwd=0, action=0):
        """ Execute the timestep and return the new observable state. """

        self.timestep_index += 1
        self.punch = False
        reward = rwd
        wall_types = [CellType.WALL, CellType.COLLABORATOR_HEAD]

        old_head = self.vehicles[agent_index].head
        old_direction = self.vehicles[agent_index].direction  # first move one step forward and then turn head
        if action != 0: self.vehicles[agent_index].body[0] += old_direction

        self.choose_action_multiagents(action, self.vehicles, agent_index)

        self.vehicles[agent_index].move()

        self.field.update_snake_footprint(old_head, self.vehicles[agent_index].head)

        # Hit a wall or own body?
        if not self.is_alive_v(self.vehicles[agent_index]):
            # if self.has_hit_wall():
            if self.has_hit_wall_v(self.vehicles[agent_index]):
                self.stats.termination_reason = 'hit_wall'
            # if self.has_hit_own_body():
            if self.has_hit_own_body_v(self.vehicles[agent_index]):
                self.stats.termination_reason = 'hit_own_body'
            self.is_game_over = True

        # Exceeded the limit of moves?
        if self.timestep_index >= self.max_step_limit:
            self.is_game_over = True
            self.stats.termination_reason = 'timestep_limit_exceeded'

        result = TimestepResult(
            observation=self.get_observation(),
            reward=reward,
            is_episode_end=self.is_game_over
        )
        self.record_timestep_stats(result, agent_mode)
        return result

    def timestep_pedestrians(self, agent_index, rwd=0):
        """ Execute the timestep and return the new observable state. """

        self.timestep_index += 1
        wall_types = [CellType.WALL, CellType.COLLABORATOR_HEAD]
        reward = rwd
        old_head = self.pedestrians[agent_index].head
        next_move = self.pedestrians[agent_index].peek_next_move()
        self.pedestrians[agent_index].move()

        self.field_pedestrian.update_snake_footprint(old_head, self.pedestrians[agent_index].head)

        if not self.is_alive_p(self.pedestrians[agent_index]):
            # if self.has_hit_wall():
            if self.has_hit_wall_p(self.pedestrians[agent_index]):
                dx, dy = self.pedestrians[agent_index].direction
                self.pedestrians[agent_index].direction = Point(-1 * dx, -1 * dy)
                # self.pedestrians[agent_index].direction *= -1  # turn back

                backward_point = Point(self.pedestrians[agent_index].last_head[0], self.pedestrians[agent_index].last_head[1])
                self.pedestrians[agent_index].body = deque([backward_point])

        # Exceeded the limit of moves?
        if self.timestep_index >= self.max_step_limit:
            self.is_game_over = True
            self.stats.termination_reason = 'timestep_limit_exceeded'

        result = TimestepResult(
            observation=self.get_observation_pedestrians(),
            reward=reward,
            is_episode_end=self.is_game_over
        )
        self.record_timestep_stats(result, agent_mode=0)
        return result

    def has_hit_wall_v(self, agent=None):
        """ True if the snake has hit a wall, False otherwise. """
        if agent is None: agent = self.vehicle
        return self.field[agent.head] == CellType.WALL

    def has_hit_own_body_v(self, agent=None):
        """ True if the snake has hit its own body, False otherwise. """
        if agent is None: agent = self.vehicle
        return self.field[agent.head] == CellType.COLLABORATOR_BODY

    def is_alive_v(self, agent=None):
        """ True if the snake is still alive, False otherwise. """
        if agent is None: agent = self.vehicle
        return not self.has_hit_wall_v(agent=agent) and not self.has_hit_own_body_v(agent=agent)

    def has_hit_wall_p(self, agent=None):
        """ True if the snake has hit a wall, False otherwise. """
        if agent is None: agent = self.vehicle
        return self.field_pedestrian[agent.head] == CellType.WALL

    def has_hit_own_body_p(self, agent=None):
        """ True if the snake has hit its own body, False otherwise. """
        if agent is None: agent = self.vehicle
        return self.field_pedestrian[agent.head] == CellType.COLLABORATOR_BODY

    def is_alive_p(self, agent=None):
        """ True if the snake is still alive, False otherwise. """
        if agent is None: agent = self.vehicle
        return not self.has_hit_wall_p(agent=agent) and not self.has_hit_own_body_p(agent=agent)


class TimestepResult(object):
    """ Represents the information provided to the agent after each timestep. """

    def __init__(self, observation, reward, is_episode_end):
        self.observation = observation
        self.reward = reward
        self.is_episode_end = is_episode_end

    def __str__(self):
        field_map = '\n'.join([
            ''.join(str(cell) for cell in row)
            for row in self.observation
        ])
        return f'{field_map}\nR={self.reward}\nend={self.is_episode_end}\n'


class EpisodeStatistics(object):
    """ Represents the summary of the agent's performance during the episode. """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Forget all previous statistics and prepare for a new episode. """
        self.timesteps_survived = 0
        self.sum_episode_rewards = 0
        self.good_fruits_eaten = 0
        self.bad_fruits_eaten = 0
        self.lava_crossed = 0
        self.termination_reason = None
        self.action_counter = {
            action: 0
            for action in ALL_SNAKE_ACTIONS
        }


    def record_timestep(self, action, result):
        """ Update the stats based on the current timestep results. """
        self.sum_episode_rewards += result.reward
        if action is not None:
            self.action_counter[action] += 1

    def flatten(self):
        """ Format all episode statistics as a flat object. """
        flat_stats = {
            'timesteps_survived': self.timesteps_survived,
            'sum_episode_rewards': self.sum_episode_rewards,
            'mean_reward': self.sum_episode_rewards / self.timesteps_survived if self.timesteps_survived else None,
            'good_fruits_eaten': self.good_fruits_eaten,
            'bad_fruits_eaten': self.bad_fruits_eaten,
            'lava_crossed': self.lava_crossed,
            'termination_reason': self.termination_reason,
        }
        flat_stats.update({
            f'action_counter_{action}': self.action_counter.get(action, 0)
            for action in ALL_SNAKE_ACTIONS
        })
        return flat_stats

    def to_dataframe(self):
        """ Convert the episode statistics to a Pandas data frame. """
        return pd.DataFrame([self.flatten()])

    def __str__(self):
        return pprint.pformat(self.flatten())
