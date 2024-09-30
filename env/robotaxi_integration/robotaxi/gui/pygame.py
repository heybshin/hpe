import numpy as np
import pygame
import time
import cv2
import os
import threading

from robotaxi_integration.agent import HumanAgent
from robotaxi_integration.gameplay.entities import (CellType, SnakeAction, SnakeDirection, ALL_SNAKE_DIRECTIONS)
from robotaxi_integration.gameplay.environment import PLAY_SOUND, TimestepResult
from robotaxi_integration.utils.next_step_choice import get_next_step_choices_car, get_next_step_choices_pedestrian
import ctypes
from robotaxi_integration.utils.dualsense import DualSense
from pylsl import StreamInfo, StreamOutlet
ctypes.windll.user32.SetProcessDPIAware()

frame_ct = -1


class PyGameGUI:
    """ Provides a Snake GUI powered by Pygame. """

    FPS_LIMIT = 600
    AI_TIMESTEP_DELAY = 5
    HUMAN_TIMESTEP_DELAY = 500

    SNAKE_CONTROL_KEYS = [
        pygame.K_UP,
        pygame.K_LEFT,
        pygame.K_DOWN,
        pygame.K_RIGHT
    ]

    def __init__(self, save_frames=False, field_size=8, test=False):
        # pygame.mixer.pre_init(44100, -16, 2, 32)
        pygame.init()
        pygame.mixer.init()

        pygame.mouse.set_visible(True)
        self.intermediate_frames = 20
        self.punch_sound = pygame.mixer.Sound('sound/punch.wav')
        self.begin_sound = pygame.mixer.Sound('sound/begin.wav')
        self.good_sound = pygame.mixer.Sound('sound/good.wav')
        self.bad_sound = pygame.mixer.Sound('sound/road_block_crash.wav')
        self.very_bad_sound = pygame.mixer.Sound('sound/car_crash.wav')
        self.stuck_sound = pygame.mixer.Sound('sound/woop.wav')
        self.free_sound = pygame.mixer.Sound('sound/restart.wav')
        self.agent = HumanAgent()
        self.collaborating_agent = None
        self.env = None
        self.screen = None
        self.screen_size = 0
        self.fps_clock = None
        self.timestep_watch = Stopwatch()
        self.time_thresh = 50  # threshold for timestep left to render red and flash
        self.frame_num = 0
        self.pause = True
        self.sound_played = True
        self.display_time = True
        self.save_frames = save_frames
        self.timestep_delay = 0
        self.FIELD_SIZE = field_size
        self.CELL_SIZE = 96 * 20 // self.FIELD_SIZE

        self.car_schemes = ["bus", "pickup", "truck", "bulldozer"]
        self.human_schemes = []
        self.selected_icon_scheme = 0  # default
        self.set_icon_scheme_vehicle(self.selected_icon_scheme)
        # self.set_icon_scheme_pedestrian()
        self.selected_icon_scheme_collaborator = 0
        self.set_icon_scheme_collaborator(self.selected_icon_scheme_collaborator)
        self.selected = False

        self.test = test

        self.spawn_icon = pygame.transform.scale(pygame.image.load("icon/wave.png"), (self.CELL_SIZE, self.CELL_SIZE))
        self.wall_icon = pygame.transform.scale(pygame.image.load("icon/forest.png"), (self.CELL_SIZE, self.CELL_SIZE))
        self.good_fruit_icon = pygame.transform.scale(pygame.image.load("icon/man.png"), (self.CELL_SIZE * 2 // 3, self.CELL_SIZE * 2 // 3))
        self.man1_icon = pygame.transform.scale(pygame.image.load("icon/man/man1.png"), (self.CELL_SIZE * 2 // 1.9, self.CELL_SIZE * 2 // 1.9))
        self.man1_icon_west = pygame.transform.rotate(self.man1_icon, 45)
        self.man1_icon_east = pygame.transform.rotate(self.man1_icon, -45)
        self.man1_icon_south = pygame.transform.rotate(self.man1_icon, -135)

        self.cop_icon = pygame.transform.scale(pygame.image.load("icon/man/cop.png"), (self.CELL_SIZE * 2 // 2, self.CELL_SIZE * 2 // 2))
        self.cop_icon_west = pygame.transform.rotate(self.cop_icon, 45)
        self.cop_icon_east = pygame.transform.rotate(self.cop_icon, -45)
        self.cop_icon_south = pygame.transform.rotate(self.cop_icon, -135)

        self.man2_icon = pygame.transform.scale(pygame.image.load("icon/man/man2.png"), (self.CELL_SIZE * 2 // 2, self.CELL_SIZE * 2 // 2))
        self.man2_icon_west = pygame.transform.rotate(self.man2_icon, 45)
        self.man2_icon_east = pygame.transform.rotate(self.man2_icon, -45)
        self.man2_icon_south = pygame.transform.rotate(self.man2_icon, -135)

        self.man3_icon = pygame.transform.scale(pygame.image.load("icon/man/man3.png"), (self.CELL_SIZE * 2 // 2, self.CELL_SIZE * 2 // 2))
        self.man3_icon_west = pygame.transform.rotate(self.man3_icon, 45)
        self.man3_icon_east = pygame.transform.rotate(self.man3_icon, -45)
        self.man3_icon_south = pygame.transform.rotate(self.man3_icon, -135)

        self.woman1_icon = pygame.transform.scale(pygame.image.load("icon/man/woman1.png"), (self.CELL_SIZE * 2 // 2, self.CELL_SIZE * 2 // 2))
        self.woman1_icon_west = pygame.transform.rotate(self.woman1_icon, 45)
        self.woman1_icon_east = pygame.transform.rotate(self.woman1_icon, -45)
        self.woman1_icon_south = pygame.transform.rotate(self.woman1_icon, -135)

        self.woman2_icon = pygame.transform.scale(pygame.image.load("icon/man/woman2.png"), (self.CELL_SIZE * 2 // 2, self.CELL_SIZE * 2 // 2))
        self.woman2_icon_west = pygame.transform.rotate(self.woman2_icon, 45)
        self.woman2_icon_east = pygame.transform.rotate(self.woman2_icon, -45)
        self.woman2_icon_south = pygame.transform.rotate(self.woman2_icon, -135)

        self.bad_fruit_icon = pygame.transform.scale(pygame.image.load("icon/road_block.png"), (self.CELL_SIZE * 2 // 3, self.CELL_SIZE * 2 // 3))
        self.lava_icon = pygame.transform.scale(pygame.image.load("icon/purple_car.png"), (self.CELL_SIZE, self.CELL_SIZE))
        self.small_crash_icon = pygame.transform.scale(pygame.image.load("icon/road_block_broken.png"), (self.CELL_SIZE * 2 // 3, self.CELL_SIZE * 2 // 3))
        self.big_crash_icon = pygame.transform.scale(pygame.image.load("icon/broken_purple_car.png"), (self.CELL_SIZE, self.CELL_SIZE * 2 // 3))
        self.reward_icon = pygame.transform.scale(pygame.image.load("icon/dollar.png"), (self.CELL_SIZE // 3, self.CELL_SIZE // 3))
        self.curr_icon = None
        self.curr_icon_collaborator = None
        self.question1_icon = pygame.transform.scale(pygame.image.load("icon/question1.png"), (self.CELL_SIZE * 2 // 3, self.CELL_SIZE * 2 // 3))
        self.question2_icon = pygame.transform.scale(pygame.image.load("icon/question2.png"), (self.CELL_SIZE * 2 // 3, self.CELL_SIZE * 2 // 3))
        self.question3_icon = pygame.transform.scale(pygame.image.load("icon/question3.png"), (self.CELL_SIZE * 2 // 3, self.CELL_SIZE * 2 // 3))
        self.pit_icon = pygame.transform.scale(pygame.image.load("icon/stopped.png"), (self.CELL_SIZE, self.CELL_SIZE))
        self.stop_icon = pygame.transform.scale(pygame.image.load("icon/stopped.png"), (self.CELL_SIZE // 3, self.CELL_SIZE // 3))
        self.accident_icon = pygame.transform.scale(pygame.image.load("icon/accident.png"), (self.CELL_SIZE * 1.5, self.CELL_SIZE * 1.5))
        # self.head_icon = pygame.transform.scale(pygame.image.load("icon/snake.png"),(self.CELL_SIZE, self.CELL_SIZE))
        # self.body_icon = pygame.transform.scale(pygame.image.load("icon/body.png"),(self.CELL_SIZE, self.CELL_SIZE))
        self.punch_icon = pygame.transform.scale(pygame.image.load("icon/scary_tree.png"), (self.CELL_SIZE, self.CELL_SIZE))

        self.arrow_left_north = pygame.transform.scale(pygame.image.load("icon/Arrows/left_north.png"), (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_right_north = pygame.transform.scale(pygame.image.load("icon/Arrows/right_north.png"), (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_straight_north = pygame.transform.scale(pygame.image.load("icon/Arrows/straight_north.png"), (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_left_south = pygame.transform.scale(pygame.image.load("icon/Arrows/left_south.png"), (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_right_south = pygame.transform.scale(pygame.image.load("icon/Arrows/right_south.png"), (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_straight_south = pygame.transform.scale(pygame.image.load("icon/Arrows/straight_south.png"), (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_left_east = pygame.transform.scale(pygame.image.load("icon/Arrows/left_east.png"), (self.CELL_SIZE, self.CELL_SIZE // 2))
        self.arrow_right_east = pygame.transform.scale(pygame.image.load("icon/Arrows/right_east.png"), (self.CELL_SIZE, self.CELL_SIZE // 2))
        self.arrow_straight_east = pygame.transform.scale(pygame.image.load("icon/Arrows/straight_east.png"), (self.CELL_SIZE, self.CELL_SIZE // 2))
        self.arrow_left_west = pygame.transform.scale(pygame.image.load("icon/Arrows/left_west.png"), (self.CELL_SIZE, self.CELL_SIZE // 2))
        self.arrow_right_west = pygame.transform.scale(pygame.image.load("icon/Arrows/right_west.png"), (self.CELL_SIZE, self.CELL_SIZE // 2))
        self.arrow_straight_west = pygame.transform.scale(pygame.image.load("icon/Arrows/straight_west.png"), (self.CELL_SIZE, self.CELL_SIZE // 2))
        self.arrow_all_north = pygame.transform.scale(pygame.image.load("icon/Arrows/all_north.png"), (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_all_south = pygame.transform.scale(pygame.image.load("icon/Arrows/all_south.png"), (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_all_east = pygame.transform.scale(pygame.image.load("icon/Arrows/all_east.png"), (self.CELL_SIZE, self.CELL_SIZE // 2))
        self.arrow_all_west = pygame.transform.scale(pygame.image.load("icon/Arrows/all_west.png"), (self.CELL_SIZE, self.CELL_SIZE // 2))
        self.arrow_noright_north = pygame.transform.scale(pygame.image.load("icon/Arrows/noright_north.png"), (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_noright_south = pygame.transform.scale(pygame.image.load("icon/Arrows/noright_south.png"), (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_noright_east = pygame.transform.scale(pygame.image.load("icon/Arrows/noright_east.png"), (self.CELL_SIZE, self.CELL_SIZE // 1.3))
        self.arrow_noright_west = pygame.transform.scale(pygame.image.load("icon/Arrows/noright_west.png"), (self.CELL_SIZE, self.CELL_SIZE // 2))
        self.arrow_rightonly_south = pygame.transform.flip(self.arrow_left_south, 1, 0)
        self.arrow_uturn_north = pygame.transform.scale(pygame.image.load("icon/Arrows/uturn.png"), (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_uturn_west = pygame.transform.rotate(self.arrow_uturn_north, 90)

        self.house_icon = pygame.transform.scale(pygame.image.load("icon/Mine/house.png"), (self.CELL_SIZE * 1.5, self.CELL_SIZE * 1.5))
        self.school_icon = pygame.transform.scale(pygame.image.load("icon/Mine/school.png"), (self.CELL_SIZE * 1.5, self.CELL_SIZE * 1.5))
        self.park_icon = pygame.transform.scale(pygame.image.load("icon/Mine/park.png"), (self.CELL_SIZE * 2, self.CELL_SIZE * 2))
        self.triangle_icon = pygame.transform.scale(pygame.image.load("icon/Mine/ps_triangle.png"), (self.CELL_SIZE / 1.38, self.CELL_SIZE / 1.38))
        self.circle_icon = pygame.transform.scale(pygame.image.load("icon/Mine/ps_circle.png"), (self.CELL_SIZE / 1.38, self.CELL_SIZE / 1.38))
        self.square_icon = pygame.transform.scale(pygame.image.load("icon/Mine/ps_square.png"), (self.CELL_SIZE / 1.44, self.CELL_SIZE / 1.44))
        self.cross_icon = pygame.transform.scale(pygame.image.load("icon/Mine/ps_cross.png"), (self.CELL_SIZE / 1.44, self.CELL_SIZE / 1.44))

        self.two_all_red_north = pygame.transform.scale(pygame.image.load("icon/trafficlight/2_all_red.png"), (self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        self.two_left_red_north = pygame.transform.scale(pygame.image.load("icon/trafficlight/2_left_red.png"), (self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        self.two_straight_red_north = pygame.transform.scale(pygame.image.load("icon/trafficlight/2_straight_red.png"), (self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        self.two_left_red_off_north = pygame.transform.scale(pygame.image.load("icon/trafficlight/2_left_red_off.png"), (self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        self.two_straight_red_off_north = pygame.transform.scale(pygame.image.load("icon/trafficlight/2_straight_red_off.png"), (self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        # self.two_off_north = pygame.transform.scale(pygame.image.load("icon/trafficlight/2_off.png"),(self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        # self.three_all_red_north = pygame.transform.scale(pygame.image.load("icon/trafficlight/3_all_red.png"),(self.CELL_SIZE // 2, self.CELL_SIZE // 2))
        # self.three_off_north = pygame.transform.scale(pygame.image.load("icon/trafficlight/3_off.png"),(self.CELL_SIZE // 2, self.CELL_SIZE // 2))
        self.single_red = pygame.transform.scale(pygame.image.load("icon/trafficlight/single_red.png"), (self.CELL_SIZE // 1.5, self.CELL_SIZE // 1.5))
        self.single_green = pygame.transform.scale(pygame.image.load("icon/trafficlight/single_green.png"), (self.CELL_SIZE // 1.5, self.CELL_SIZE // 1.5))
        self.single_off = pygame.transform.scale(pygame.image.load("icon/trafficlight/single_off.png"), (self.CELL_SIZE // 1.5, self.CELL_SIZE // 1.5))

        self.pedestrian_red_north = pygame.transform.scale(pygame.image.load("icon/trafficlight/stop.png"), (self.CELL_SIZE // 1.7, self.CELL_SIZE // 1.7))
        self.pedestrian_green_north = pygame.transform.scale(pygame.image.load("icon/trafficlight/pass.png"), (self.CELL_SIZE // 1.7, self.CELL_SIZE // 1.7))
        self.pedestrian_green_off = pygame.transform.scale(pygame.image.load("icon/trafficlight/pass_off.png"), (self.CELL_SIZE // 1.7, self.CELL_SIZE // 1.7))

        self.uturn_all_red = pygame.transform.scale(pygame.image.load("icon/trafficlight/uturn_all_red.png"), (self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        self.uturn_all_red_west = pygame.transform.rotate(self.uturn_all_red, 90)
        self.uturn_uturn_red = pygame.transform.scale(pygame.image.load("icon/trafficlight/uturn_uturn_red.png"), (self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        self.uturn_uturn_red_west = pygame.transform.rotate(self.uturn_uturn_red, 90)
        self.uturn_uturn_red_off = pygame.transform.scale(pygame.image.load("icon/trafficlight/uturn_uturn_red_off.png"), (self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        self.uturn_uturn_red_off_west = pygame.transform.rotate(self.uturn_uturn_red_off, 90)
        self.uturn_straight_red = pygame.transform.scale(pygame.image.load("icon/trafficlight/uturn_straight_red.png"), (self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        self.uturn_straight_red_west = pygame.transform.rotate(self.uturn_straight_red, 90)
        self.uturn_straight_red_off = pygame.transform.scale(pygame.image.load("icon/trafficlight/uturn_straight_red_off.png"), (self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        self.uturn_straight_red_off_west = pygame.transform.rotate(self.uturn_straight_red_off, 90)
        # self.pedestrian_red_south = pygame.transform.rotate(self.pedestrian_red_north, 180)
        # self.pedestrian_red_east = pygame.transform.rotate(self.pedestrian_red_north, -90)
        # self.pedestrian_red_west = pygame.transform.rotate(self.pedestrian_red_north, 90)
        # self.pedestrian_green_south = pygame.transform.rotate(self.pedestrian_green_north, 180)
        # self.pedestrian_green_east = pygame.transform.rotate(self.pedestrian_green_north, -90)
        # self.pedestrian_green_west = pygame.transform.rotate(self.pedestrian_green_north, 90)

        self.two_all_red_west = pygame.transform.rotate(self.two_all_red_north, 90)
        self.two_all_red_east = pygame.transform.rotate(self.two_all_red_north, -90)
        self.two_all_red_south = pygame.transform.rotate(self.two_all_red_north, 180)
        self.two_left_red_west = pygame.transform.rotate(self.two_left_red_north, 90)
        self.two_left_red_east = pygame.transform.rotate(self.two_left_red_north, -90)
        self.two_left_red_south = pygame.transform.rotate(self.two_left_red_north, 180)
        self.two_straight_red_west = pygame.transform.rotate(self.two_straight_red_north, 90)
        self.two_straight_red_east = pygame.transform.rotate(self.two_straight_red_north, -90)
        self.two_straight_red_south = pygame.transform.rotate(self.two_straight_red_north, 180)
        self.two_left_red_off_west = pygame.transform.rotate(self.two_left_red_off_north, 90)
        self.two_left_red_off_east = pygame.transform.rotate(self.two_left_red_off_north, -90)
        self.two_left_red_off_south = pygame.transform.rotate(self.two_left_red_off_north, 180)
        self.two_straight_red_off_west = pygame.transform.rotate(self.two_straight_red_off_north, 90)
        self.two_straight_red_off_east = pygame.transform.rotate(self.two_straight_red_off_north, -90)
        self.two_straight_red_off_south = pygame.transform.rotate(self.two_straight_red_off_north, 180)

        self.traffic_light_status = initialize_traffic_lights_status()

        self.action_car_north, self.action_car_south, self.action_car_east, self.action_car_west = get_next_step_choices_car()
        self.action_pedestrian_north, self.action_pedestrian_south, self.action_pedestrian_east, self.action_pedestrian_west = get_next_step_choices_pedestrian()

        self.change_lane_back_vehicles = [None]
        self.curr_head = [0, 0]
        self.last_head = [0, 0]
        self.curr_head_collaborator = [0, 0]
        self.last_head_collaborator = [0, 0]
        self.internal_padding = self.CELL_SIZE // 5
        self.text_font = pygame.font.Font("fonts/gyparody_hv.ttf", int(23 * (self.CELL_SIZE / 40.0)))
        self.num_font = pygame.font.Font("fonts/gyparody_tf.ttf", int(36 * (self.CELL_SIZE / 40.0)))
        self.marker_font = pygame.font.Font("fonts/OpenSans-Bold.ttf", int(12 * (self.CELL_SIZE / 40.0)))

        self.warning_icon = pygame.transform.scale(pygame.image.load("icon/question1.png"), (self.CELL_SIZE, self.CELL_SIZE))
        self.pause_time = 0
        self.violation_scoreboard = 0
        self.accident_scoreboard = 0
        self.visual_feedback_v = None
        self.visual_feedback_p = None

        self.callbacks = {
            'triangle_pressed': self.triangle_pressed_callback,
            'circle_pressed': self.circle_pressed_callback,
            'square_pressed': self.square_pressed_callback,
            'cross_pressed': self.cross_pressed_callback,
            'ps_pressed': self.ps_pressed_callback
        }

        self.dualsense = DualSense(callbacks=self.callbacks)
        self.event_lock = False  # lock to prevent multiple violations from being triggered at the same time
        self.button_pressed = False
        self.button_pressed_value = None

        info = StreamInfo("Robotaxi", "Markers", 1, 0, "string", "myuid000")

        # next make an outlet
        self.outlet = StreamOutlet(info)

        pygame.display.set_caption('Robotaxi')

    def triangle_pressed_callback(self):
        # print("Triangle button was pressed!")
        if self.running and (self.accident_type is not None) and (not self.button_pressed):
            self.outlet.push_sample(["tri"])
            if self.accident_type == "vehicle_violation":
                result = TimestepResult(observation=self.env.get_observation(), reward=10, is_episode_end=False)
                self.env.record_timestep_stats(result, 0)
                self.button_pressed_value = 10

            else:
                result = TimestepResult(observation=self.env.get_observation(), reward=-10, is_episode_end=False)
                self.env.record_timestep_stats(result, 0)
                self.button_pressed_value = -10
            self.button_pressed = True
        elif self.accident_type is None:
            print("No violation to report")

    def circle_pressed_callback(self):
        # print("Triangle button was pressed!")
        if self.running and (self.accident_type is not None) and (not self.button_pressed):
            self.outlet.push_sample(["cir"])
            if self.accident_type == "pedestrian_violation":
                result = TimestepResult(observation=self.env.get_observation(), reward=10, is_episode_end=False)
                self.env.record_timestep_stats(result, 0)
                self.button_pressed_value = 10
            else:
                result = TimestepResult(observation=self.env.get_observation(), reward=-10, is_episode_end=False)
                self.env.record_timestep_stats(result, 0)
                self.button_pressed_value = -10
            self.button_pressed = True
        elif self.accident_type is None:
            print("No violation to report")

    def square_pressed_callback(self):
        # print("Triangle button was pressed!")
        if self.running and (self.accident_type is not None) and (not self.button_pressed):
            self.outlet.push_sample(["squ"])
            if self.accident_type == "vehicle_accident":
                result = TimestepResult(observation=self.env.get_observation(), reward=20, is_episode_end=False)
                self.env.record_timestep_stats(result, 0)
                self.button_pressed_value = 20
            else:
                result = TimestepResult(observation=self.env.get_observation(), reward=-10, is_episode_end=False)
                self.env.record_timestep_stats(result, 0)
                self.button_pressed_value = -10
            self.button_pressed = True
        elif self.accident_type is None:
            print("No violation to report")

    def cross_pressed_callback(self):
        # print("Triangle button was pressed!")
        if self.running and (self.accident_type is not None) and (not self.button_pressed):
            self.outlet.push_sample(["cro"])
            if self.accident_type == "pedestrian_accident":
                result = TimestepResult(observation=self.env.get_observation(), reward=20, is_episode_end=False)
                self.env.record_timestep_stats(result, 0)
                self.button_pressed_value = 20
            else:
                result = TimestepResult(observation=self.env.get_observation(), reward=-10, is_episode_end=False)
                self.env.record_timestep_stats(result, 0)
                self.button_pressed_value = -10
            self.button_pressed = True
        elif self.accident_type is None:
            print("No violation to report")

    def ps_pressed_callback(self):
        pass

    def set_icon_scheme_vehicle(self, idx):

        # scheme = self.car_schemes[idx]

        self.south, self.north, self.east, self.west = [], [], [], []
        for i in range(len(self.car_schemes)):
            scheme = self.car_schemes[i]
            self.south.append(pygame.transform.scale(pygame.image.load("icon/" + scheme + "_south.png"),
                                                     (self.CELL_SIZE, self.CELL_SIZE - 5)))
            self.north.append(pygame.transform.scale(pygame.image.load("icon/" + scheme + "_north.png"),
                                                     (self.CELL_SIZE, self.CELL_SIZE - 5)))
            self.east.append(pygame.transform.scale(pygame.image.load("icon/" + scheme + "_east.png"),
                                                    (self.CELL_SIZE, self.CELL_SIZE - 5)))
            self.west.append(
                pygame.transform.flip(pygame.transform.scale(pygame.image.load("icon/" + scheme + "_east.png"),
                                                             (self.CELL_SIZE, self.CELL_SIZE - 5)), 1, 0))

    def set_icon_scheme_pedestrian(self, idx, direction=None):

        self.human_schemes = ['man1', 'man2', 'man3', 'woman1', 'woman2', 'cop']

        if 0 <= idx < len(self.human_schemes):
            if direction in [SnakeDirection.NORTH, SnakeDirection.WEST]:
                attribute_name = f"{self.human_schemes[idx]}_icon_west"
            elif direction == SnakeDirection.EAST:
                attribute_name = f"{self.human_schemes[idx]}_icon_east"
            elif direction == SnakeDirection.SOUTH:
                attribute_name = f"{self.human_schemes[idx]}_icon_south"
            elif direction is None:
                attribute_name = f"{self.human_schemes[idx]}_icon"

            return getattr(self, attribute_name)
        else:
            raise IndexError("Index out of range for human_schemes list")

    def set_icon_scheme_collaborator(self, idx):
        scheme = self.car_schemes[idx]
        self.south_collaborator = pygame.transform.scale(pygame.image.load("icon/" + scheme + "_south.png"),
                                                         (self.CELL_SIZE, self.CELL_SIZE - 5))
        self.north_collaborator = pygame.transform.scale(pygame.image.load("icon/" + scheme + "_north.png"),
                                                         (self.CELL_SIZE, self.CELL_SIZE - 5))
        self.east_collaborator = pygame.transform.scale(pygame.image.load("icon/" + scheme + "_east.png"),
                                                        (self.CELL_SIZE, self.CELL_SIZE - 5))
        self.west_collaborator = pygame.transform.flip(self.east_collaborator, 1, 0)

    def set_fixed_icon_scheme_collaborator(self):
        scheme = 'bulldozer'
        self.south_collaborator = pygame.transform.scale(pygame.image.load("icon/" + scheme + "_south.png"),
                                                         (self.CELL_SIZE, self.CELL_SIZE - 5))
        self.north_collaborator = pygame.transform.scale(pygame.image.load("icon/" + scheme + "_north.png"),
                                                         (self.CELL_SIZE, self.CELL_SIZE - 5))
        self.east_collaborator = pygame.transform.scale(pygame.image.load("icon/" + scheme + "_east.png"),
                                                        (self.CELL_SIZE, self.CELL_SIZE - 5))
        self.west_collaborator = pygame.transform.flip(self.east_collaborator, 1, 0)

    def load_environment(self, environment):
        """ Load the RL environment into the GUI. """
        self.env = environment
        self.screen_size = ((self.env.field.size + 6) * self.CELL_SIZE, self.env.field.size * self.CELL_SIZE)
        # flags = pygame.OPENGL | pygame.FULLSCREEN

        WIDTH, HEIGHT = self.screen_size  # Width and height of the window
        CELL_SIZE = self.CELL_SIZE  # Size of each cell
        GRID_COLOR = (0, 0, 0)  # Color of the grid lines (black)
        GRID_LINE_WIDTH = 5  # Thickness of the grid lines

        self.surface = self.create_grid_surface(WIDTH, HEIGHT, CELL_SIZE, GRID_COLOR, GRID_LINE_WIDTH)
        self.screen = pygame.display.set_mode(self.screen_size, pygame.RESIZABLE)
        self.screen.fill(Colors.SCREEN_BACKGROUND)

        self.total_score = 0.0

    def create_grid_surface(self, width, height, cell_size, grid_color, grid_line_width):
        # Create a surface for the grid
        grid_surface = pygame.Surface((width, height))  # , flags=pygame.FULLSCREEN|pygame.HWSURFACE)
        grid_surface.fill(Colors.SCREEN_BACKGROUND)  # Fill with white background
        zebra_crossing_color = (255, 255, 255)
        lane_color_w = (255, 255, 255)
        lane_color_orange = (255, 165, 0)

        create_road = True
        create_sidewalk = True
        create_zebra_crossing = True
        create_lane = True
        create_arrow = True
        create_building = True
        # first dimension control column
        pygame.draw.line(grid_surface, grid_color, (cell_size, cell_size), (cell_size, cell_size * 22), grid_line_width)
        pygame.draw.line(grid_surface, grid_color, (cell_size * 22, cell_size), (cell_size * 22, cell_size * 22),
                         grid_line_width)
        pygame.draw.line(grid_surface, grid_color, (cell_size, cell_size), (cell_size * 22, cell_size), grid_line_width)
        pygame.draw.line(grid_surface, grid_color, (cell_size, cell_size * 22), (cell_size * 22, cell_size * 22),
                         grid_line_width)

        if create_road:
            pygame.draw.line(grid_surface, grid_color, (cell_size * 1, cell_size * 3), (cell_size * 4, cell_size * 3),
                             grid_line_width)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 11, cell_size * 3), (cell_size * 14, cell_size * 3),
                             grid_line_width)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 18, cell_size * 3), (cell_size * 22, cell_size * 3),
                             grid_line_width)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 1, cell_size * 10), (cell_size * 4, cell_size * 10),
                             grid_line_width)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 11, cell_size * 10),
                             (cell_size * 14, cell_size * 10), grid_line_width)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 18, cell_size * 10),
                             (cell_size * 22, cell_size * 10), grid_line_width)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 1, cell_size * 15), (cell_size * 4, cell_size * 15),
                             grid_line_width)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 11, cell_size * 15),
                             (cell_size * 14, cell_size * 15), grid_line_width)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 18, cell_size * 15),
                             (cell_size * 22, cell_size * 15), grid_line_width)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 1, cell_size * 20),
                             (cell_size * 14, cell_size * 20), grid_line_width)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 18, cell_size * 20),
                             (cell_size * 22, cell_size * 20), grid_line_width)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 4, cell_size * 1), (cell_size * 4, cell_size * 3),
                             grid_line_width)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 4, cell_size * 10), (cell_size * 4, cell_size * 15),
                             grid_line_width)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 11, cell_size * 1), (cell_size * 11, cell_size * 3),
                             grid_line_width)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 11, cell_size * 10),
                             (cell_size * 11, cell_size * 15), grid_line_width)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 14, cell_size * 1), (cell_size * 14, cell_size * 3),
                             grid_line_width)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 14, cell_size * 10),
                             (cell_size * 14, cell_size * 15), grid_line_width)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 14, cell_size * 20),
                             (cell_size * 14, cell_size * 22), grid_line_width)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 18, cell_size * 1), (cell_size * 18, cell_size * 3),
                             grid_line_width)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 18, cell_size * 10),
                             (cell_size * 18, cell_size * 15), grid_line_width)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 18, cell_size * 20),
                             (cell_size * 18, cell_size * 22), grid_line_width)
        if create_sidewalk:
            pygame.draw.line(grid_surface, grid_color, (cell_size * 1, cell_size * 4), (cell_size * 5, cell_size * 4),
                             grid_line_width - 2)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 10, cell_size * 4), (cell_size * 15, cell_size * 4),
                             grid_line_width - 2)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 17, cell_size * 4), (cell_size * 22, cell_size * 4),
                             grid_line_width - 2)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 1, cell_size * 9), (cell_size * 5, cell_size * 9),
                             grid_line_width - 2)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 10, cell_size * 9), (cell_size * 15, cell_size * 9),
                             grid_line_width - 2)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 17, cell_size * 9), (cell_size * 22, cell_size * 9),
                             grid_line_width - 2)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 1, cell_size * 16), (cell_size * 5, cell_size * 16),
                             grid_line_width - 2)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 10, cell_size * 16),
                             (cell_size * 15, cell_size * 16), grid_line_width - 2)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 17, cell_size * 16),
                             (cell_size * 22, cell_size * 16), grid_line_width - 2)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 1, cell_size * 19),
                             (cell_size * 15, cell_size * 19), grid_line_width - 2)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 17, cell_size * 19),
                             (cell_size * 22, cell_size * 19), grid_line_width - 2)

            pygame.draw.line(grid_surface, grid_color, (cell_size * 5, cell_size * 1), (cell_size * 5, cell_size * 4),
                             grid_line_width - 2)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 5, cell_size * 9), (cell_size * 5, cell_size * 16),
                             grid_line_width - 2)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 10, cell_size * 1), (cell_size * 10, cell_size * 4),
                             grid_line_width - 2)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 10, cell_size * 9),
                             (cell_size * 10, cell_size * 16), grid_line_width - 2)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 15, cell_size * 1), (cell_size * 15, cell_size * 4),
                             grid_line_width - 2)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 15, cell_size * 9),
                             (cell_size * 15, cell_size * 16), grid_line_width - 2)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 15, cell_size * 19),
                             (cell_size * 15, cell_size * 22), grid_line_width - 2)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 17, cell_size * 1), (cell_size * 17, cell_size * 4),
                             grid_line_width - 2)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 17, cell_size * 9),
                             (cell_size * 17, cell_size * 16), grid_line_width - 2)
            pygame.draw.line(grid_surface, grid_color, (cell_size * 17, cell_size * 19),
                             (cell_size * 17, cell_size * 22), grid_line_width - 2)
        if create_zebra_crossing:
            for idx in range(30, 400, 50):
                pygame.draw.line(grid_surface, zebra_crossing_color, (cell_size * 4 + 10, cell_size * 4 + idx),
                                 (cell_size * 5 - 10, cell_size * 4 + idx), grid_line_width + 20)
                pygame.draw.line(grid_surface, zebra_crossing_color, (cell_size * 10 + 10, cell_size * 4 + idx),
                                 (cell_size * 11 - 10, cell_size * 4 + idx), grid_line_width + 20)
                pygame.draw.line(grid_surface, zebra_crossing_color, (cell_size * 14 + 10, cell_size * 4 + idx),
                                 (cell_size * 15 - 10, cell_size * 4 + idx), grid_line_width + 20)
                pygame.draw.line(grid_surface, zebra_crossing_color, (cell_size * 17 + 10, cell_size * 4 + idx),
                                 (cell_size * 18 - 10, cell_size * 4 + idx), grid_line_width + 20)

                if idx < 210:
                    pygame.draw.line(grid_surface, zebra_crossing_color,
                                     (cell_size * 4 + 10, cell_size * 16 + idx + 15),
                                     (cell_size * 5 - 10, cell_size * 16 + idx + 15), grid_line_width + 20)
                    pygame.draw.line(grid_surface, zebra_crossing_color,
                                     (cell_size * 10 + 10, cell_size * 16 + idx + 15),
                                     (cell_size * 11 - 10, cell_size * 16 + idx + 15), grid_line_width + 20)
                    pygame.draw.line(grid_surface, zebra_crossing_color,
                                     (cell_size * 14 + 10, cell_size * 16 + idx + 15),
                                     (cell_size * 15 - 10, cell_size * 16 + idx + 15), grid_line_width + 20)
                    pygame.draw.line(grid_surface, zebra_crossing_color,
                                     (cell_size * 17 + 10, cell_size * 16 + idx + 15),
                                     (cell_size * 18 - 10, cell_size * 16 + idx + 15), grid_line_width + 20)

                pygame.draw.line(grid_surface, zebra_crossing_color, (cell_size * 5 + idx, cell_size * 3 + 15),
                                 (cell_size * 5 + idx, cell_size * 4 - 15), grid_line_width + 20)
                pygame.draw.line(grid_surface, zebra_crossing_color, (cell_size * 5 + idx, cell_size * 9 + 15),
                                 (cell_size * 5 + idx, cell_size * 10 - 15), grid_line_width + 20)
                pygame.draw.line(grid_surface, zebra_crossing_color, (cell_size * 5 + idx, cell_size * 15 + 15),
                                 (cell_size * 5 + idx, cell_size * 16 - 15), grid_line_width + 20)

                if idx < 150:
                    pygame.draw.line(grid_surface, zebra_crossing_color, (cell_size * 15 + idx, cell_size * 3 + 15),
                                     (cell_size * 15 + idx, cell_size * 4 - 15), grid_line_width + 20)
                    pygame.draw.line(grid_surface, zebra_crossing_color, (cell_size * 15 + idx, cell_size * 9 + 15),
                                     (cell_size * 15 + idx, cell_size * 10 - 15), grid_line_width + 20)
                    pygame.draw.line(grid_surface, zebra_crossing_color, (cell_size * 15 + idx, cell_size * 15 + 15),
                                     (cell_size * 15 + idx, cell_size * 16 - 15), grid_line_width + 20)
                    pygame.draw.line(grid_surface, zebra_crossing_color, (cell_size * 15 + idx, cell_size * 19 + 15),
                                     (cell_size * 15 + idx, cell_size * 20 - 15), grid_line_width + 20)
        if create_lane:
            # Row 4-8
            pygame.draw.line(grid_surface, lane_color_w, (cell_size + 4, cell_size * 4 + 10),
                             (cell_size * 4 - 10, cell_size * 4 + 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_w, (cell_size + 4, cell_size * 9 - 10),
                             (cell_size * 4 - 10, cell_size * 9 - 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_orange, (cell_size + 4, cell_size * 6 + 5),
                             (cell_size * 4 - 10, cell_size * 6 + 5), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_orange, (cell_size + 4, cell_size * 6 - 5),
                             (cell_size * 4 - 10, cell_size * 6 - 5), grid_line_width)

            pygame.draw.line(grid_surface, lane_color_w, (cell_size * 11 + 4, cell_size * 4 + 10),
                             (cell_size * 14 - 10, cell_size * 4 + 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_w, (cell_size * 11 + 4, cell_size * 9 - 10),
                             (cell_size * 14 - 10, cell_size * 9 - 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_orange, (cell_size * 11 + 4, cell_size * 7 + 5),
                             (cell_size * 14 - 10, cell_size * 7 + 5), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_orange, (cell_size * 11 + 4, cell_size * 7 - 5),
                             (cell_size * 14 - 10, cell_size * 7 - 5), grid_line_width)

            pygame.draw.line(grid_surface, lane_color_w, (cell_size * 18 + 4, cell_size * 4 + 10),
                             (cell_size * 22 - 4, cell_size * 4 + 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_w, (cell_size * 18 + 4, cell_size * 9 - 10),
                             (cell_size * 22 - 4, cell_size * 9 - 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_orange, (cell_size * 18 + 4, cell_size * 7 + 5),
                             (cell_size * 22 - 4, cell_size * 7 + 5), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_orange, (cell_size * 18 + 4, cell_size * 7 - 5),
                             (cell_size * 22 - 4, cell_size * 7 - 5), grid_line_width)

            for idx in range(20, 300, 80):
                if idx < 240:
                    pygame.draw.line(grid_surface, lane_color_w, (cell_size + idx, cell_size * 5),
                                     (cell_size + idx + 40, cell_size * 5), grid_line_width - 2)
                    pygame.draw.line(grid_surface, lane_color_w, (cell_size + idx, cell_size * 7),
                                     (cell_size + idx + 40, cell_size * 7), grid_line_width - 2)
                    pygame.draw.line(grid_surface, lane_color_w, (cell_size + idx, cell_size * 8),
                                     (cell_size + idx + 40, cell_size * 8), grid_line_width - 2)

                    pygame.draw.line(grid_surface, lane_color_w, (cell_size * 11 + idx, cell_size * 5),
                                     (cell_size * 11 + idx + 40, cell_size * 5), grid_line_width - 2)
                    pygame.draw.line(grid_surface, lane_color_w, (cell_size * 11 + idx, cell_size * 6),
                                     (cell_size * 11 + idx + 40, cell_size * 6), grid_line_width - 2)
                    pygame.draw.line(grid_surface, lane_color_w, (cell_size * 11 + idx, cell_size * 8),
                                     (cell_size * 11 + idx + 40, cell_size * 8), grid_line_width - 2)

                pygame.draw.line(grid_surface, lane_color_w, (cell_size * 18 + idx, cell_size * 5),
                                 (cell_size * 18 + idx + 40, cell_size * 5), grid_line_width - 2)
                pygame.draw.line(grid_surface, lane_color_w, (cell_size * 18 + idx, cell_size * 6),
                                 (cell_size * 18 + idx + 40, cell_size * 6), grid_line_width - 2)
                pygame.draw.line(grid_surface, lane_color_w, (cell_size * 18 + idx, cell_size * 8),
                                 (cell_size * 18 + idx + 40, cell_size * 8), grid_line_width - 2)

            # Row 17-20
            pygame.draw.line(grid_surface, lane_color_w, (cell_size + 4, cell_size * 16 + 10),
                             (cell_size * 4 - 10, cell_size * 16 + 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_w, (cell_size + 4, cell_size * 19 - 10),
                             (cell_size * 4 - 10, cell_size * 19 - 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_orange, (cell_size + 4, cell_size * 17 + 5),
                             (cell_size * 4 - 10, cell_size * 17 + 5), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_orange, (cell_size + 4, cell_size * 17 - 5),
                             (cell_size * 4 - 10, cell_size * 17 - 5), grid_line_width)

            pygame.draw.line(grid_surface, lane_color_w, (cell_size * 11 + 4, cell_size * 16 + 10),
                             (cell_size * 14 - 10, cell_size * 16 + 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_w, (cell_size * 11 + 4, cell_size * 19 - 10),
                             (cell_size * 14 - 10, cell_size * 19 - 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_orange, (cell_size * 11 + 4, cell_size * 18 + 5),
                             (cell_size * 14 - 10, cell_size * 18 + 5), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_orange, (cell_size * 11 + 4, cell_size * 18 - 5),
                             (cell_size * 14 - 10, cell_size * 18 - 5), grid_line_width)

            pygame.draw.line(grid_surface, lane_color_w, (cell_size * 18 + 4, cell_size * 16 + 10),
                             (cell_size * 22 - 4, cell_size * 16 + 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_w, (cell_size * 18 + 4, cell_size * 19 - 10),
                             (cell_size * 22 - 4, cell_size * 19 - 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_orange, (cell_size * 18 + 4, cell_size * 18 + 5),
                             (cell_size * 22 - 4, cell_size * 18 + 5), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_orange, (cell_size * 18 + 4, cell_size * 18 - 5),
                             (cell_size * 22 - 4, cell_size * 18 - 5), grid_line_width)

            for idx in range(20, 300, 80):
                if idx < 240:
                    pygame.draw.line(grid_surface, lane_color_w, (cell_size + idx, cell_size * 18),
                                     (cell_size + idx + 40, cell_size * 18), grid_line_width - 2)
                    pygame.draw.line(grid_surface, lane_color_w, (cell_size * 11 + idx, cell_size * 17),
                                     (cell_size * 11 + idx + 40, cell_size * 17), grid_line_width - 2)
                pygame.draw.line(grid_surface, lane_color_w, (cell_size * 18 + idx, cell_size * 17),
                                 (cell_size * 18 + idx + 40, cell_size * 17), grid_line_width - 2)

            # Column 6-10
            pygame.draw.line(grid_surface, lane_color_w, (cell_size * 5 + 7, cell_size + 4),
                             (cell_size * 5 + 7, cell_size * 3 - 10), grid_line_width - 2)
            pygame.draw.line(grid_surface, lane_color_w, (cell_size * 5 + 7, cell_size * 10 + 4),
                             (cell_size * 5 + 7, cell_size * 15 - 10), grid_line_width - 2)
            pygame.draw.line(grid_surface, lane_color_w, (cell_size * 10 - 7, cell_size + 4),
                             (cell_size * 10 - 7, cell_size * 3 - 10), grid_line_width - 2)
            pygame.draw.line(grid_surface, lane_color_w, (cell_size * 10 - 7, cell_size * 10 + 4),
                             (cell_size * 10 - 7, cell_size * 15 - 10), grid_line_width - 2)
            pygame.draw.line(grid_surface, lane_color_orange, (cell_size * 8 + 5, cell_size + 4),
                             (cell_size * 8 + 5, cell_size * 3 - 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_orange, (cell_size * 8 - 5, cell_size + 4),
                             (cell_size * 8 - 5, cell_size * 3 - 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_orange, (cell_size * 7 + 5, cell_size * 10 + 4),
                             (cell_size * 7 + 5, cell_size * 15 - 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_orange, (cell_size * 7 - 5, cell_size * 10 + 4),
                             (cell_size * 7 - 5, cell_size * 15 - 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_orange, (cell_size * 16 - 5, cell_size * 20 + 4),
                             (cell_size * 16 - 5, cell_size * 22 - 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_orange, (cell_size * 16 + 5, cell_size * 20 + 4),
                             (cell_size * 16 + 5, cell_size * 22 - 10), grid_line_width)

            for idx in range(20, 400, 80):
                if idx < 180:
                    pygame.draw.line(grid_surface, lane_color_w, (cell_size * 6, cell_size + idx),
                                     (cell_size * 6, cell_size + idx + 40), grid_line_width - 2)
                    pygame.draw.line(grid_surface, lane_color_w, (cell_size * 7, cell_size + idx),
                                     (cell_size * 7, cell_size + idx + 40), grid_line_width - 2)
                    pygame.draw.line(grid_surface, lane_color_w, (cell_size * 9, cell_size + idx),
                                     (cell_size * 9, cell_size + idx + 40), grid_line_width - 2)

                pygame.draw.line(grid_surface, lane_color_w, (cell_size * 6, cell_size * 10 + idx),
                                 (cell_size * 6, cell_size * 10 + idx + 40), grid_line_width - 2)
                pygame.draw.line(grid_surface, lane_color_w, (cell_size * 8, cell_size * 10 + idx),
                                 (cell_size * 8, cell_size * 10 + idx + 40), grid_line_width - 2)
                pygame.draw.line(grid_surface, lane_color_w, (cell_size * 9, cell_size * 10 + idx),
                                 (cell_size * 9, cell_size * 10 + idx + 40), grid_line_width - 2)

            # Column 16-19
            pygame.draw.line(grid_surface, lane_color_w, (cell_size * 15 + 7, cell_size + 4),
                             (cell_size * 15 + 7, cell_size * 3 - 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_w, (cell_size * 17 - 7, cell_size + 4),
                             (cell_size * 17 - 7, cell_size * 3 - 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_w, (cell_size * 15 + 7, cell_size * 10 + 4),
                             (cell_size * 15 + 7, cell_size * 15 - 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_w, (cell_size * 17 - 7, cell_size * 10 + 4),
                             (cell_size * 17 - 7, cell_size * 15 - 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_w, (cell_size * 15 + 7, cell_size * 20 + 4),
                             (cell_size * 15 + 7, cell_size * 22 - 10), grid_line_width)
            pygame.draw.line(grid_surface, lane_color_w, (cell_size * 17 - 7, cell_size * 20 + 4),
                             (cell_size * 17 - 7, cell_size * 22 - 10), grid_line_width)
            for idx in range(20, 400, 80):
                if idx < 180:
                    pygame.draw.line(grid_surface, lane_color_w, (cell_size * 16, cell_size + idx),
                                     (cell_size * 16, cell_size + idx + 40), grid_line_width - 2)
                    # pygame.draw.line(grid_surface, lane_color_w, (cell_size * 16, cell_size * 20 + idx), (cell_size * 16, cell_size * 20 + idx + 40), grid_line_width - 2)

                pygame.draw.line(grid_surface, lane_color_w, (cell_size * 16, cell_size * 10 + idx),
                                 (cell_size * 16, cell_size * 10 + idx + 40), grid_line_width - 2)
        if create_arrow:
            # Arrow
            grid_surface.blit(self.arrow_left_north, (cell_size * 7 + 20, cell_size * 10 + 15))
            grid_surface.blit(self.arrow_straight_north, (cell_size * 8 + 20, cell_size * 10 + 15))
            grid_surface.blit(self.arrow_right_north, (cell_size * 9 + 20, cell_size * 10 + 15))
            grid_surface.blit(self.arrow_left_south, (cell_size * 7 + 20, cell_size * 2 - 15))
            grid_surface.blit(self.arrow_straight_south, (cell_size * 6 + 20, cell_size * 2 - 15))
            grid_surface.blit(self.arrow_right_south, (cell_size * 5 + 20, cell_size * 2 - 15))

            grid_surface.blit(self.arrow_left_east, (cell_size * 3 - 15, cell_size * 6 + 20))
            grid_surface.blit(self.arrow_straight_east, (cell_size * 3 - 15, cell_size * 7 + 20))
            grid_surface.blit(self.arrow_right_east, (cell_size * 3 - 15, cell_size * 8 + 20))
            grid_surface.blit(self.arrow_left_west, (cell_size * 11 + 15, cell_size * 6 + 20))
            grid_surface.blit(self.arrow_straight_west, (cell_size * 11 + 15, cell_size * 5 + 20))
            grid_surface.blit(self.arrow_right_west, (cell_size * 11 + 15, cell_size * 4 + 20))

            grid_surface.blit(self.arrow_uturn_west, (cell_size * 18 + 15, cell_size * 6 + 20))
            grid_surface.blit(self.arrow_straight_west, (cell_size * 18 + 15, cell_size * 5 + 20))
            grid_surface.blit(self.arrow_right_west, (cell_size * 18 + 15, cell_size * 4 + 20))

            grid_surface.blit(self.arrow_left_north, (cell_size * 15 + 20, cell_size * 10 + 15))
            grid_surface.blit(self.arrow_right_north, (cell_size * 16 + 20, cell_size * 10 + 15))
            grid_surface.blit(self.arrow_right_north, (cell_size * 16 + 20, cell_size * 20 + 15))

            grid_surface.blit(self.arrow_left_east, (cell_size * 3 - 15, cell_size * 17 + 20))
            grid_surface.blit(self.arrow_straight_east, (cell_size * 3 - 15, cell_size * 18 + 15))
            grid_surface.blit(self.arrow_all_east, (cell_size * 13 - 15, cell_size * 18 + 15))

            grid_surface.blit(self.arrow_straight_east, (cell_size * 13 - 5, cell_size * 7 + 25))
            grid_surface.blit(self.arrow_noright_east, (cell_size * 13 - 5, cell_size * 8 + 5))

            grid_surface.blit(self.arrow_uturn_west, (cell_size * 11 + 15, cell_size * 17 + 20))
            grid_surface.blit(self.arrow_right_west, (cell_size * 11 + 15, cell_size * 16 + 20))
            grid_surface.blit(self.arrow_straight_west, (cell_size * 18 + 15, cell_size * 17 + 20))
            grid_surface.blit(self.arrow_right_west, (cell_size * 18 + 15, cell_size * 16 + 20))

            grid_surface.blit(self.arrow_left_south, (cell_size * 6 + 20, cell_size * 14 - 15))
            grid_surface.blit(self.arrow_rightonly_south, (cell_size * 5 + 20, cell_size * 14 - 15))
        if create_building:
            grid_surface.blit(self.house_icon, (cell_size * 1 + 30, cell_size * 20 + 20))
            grid_surface.blit(self.school_icon, (cell_size * 20 + 20, cell_size * 1 + 20))
            grid_surface.blit(self.park_icon, (cell_size * 11 + 40, cell_size * 10 + 20))
            grid_surface.blit(self.park_icon, (cell_size * 11 + 40, cell_size * 12 + 60))

        # Draw vertical lines
        # for x in range(0, width, cell_size):
        #     pygame.draw.line(grid_surface, grid_color, (x, 0), (x, height), grid_line_width)
        #
        # # Draw horizontal lines
        # for y in range(0, height, cell_size):
        #     pygame.draw.line(grid_surface, grid_color, (0, y), (width, y), grid_line_width)
        return grid_surface

    def load_agent(self, agent, agent_name):
        """ Load the RL agent into the GUI. """
        self.agent = agent
        self.agent_name = agent_name

    def render_scoreboard(self, score, time_elapsed, reward):
        # scores
        self.total_score = reward
        text = ("Earnings", '$' + str(score))
        ct = 0
        disp_text = self.text_font.render(text[0], True, (0, 0, 0))
        disp_num = self.num_font.render(text[1], True, (0, 0, 0))

        # self.screen.blit(disp_text, (75+self.screen_size[0]- 4.75*self.CELL_SIZE , 15+self.screen_size[1] // 2 + self.screen_size[1] // 12 * ct - disp_text.get_height() ))
        self.screen.blit(disp_text,
                         ((self.FIELD_SIZE - 0.5) * self.CELL_SIZE, self.screen_size[1] // 2 + 6 * self.CELL_SIZE))

        ct += 1
        cell_coords = pygame.Rect(
            (self.FIELD_SIZE - 0.5) * self.CELL_SIZE,
            self.screen_size[1] // 2 + disp_num.get_height() + 6 * self.CELL_SIZE,
            2.5 * self.CELL_SIZE,
            disp_num.get_height() - 10,
        )

        bar_size = 5
        if score <= 0:
            bar_coords = pygame.Rect(self.screen_size[0] - 1.95 * self.CELL_SIZE,
                                     self.screen_size[1] // 2 + 7 * self.CELL_SIZE,
                                     0.8 * self.CELL_SIZE, -bar_size * score, )
        else:
            bar_coords = pygame.Rect(self.screen_size[0] - 1.95 * self.CELL_SIZE,
                                     self.screen_size[1] // 2 + 7 * self.CELL_SIZE - bar_size * score,
                                     0.8 * self.CELL_SIZE,
                                     bar_size * score, )

        if reward == None:
            print("reward is None")
            print("124214")

        # print("reward", reward)
        if reward < 0:
            delta_coords = pygame.Rect(self.screen_size[0] - 1.95 * self.CELL_SIZE,
                                       self.screen_size[1] // 2 + 7 * self.CELL_SIZE - bar_size * (score - reward),
                                       0.8 * self.CELL_SIZE,
                                       -bar_size * reward, )
        elif reward > 0:
            delta_coords = pygame.Rect(self.screen_size[0] - 1.95 * self.CELL_SIZE,
                                       self.screen_size[1] // 2 + 7 * self.CELL_SIZE - bar_size * score,
                                       0.8 * self.CELL_SIZE,
                                       bar_size * reward, )

        x_start = self.screen_size[0] - 2.5 * self.CELL_SIZE
        x_end = self.screen_size[0] - 0.9 * self.CELL_SIZE
        y = self.screen_size[1] // 2 + 7 * self.CELL_SIZE

        origin_marker = self.marker_font.render("$0", True, (0, 0, 0))
        positive_marker = self.marker_font.render("+", True, (0, 0, 0))
        negative_marker = self.marker_font.render("-", True, (0, 0, 0))

        pygame.draw.line(self.screen, (0, 0, 0), (x_start, y), (x_end, y), 4)

        if reward == self.env.rewards['good_fruit']:
            pygame.draw.rect(self.screen, Colors.SCORE_GOOD, cell_coords)
            pygame.draw.rect(self.screen, Colors.SCORE, bar_coords)
            pygame.draw.rect(self.screen, Colors.SCORE_GOOD, delta_coords)
        elif reward == self.env.rewards['correct_accident']:
            pygame.draw.rect(self.screen, Colors.SCORE_GOOD, cell_coords)
            pygame.draw.rect(self.screen, Colors.SCORE, bar_coords)
            pygame.draw.rect(self.screen, Colors.SCORE_GOOD, delta_coords)
        elif reward == self.env.rewards['violation']:
            pygame.draw.rect(self.screen, Colors.SCORE_BAD, cell_coords)
            pygame.draw.rect(self.screen, Colors.SCORE, bar_coords)
            pygame.draw.rect(self.screen, Colors.SCORE_BAD, delta_coords)
        elif reward == self.env.rewards['accident']:
            pygame.draw.rect(self.screen, Colors.SCORE_VERY_BAD, cell_coords)
            pygame.draw.rect(self.screen, Colors.SCORE, bar_coords)
            pygame.draw.rect(self.screen, Colors.SCORE_VERY_BAD, delta_coords)
        else:
            pygame.draw.rect(self.screen, Colors.SCORE, cell_coords)
            pygame.draw.rect(self.screen, Colors.SCORE, bar_coords)

        pygame.draw.rect(self.screen, (0, 0, 0), bar_coords, 3)
        self.screen.blit(origin_marker, (
            self.screen_size[0] - 0.8 * self.CELL_SIZE, self.screen_size[1] // 2 - 20 + 7 * self.CELL_SIZE))
        self.screen.blit(positive_marker, (
            self.screen_size[0] - 2.43 * self.CELL_SIZE, -33 + self.screen_size[1] // 2 + 7 * self.CELL_SIZE))
        self.screen.blit(negative_marker, (
            self.screen_size[0] - 2.4 * self.CELL_SIZE, -7 + self.screen_size[1] // 2 + 7 * self.CELL_SIZE))
        self.screen.blit(disp_num,
                         ((self.FIELD_SIZE - 0.3) * self.CELL_SIZE, self.screen_size[1] // 2 + 7 * self.CELL_SIZE))

        ct += 2

        if self.display_time:
            time_elapsed = time_elapsed / 1000.0

            total_seconds = 50 + time_elapsed
            total_minutes = 59 + total_seconds // 59
            total_hours = 6 + total_minutes // 60

            # Apply the carry-over operations
            seconds = total_seconds % 59
            minutes = total_minutes % 60
            hours = total_hours

            # text = ("Time", str(round(time_remaining / (1000.0 / self.timestep_delay))))
            text = ("Time for school", str('{:02.0f}:{:02.0f}:{:02.0f}'.format(hours, minutes, seconds)))
            # print(time_elapsed)
            disp_text = self.text_font.render(text[0], True, (0, 0, 0))
            disp_num = self.num_font.render(text[1], True, (50, 205, 50))

            self.screen.blit(disp_text, (self.screen_size[0] - 6.45 * self.CELL_SIZE, 70))
            ct += 1
            cell_coords = pygame.Rect(self.screen_size[0] - 6.35 * self.CELL_SIZE, 135, 4 * self.CELL_SIZE,
                                      disp_num.get_height() - 10)
            pygame.draw.rect(self.screen, (59, 59, 59), cell_coords)
            self.screen.blit(disp_num, (self.screen_size[0] - 5.95 * self.CELL_SIZE, 130))

            return seconds  # for controlling traffic lights

    def render_initial_traffic_lights(self):

        self.render(counter=0)
        self.screen.blit(self.two_all_red_south, (self.CELL_SIZE * 6 - 10, self.CELL_SIZE * 4 + 10))  # 1
        self.screen.blit(self.two_all_red_west, (self.CELL_SIZE * 9 + 15, self.CELL_SIZE * 5 - 10))  # 2
        self.screen.blit(self.two_all_red_north, (self.CELL_SIZE * 8 - 10, self.CELL_SIZE * 8 + 20))  # 3
        self.screen.blit(self.two_all_red_east, (self.CELL_SIZE * 5 + 10, self.CELL_SIZE * 7 - 10))  # 4
        self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 7 + 60, self.CELL_SIZE * 3 + 17))  # 5
        self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 10 + 17, self.CELL_SIZE * 6 + 60))  # 6
        self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 6 + 55, self.CELL_SIZE * 9 + 17))  # 7
        self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 4 + 17, self.CELL_SIZE * 5 + 60))  # 8
        self.screen.blit(self.uturn_all_red_west, (self.CELL_SIZE * 16 + 15, self.CELL_SIZE * 5 - 10))  # 9
        self.screen.blit(self.single_red, (self.CELL_SIZE * 16 - 30, self.CELL_SIZE * 8 + 35))  # 10
        self.screen.blit(self.single_red, (self.CELL_SIZE * 15, self.CELL_SIZE * 8 - 25))  # 11
        self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 15 + 60, self.CELL_SIZE * 3 + 17))  # 12
        self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 17 + 17, self.CELL_SIZE * 6 + 60))  # 13
        self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 15 + 60, self.CELL_SIZE * 9 + 17))  # 14
        self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 14 + 17, self.CELL_SIZE * 6 + 60))  # 15
        self.screen.blit(self.single_red, (self.CELL_SIZE * 6 - 25, self.CELL_SIZE * 16))  # 16
        self.screen.blit(self.single_red, (self.CELL_SIZE * 9 + 30, self.CELL_SIZE * 17 - 30))  # 17
        self.screen.blit(self.single_red, (self.CELL_SIZE * 5, self.CELL_SIZE * 18 - 25))  # 18
        self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 6 + 55, self.CELL_SIZE * 15 + 17))  # 19
        self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 10 + 17, self.CELL_SIZE * 17 + 60))  # 20
        self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 4 + 17, self.CELL_SIZE * 16 + 60))  # 21
        self.screen.blit(self.single_red, (self.CELL_SIZE * 16 + 30, self.CELL_SIZE * 17 - 30))  # 22
        self.screen.blit(self.single_red, (self.CELL_SIZE * 16 + 15, self.CELL_SIZE * 18 + 35))  # 23
        self.screen.blit(self.single_red, (self.CELL_SIZE * 15, self.CELL_SIZE * 18 + 15))  # 24
        self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 15 + 60, self.CELL_SIZE * 15 + 17))  # 25
        self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 17 + 17, self.CELL_SIZE * 17 + 60))  # 26
        self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 15 + 60, self.CELL_SIZE * 19 + 17))  # 27
        self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 14 + 17, self.CELL_SIZE * 17 + 60))  # 28

        # self.screen.blit(self.pedestrian_red_east, (self.CELL_SIZE * 5 - 15, self.CELL_SIZE * 3 + 15))
        # self.screen.blit(self.pedestrian_red_west, (self.CELL_SIZE * 10 - 35, self.CELL_SIZE * 3 + 15))
        # self.screen.blit(self.pedestrian_red_east, (self.CELL_SIZE * 5 - 15, self.CELL_SIZE * 9 + 20))
        # self.screen.blit(self.pedestrian_red_west, (self.CELL_SIZE * 10 - 35, self.CELL_SIZE * 9 + 20))
        # self.screen.blit(self.pedestrian_red_east, (self.CELL_SIZE * 5 - 15, self.CELL_SIZE * 15 + 20))
        # self.screen.blit(self.pedestrian_red_west, (self.CELL_SIZE * 10 - 35, self.CELL_SIZE * 15 + 20))
        # self.screen.blit(self.pedestrian_red_east, (self.CELL_SIZE * 15 - 15, self.CELL_SIZE * 3 + 15))
        # self.screen.blit(self.pedestrian_red_west, (self.CELL_SIZE * 17 - 35, self.CELL_SIZE * 3 + 15))
        # self.screen.blit(self.pedestrian_red_east, (self.CELL_SIZE * 15 - 15, self.CELL_SIZE * 9 + 20))
        # self.screen.blit(self.pedestrian_red_west, (self.CELL_SIZE * 17 - 35, self.CELL_SIZE * 9 + 20))
        # self.screen.blit(self.pedestrian_red_east, (self.CELL_SIZE * 15 - 15, self.CELL_SIZE * 15 + 20))
        # self.screen.blit(self.pedestrian_red_west, (self.CELL_SIZE * 17 - 35, self.CELL_SIZE * 15 + 20))
        # self.screen.blit(self.pedestrian_red_east, (self.CELL_SIZE * 15 - 15, self.CELL_SIZE * 19 + 20))
        # self.screen.blit(self.pedestrian_red_west, (self.CELL_SIZE * 17 - 35, self.CELL_SIZE * 19 + 20))
        # self.screen.blit(self.pedestrian_red_south, (self.CELL_SIZE * 10 + 17, self.CELL_SIZE * 3 + 70))  # 5
        # self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 10 + 17, self.CELL_SIZE * 8 + 50))  # 5
        # self.screen.blit(self.pedestrian_red_south, (self.CELL_SIZE * 10 + 17, self.CELL_SIZE * 15 + 70))  #
        # self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 10 + 17, self.CELL_SIZE * 18 + 45))  # 5
        # self.screen.blit(self.pedestrian_red_south, (self.CELL_SIZE * 4 + 17, self.CELL_SIZE * 3 + 70))  # 5
        # self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 4 + 17, self.CELL_SIZE * 8 + 50))  # 5
        # self.screen.blit(self.pedestrian_red_south, (self.CELL_SIZE * 4 + 17, self.CELL_SIZE * 15 + 70))  # 5
        # self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 4 + 17, self.CELL_SIZE * 18 + 45))  # 5
        # self.screen.blit(self.pedestrian_red_south, (self.CELL_SIZE * 14 + 17, self.CELL_SIZE * 3 + 70))  # 5
        # self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 14 + 17, self.CELL_SIZE * 8 + 50))  # 5
        # self.screen.blit(self.pedestrian_red_south, (self.CELL_SIZE * 14 + 17, self.CELL_SIZE * 15 + 70))  # 5
        # self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 14 + 17, self.CELL_SIZE * 18 + 45))  # 5
        # self.screen.blit(self.pedestrian_red_south, (self.CELL_SIZE * 17 + 17, self.CELL_SIZE * 3 + 70))  # 5
        # self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 17 + 17, self.CELL_SIZE * 8 + 50))  # 5
        # self.screen.blit(self.pedestrian_red_south, (self.CELL_SIZE * 17 + 17, self.CELL_SIZE * 15 + 70))  # 5
        # self.screen.blit(self.pedestrian_red_north, (self.CELL_SIZE * 17 + 17, self.CELL_SIZE * 18 + 45))  # 5

    # def render_traffic_lights(self, traffic_lights_pos_dict, time):
    #
    #     time = time / 1000.0
    #     print(time)
    #     if 0 < time % 40 < 10:  # NS straight
    #         self.screen.blit(self.two_left_red_south, traffic_lights_pos_dict['01'])
    #         self.screen.blit(self.two_left_red_north, traffic_lights_pos_dict['03'])
    #         self.screen.blit(self.pedestrian_green_north, traffic_lights_pos_dict['06'])
    #         self.screen.blit(self.pedestrian_green_north, traffic_lights_pos_dict['08'])
    #
    #         self.screen.blit(self.two_all_red_west, traffic_lights_pos_dict['02'])
    #         self.screen.blit(self.two_all_red_east, traffic_lights_pos_dict['04'])
    #         self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['05'])
    #         self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['07'])
    #
    #     if 10 <= time % 40 < 17:  # NS left
    #         self.screen.blit(self.two_straight_red_south, traffic_lights_pos_dict['01'])
    #         self.screen.blit(self.two_straight_red_north, traffic_lights_pos_dict['03'])
    #         self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['06'])
    #         self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['08'])
    #         self.screen.blit(self.two_all_red_west, traffic_lights_pos_dict['02'])
    #         self.screen.blit(self.two_all_red_east, traffic_lights_pos_dict['04'])
    #         self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['05'])
    #         self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['07'])
    #
    #     if 17 <= time % 40 < 20:  # All stop
    #         self.screen.blit(self.two_all_red_south, traffic_lights_pos_dict['01'])
    #         self.screen.blit(self.two_all_red_north, traffic_lights_pos_dict['03'])
    #         self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['06'])
    #         self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['08'])
    #         self.screen.blit(self.two_all_red_west, traffic_lights_pos_dict['02'])
    #         self.screen.blit(self.two_all_red_east, traffic_lights_pos_dict['04'])
    #         self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['05'])
    #         self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['07'])
    #
    #     if 20 <= time % 40 < 30:  # WE straight
    #         self.screen.blit(self.two_left_red_west, traffic_lights_pos_dict['02'])
    #         self.screen.blit(self.two_left_red_east, traffic_lights_pos_dict['04'])
    #         self.screen.blit(self.pedestrian_green_north, traffic_lights_pos_dict['05'])
    #         self.screen.blit(self.pedestrian_green_north, traffic_lights_pos_dict['07'])
    #         self.screen.blit(self.two_all_red_south, traffic_lights_pos_dict['01'])
    #         self.screen.blit(self.two_all_red_north, traffic_lights_pos_dict['03'])
    #         self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['06'])
    #         self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['08'])
    #
    #     if 30 <= time % 40 < 37:  # WE left
    #         self.screen.blit(self.two_straight_red_west, traffic_lights_pos_dict['02'])
    #         self.screen.blit(self.two_straight_red_east, traffic_lights_pos_dict['04'])
    #         self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['05'])
    #         self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['07'])
    #         self.screen.blit(self.two_all_red_south, traffic_lights_pos_dict['01'])
    #         self.screen.blit(self.two_all_red_north, traffic_lights_pos_dict['03'])
    #         self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['06'])
    #         self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['08'])
    #
    #     if 37 <= time % 40 < 40:  # All stop
    #         self.screen.blit(self.two_all_red_west, traffic_lights_pos_dict['02'])
    #         self.screen.blit(self.two_all_red_east, traffic_lights_pos_dict['04'])
    #         self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['05'])
    #         self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['07'])
    #         self.screen.blit(self.two_all_red_south, traffic_lights_pos_dict['01'])
    #         self.screen.blit(self.two_all_red_north, traffic_lights_pos_dict['03'])
    #         self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['06'])
    #         self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['08'])
    def render_traffic_lights(self, traffic_lights_pos_dict, time):
        time = time / 1000.0
        # print(time)

        # Calculate the blinking effect: blink every 0.5 seconds (you can adjust the blink rate)
        blink = (time * 2) % 2 < 1

        ### FIRST INTERSECTION
        if 0 < time % 40 < 8:  # NS straight
            if 5 <= time % 40 and blink:  # Blinking effect during the last 3 seconds
                self.screen.blit(self.two_left_red_off_south, traffic_lights_pos_dict['01'])
                self.screen.blit(self.two_left_red_off_north, traffic_lights_pos_dict['03'])
                self.screen.blit(self.pedestrian_green_off, traffic_lights_pos_dict['06'])
                self.screen.blit(self.pedestrian_green_off, traffic_lights_pos_dict['08'])
                self.traffic_light_status['01'] = 13
                self.traffic_light_status['03'] = 13
                self.traffic_light_status['06'] = 3
                self.traffic_light_status['08'] = 3
            else:
                self.screen.blit(self.two_left_red_south, traffic_lights_pos_dict['01'])
                self.screen.blit(self.two_left_red_north, traffic_lights_pos_dict['03'])
                self.screen.blit(self.pedestrian_green_north, traffic_lights_pos_dict['06'])
                self.screen.blit(self.pedestrian_green_north, traffic_lights_pos_dict['08'])
                self.traffic_light_status['01'] = 12
                self.traffic_light_status['03'] = 12
                self.traffic_light_status['06'] = 2
                self.traffic_light_status['08'] = 2
            self.screen.blit(self.two_all_red_west, traffic_lights_pos_dict['02'])
            self.screen.blit(self.two_all_red_east, traffic_lights_pos_dict['04'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['05'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['07'])
            self.traffic_light_status['02'] = 11
            self.traffic_light_status['04'] = 11
            self.traffic_light_status['05'] = 1
            self.traffic_light_status['07'] = 1
        if 8 <= time % 40 < 13:  # NS left
            if 10 <= time % 40 and blink:
                self.screen.blit(self.two_straight_red_off_north, traffic_lights_pos_dict['03'])
                self.traffic_light_status['03'] = 31
            else:
                self.screen.blit(self.two_straight_red_north, traffic_lights_pos_dict['03'])
                self.traffic_light_status['03'] = 21

            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['06'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['08'])
            self.screen.blit(self.two_all_red_west, traffic_lights_pos_dict['02'])
            self.screen.blit(self.two_all_red_east, traffic_lights_pos_dict['04'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['05'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['07'])
            self.screen.blit(self.two_all_red_south, traffic_lights_pos_dict['01'])

            self.traffic_light_status['06'] = 1
            self.traffic_light_status['08'] = 1
            self.traffic_light_status['02'] = 11
            self.traffic_light_status['04'] = 11
            self.traffic_light_status['05'] = 1
            self.traffic_light_status['07'] = 1
            self.traffic_light_status['01'] = 11

        if 13 <= time % 40 < 15:  # All stop
            self.screen.blit(self.two_all_red_south, traffic_lights_pos_dict['01'])
            self.screen.blit(self.two_all_red_north, traffic_lights_pos_dict['03'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['06'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['08'])
            self.screen.blit(self.two_all_red_west, traffic_lights_pos_dict['02'])
            self.screen.blit(self.two_all_red_east, traffic_lights_pos_dict['04'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['05'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['07'])
            self.traffic_light_status['01'] = 11
            self.traffic_light_status['03'] = 11
            self.traffic_light_status['06'] = 1
            self.traffic_light_status['08'] = 1
            self.traffic_light_status['02'] = 11
            self.traffic_light_status['04'] = 11
            self.traffic_light_status['05'] = 1
            self.traffic_light_status['07'] = 1
        if 15 <= time % 40 < 20:  # NS left
            if 17 <= time % 40 and blink:
                self.screen.blit(self.two_straight_red_off_south, traffic_lights_pos_dict['01'])
                self.traffic_light_status['01'] = 31
            else:
                self.screen.blit(self.two_straight_red_south, traffic_lights_pos_dict['01'])
                self.traffic_light_status['01'] = 21

            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['06'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['08'])
            self.screen.blit(self.two_all_red_west, traffic_lights_pos_dict['02'])
            self.screen.blit(self.two_all_red_east, traffic_lights_pos_dict['04'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['05'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['07'])
            self.screen.blit(self.two_all_red_north, traffic_lights_pos_dict['03'])

            self.traffic_light_status['06'] = 1
            self.traffic_light_status['08'] = 1
            self.traffic_light_status['02'] = 11
            self.traffic_light_status['04'] = 11
            self.traffic_light_status['05'] = 1
            self.traffic_light_status['07'] = 1
            self.traffic_light_status['03'] = 11

        if 20 <= time % 40 < 28:  # WE straight
            if 25 <= time % 40 and blink:  # Blinking effect during the last 3 seconds
                self.screen.blit(self.two_left_red_off_west, traffic_lights_pos_dict['02'])
                self.screen.blit(self.two_left_red_off_east, traffic_lights_pos_dict['04'])
                self.screen.blit(self.pedestrian_green_off, traffic_lights_pos_dict['05'])
                self.screen.blit(self.pedestrian_green_off, traffic_lights_pos_dict['07'])
                self.traffic_light_status['02'] = 13
                self.traffic_light_status['04'] = 13
                self.traffic_light_status['05'] = 3
                self.traffic_light_status['07'] = 3
            else:
                self.screen.blit(self.two_left_red_west, traffic_lights_pos_dict['02'])
                self.screen.blit(self.two_left_red_east, traffic_lights_pos_dict['04'])
                self.screen.blit(self.pedestrian_green_north, traffic_lights_pos_dict['05'])
                self.screen.blit(self.pedestrian_green_north, traffic_lights_pos_dict['07'])
                self.traffic_light_status['02'] = 12
                self.traffic_light_status['04'] = 12
                self.traffic_light_status['05'] = 2
                self.traffic_light_status['07'] = 2

            self.screen.blit(self.two_all_red_south, traffic_lights_pos_dict['01'])
            self.screen.blit(self.two_all_red_north, traffic_lights_pos_dict['03'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['06'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['08'])
            self.traffic_light_status['01'] = 11
            self.traffic_light_status['03'] = 11
            self.traffic_light_status['06'] = 1
            self.traffic_light_status['08'] = 1
        if 28 <= time % 40 < 33:  # WE left
            if 30 <= time % 40 and blink:
                self.screen.blit(self.two_straight_red_off_east, traffic_lights_pos_dict['04'])
                self.traffic_light_status['04'] = 31
            else:
                self.screen.blit(self.two_straight_red_east, traffic_lights_pos_dict['04'])
                self.traffic_light_status['04'] = 21

            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['05'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['07'])
            self.screen.blit(self.two_all_red_south, traffic_lights_pos_dict['01'])
            self.screen.blit(self.two_all_red_north, traffic_lights_pos_dict['03'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['06'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['08'])
            self.screen.blit(self.two_all_red_west, traffic_lights_pos_dict['02'])

            self.traffic_light_status['05'] = 1
            self.traffic_light_status['07'] = 1
            self.traffic_light_status['01'] = 11
            self.traffic_light_status['03'] = 11
            self.traffic_light_status['06'] = 1
            self.traffic_light_status['08'] = 1
            self.traffic_light_status['02'] = 11

        if 33 <= time % 40 < 35:  # All stop
            self.screen.blit(self.two_all_red_west, traffic_lights_pos_dict['02'])
            self.screen.blit(self.two_all_red_east, traffic_lights_pos_dict['04'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['05'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['07'])
            self.screen.blit(self.two_all_red_south, traffic_lights_pos_dict['01'])
            self.screen.blit(self.two_all_red_north, traffic_lights_pos_dict['03'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['06'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['08'])
            self.traffic_light_status['02'] = 11
            self.traffic_light_status['04'] = 11
            self.traffic_light_status['05'] = 1
            self.traffic_light_status['07'] = 1
            self.traffic_light_status['01'] = 11
            self.traffic_light_status['03'] = 11
            self.traffic_light_status['06'] = 1
            self.traffic_light_status['08'] = 1

        if 35 <= time % 40 < 40:  # WE left
            if 38 <= time % 40 and blink:
                self.screen.blit(self.two_straight_red_off_west, traffic_lights_pos_dict['02'])
                self.traffic_light_status['02'] = 31
            else:
                self.screen.blit(self.two_straight_red_west, traffic_lights_pos_dict['02'])
                self.traffic_light_status['02'] = 21

            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['05'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['07'])
            self.screen.blit(self.two_all_red_south, traffic_lights_pos_dict['01'])
            self.screen.blit(self.two_all_red_north, traffic_lights_pos_dict['03'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['06'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['08'])
            self.screen.blit(self.two_all_red_east, traffic_lights_pos_dict['04'])

            self.traffic_light_status['05'] = 1
            self.traffic_light_status['07'] = 1
            self.traffic_light_status['01'] = 11
            self.traffic_light_status['03'] = 11
            self.traffic_light_status['06'] = 1
            self.traffic_light_status['08'] = 1
            self.traffic_light_status['04'] = 11

        ### SECOND INTERSECTION
        if 0 < time % 33 < 10:  # NS straight
            if 7 <= time % 33 and blink:  # Blinking effect during the last 3 seconds
                self.screen.blit(self.single_off, traffic_lights_pos_dict['10'])
                self.screen.blit(self.pedestrian_green_off, traffic_lights_pos_dict['13'])
                self.traffic_light_status['10'] = 3
                self.traffic_light_status['13'] = 3
            else:
                self.screen.blit(self.single_green, traffic_lights_pos_dict['10'])
                self.screen.blit(self.pedestrian_green_north, traffic_lights_pos_dict['13'])
                self.traffic_light_status['10'] = 2
                self.traffic_light_status['13'] = 2
            self.screen.blit(self.single_red, traffic_lights_pos_dict['11'])
            self.screen.blit(self.uturn_all_red_west, traffic_lights_pos_dict['09'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['12'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['14'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['15'])
            self.traffic_light_status['11'] = 1
            self.traffic_light_status['09'] = 11
            self.traffic_light_status['12'] = 1
            self.traffic_light_status['14'] = 1
            self.traffic_light_status['15'] = 1
        if (10 <= time % 33 < 13) or (30 <= time % 33 < 33):  # All stop
            self.screen.blit(self.single_red, traffic_lights_pos_dict['10'])
            self.screen.blit(self.single_red, traffic_lights_pos_dict['11'])
            self.screen.blit(self.uturn_all_red_west, traffic_lights_pos_dict['09'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['12'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['13'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['14'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['15'])
            self.traffic_light_status['10'] = 1
            self.traffic_light_status['11'] = 1
            self.traffic_light_status['09'] = 11
            self.traffic_light_status['12'] = 1
            self.traffic_light_status['13'] = 1
            self.traffic_light_status['14'] = 1
            self.traffic_light_status['15'] = 1
        if 13 <= time % 33 < 23:  # WE straight
            if 20 <= time % 33 and blink:  # Blinking effect during the last 3 seconds
                self.screen.blit(self.single_off, traffic_lights_pos_dict['11'])
                self.screen.blit(self.uturn_uturn_red_off_west, traffic_lights_pos_dict['09'])
                self.screen.blit(self.pedestrian_green_off, traffic_lights_pos_dict['12'])
                self.screen.blit(self.pedestrian_green_off, traffic_lights_pos_dict['14'])
                self.traffic_light_status['11'] = 3
                self.traffic_light_status['09'] = 13
                self.traffic_light_status['12'] = 3
                self.traffic_light_status['14'] = 3

            else:
                self.screen.blit(self.single_green, traffic_lights_pos_dict['11'])
                self.screen.blit(self.uturn_uturn_red_west, traffic_lights_pos_dict['09'])
                self.screen.blit(self.pedestrian_green_north, traffic_lights_pos_dict['12'])
                self.screen.blit(self.pedestrian_green_north, traffic_lights_pos_dict['14'])
                self.traffic_light_status['11'] = 2
                self.traffic_light_status['09'] = 12
                self.traffic_light_status['12'] = 2
                self.traffic_light_status['14'] = 2
            self.screen.blit(self.single_red, traffic_lights_pos_dict['10'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['13'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['15'])
            self.traffic_light_status['10'] = 1
            self.traffic_light_status['13'] = 1
            self.traffic_light_status['15'] = 1
        if 23 <= time % 33 < 30:  # WE left
            if 27 <= time % 33 and blink:
                self.screen.blit(self.pedestrian_green_off, traffic_lights_pos_dict['15'])
                self.screen.blit(self.uturn_straight_red_off_west, traffic_lights_pos_dict['09'])
                self.traffic_light_status['15'] = 3
                self.traffic_light_status['09'] = 31
            else:
                self.screen.blit(self.pedestrian_green_north, traffic_lights_pos_dict['15'])
                self.screen.blit(self.uturn_straight_red_west, traffic_lights_pos_dict['09'])
                self.traffic_light_status['15'] = 2
                self.traffic_light_status['09'] = 21
            self.screen.blit(self.single_red, traffic_lights_pos_dict['10'])
            self.screen.blit(self.single_red, traffic_lights_pos_dict['11'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['12'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['13'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['14'])
            self.traffic_light_status['10'] = 1
            self.traffic_light_status['11'] = 1
            self.traffic_light_status['12'] = 1
            self.traffic_light_status['13'] = 1
            self.traffic_light_status['14'] = 1

        ### THIRD INTERSECTION
        if 0 < time % 33 < 10:  # WE uturn
            if 7 <= time % 33 and blink:  # Blinking effect during the last 3 seconds
                self.screen.blit(self.single_off, traffic_lights_pos_dict['17'])
                self.traffic_light_status['17'] = 3
            else:
                self.screen.blit(self.single_green, traffic_lights_pos_dict['17'])
                self.traffic_light_status['17'] = 2
            self.screen.blit(self.single_red, traffic_lights_pos_dict['18'])
            self.screen.blit(self.single_red, traffic_lights_pos_dict['16'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['19'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['20'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['21'])
            self.traffic_light_status['18'] = 1
            self.traffic_light_status['16'] = 1
            self.traffic_light_status['19'] = 1
            self.traffic_light_status['20'] = 1
            self.traffic_light_status['21'] = 1
        if 10 <= time % 33 < 17:  # WE left and straight
            if 14 <= time % 33 and blink:
                self.screen.blit(self.single_off, traffic_lights_pos_dict['18'])
                self.traffic_light_status['18'] = 3
            else:
                self.screen.blit(self.single_green, traffic_lights_pos_dict['18'])
                self.traffic_light_status['18'] = 2
            self.screen.blit(self.single_red, traffic_lights_pos_dict['17'])
            self.screen.blit(self.single_red, traffic_lights_pos_dict['16'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['19'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['20'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['21'])
            self.traffic_light_status['17'] = 1
            self.traffic_light_status['16'] = 1
            self.traffic_light_status['19'] = 1
            self.traffic_light_status['20'] = 1
            self.traffic_light_status['21'] = 1
        if 17 <= time % 33 < 18:  # All stop
            self.screen.blit(self.single_red, traffic_lights_pos_dict['16'])
            self.screen.blit(self.single_red, traffic_lights_pos_dict['17'])
            self.screen.blit(self.single_red, traffic_lights_pos_dict['18'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['19'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['20'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['21'])
            self.traffic_light_status['16'] = 1
            self.traffic_light_status['17'] = 1
            self.traffic_light_status['18'] = 1
            self.traffic_light_status['19'] = 1
            self.traffic_light_status['20'] = 1
            self.traffic_light_status['21'] = 1
        if 18 <= time % 33 < 25:
            if 22 <= time % 33 and blink:
                self.screen.blit(self.pedestrian_green_off, traffic_lights_pos_dict['19'])
                self.screen.blit(self.pedestrian_green_off, traffic_lights_pos_dict['20'])
                self.screen.blit(self.pedestrian_green_off, traffic_lights_pos_dict['21'])
                self.traffic_light_status['19'] = 3
                self.traffic_light_status['20'] = 3
                self.traffic_light_status['21'] = 3
            else:
                self.screen.blit(self.pedestrian_green_north, traffic_lights_pos_dict['19'])
                self.screen.blit(self.pedestrian_green_north, traffic_lights_pos_dict['20'])
                self.screen.blit(self.pedestrian_green_north, traffic_lights_pos_dict['21'])
                self.traffic_light_status['19'] = 2
                self.traffic_light_status['20'] = 2
                self.traffic_light_status['21'] = 2
            self.screen.blit(self.single_red, traffic_lights_pos_dict['16'])
            self.screen.blit(self.single_red, traffic_lights_pos_dict['17'])
            self.screen.blit(self.single_red, traffic_lights_pos_dict['18'])
            self.traffic_light_status['16'] = 1
            self.traffic_light_status['17'] = 1
            self.traffic_light_status['18'] = 1
        if 25 <= time % 33 < 33:
            if 30 <= time % 33 and blink:
                self.screen.blit(self.single_off, traffic_lights_pos_dict['16'])
                self.traffic_light_status['16'] = 3
            else:
                self.screen.blit(self.single_green, traffic_lights_pos_dict['16'])
                self.traffic_light_status['16'] = 2
            self.screen.blit(self.single_red, traffic_lights_pos_dict['17'])
            self.screen.blit(self.single_red, traffic_lights_pos_dict['18'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['19'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['20'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['21'])
            self.traffic_light_status['17'] = 1
            self.traffic_light_status['18'] = 1
            self.traffic_light_status['19'] = 1
            self.traffic_light_status['20'] = 1
            self.traffic_light_status['21'] = 1
        ### FOURTH INTERSECTION

        if 0 < time % 33 < 10:  # NS straight
            if 7 <= time % 33 and blink:  # Blinking effect during the last 3 seconds
                self.screen.blit(self.single_off, traffic_lights_pos_dict['23'])
                self.screen.blit(self.pedestrian_green_off, traffic_lights_pos_dict['26'])
                self.screen.blit(self.pedestrian_green_off, traffic_lights_pos_dict['28'])
                self.traffic_light_status['23'] = 3
                self.traffic_light_status['26'] = 3
                self.traffic_light_status['28'] = 3
            else:
                self.screen.blit(self.single_green, traffic_lights_pos_dict['23'])
                self.screen.blit(self.pedestrian_green_north, traffic_lights_pos_dict['26'])
                self.screen.blit(self.pedestrian_green_north, traffic_lights_pos_dict['28'])
                self.traffic_light_status['23'] = 2
                self.traffic_light_status['26'] = 2
                self.traffic_light_status['28'] = 2
            self.screen.blit(self.single_red, traffic_lights_pos_dict['24'])
            self.screen.blit(self.single_red, traffic_lights_pos_dict['22'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['25'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['27'])
            self.traffic_light_status['24'] = 1
            self.traffic_light_status['22'] = 1
            self.traffic_light_status['25'] = 1
            self.traffic_light_status['27'] = 1
        if (10 <= time % 33 < 13) or (30 <= time % 33):  # All stop
            self.screen.blit(self.single_red, traffic_lights_pos_dict['22'])
            self.screen.blit(self.single_red, traffic_lights_pos_dict['23'])
            self.screen.blit(self.single_red, traffic_lights_pos_dict['24'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['25'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['26'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['27'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['28'])
            self.traffic_light_status['22'] = 1
            self.traffic_light_status['23'] = 1
            self.traffic_light_status['24'] = 1
            self.traffic_light_status['25'] = 1
            self.traffic_light_status['26'] = 1
            self.traffic_light_status['27'] = 1
            self.traffic_light_status['28'] = 1
        if 13 <= time % 33 < 23:  # WE straight
            if 20 <= time % 33 and blink:  # Blinking effect during the last 3 seconds
                self.screen.blit(self.single_off, traffic_lights_pos_dict['22'])
                self.screen.blit(self.pedestrian_green_off, traffic_lights_pos_dict['25'])
                self.screen.blit(self.pedestrian_green_off, traffic_lights_pos_dict['27'])
                self.traffic_light_status['22'] = 3
                self.traffic_light_status['25'] = 3
                self.traffic_light_status['27'] = 3
            else:
                self.screen.blit(self.single_green, traffic_lights_pos_dict['22'])
                self.screen.blit(self.pedestrian_green_north, traffic_lights_pos_dict['25'])
                self.screen.blit(self.pedestrian_green_north, traffic_lights_pos_dict['27'])
                self.traffic_light_status['22'] = 2
                self.traffic_light_status['25'] = 2
                self.traffic_light_status['27'] = 2

            self.screen.blit(self.single_red, traffic_lights_pos_dict['23'])
            self.screen.blit(self.single_red, traffic_lights_pos_dict['24'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['26'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['28'])
            self.traffic_light_status['23'] = 1
            self.traffic_light_status['24'] = 1
            self.traffic_light_status['26'] = 1
            self.traffic_light_status['28'] = 1
        if 23 <= time % 33 < 30:  # WE left
            if 27 <= time % 33 and blink:
                self.screen.blit(self.single_off, traffic_lights_pos_dict['24'])
                self.traffic_light_status['24'] = 3
            else:
                self.screen.blit(self.single_green, traffic_lights_pos_dict['24'])
                self.traffic_light_status['24'] = 2
            self.screen.blit(self.single_red, traffic_lights_pos_dict['22'])
            self.screen.blit(self.single_red, traffic_lights_pos_dict['23'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['25'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['26'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['27'])
            self.screen.blit(self.pedestrian_red_north, traffic_lights_pos_dict['28'])
            self.traffic_light_status['22'] = 1
            self.traffic_light_status['23'] = 1
            self.traffic_light_status['25'] = 1
            self.traffic_light_status['26'] = 1
            self.traffic_light_status['27'] = 1
            self.traffic_light_status['28'] = 1

    def render_cell(self, x, y, agent_idx=None, counter=None, head_update=False):
        """ Draw the cell specified by the field coordinates. """
        cell_coords = pygame.Rect(
            x * self.CELL_SIZE,
            y * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE,
        )
        # if self.env.field[x, y] == CellType.EMPTY:
        #     pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)
        if self.env.field[x, y] == CellType.GOOD_EVENT:
            cell_coords = pygame.Rect(
                x * self.CELL_SIZE + 5,
                y * self.CELL_SIZE + 5,
                self.CELL_SIZE,
                self.CELL_SIZE,
            )
            self.screen.blit(self.good_fruit_icon, cell_coords)
        if self.env.field[x, y] == CellType.BAD_EVENT:
            self.screen.blit(self.bad_fruit_icon, cell_coords)
        if self.env.field[x, y] == CellType.LAVA:
            self.screen.blit(self.lava_icon, cell_coords)
        if head_update:
            if agent_idx is not None:
                if (x, y) == self.env.vehicles[agent_idx].head:
                    # if self.env.field[x, y] == self.env.snake.head:
                    if self.env.vehicles[agent_idx].direction == SnakeDirection.NORTH:
                        rotated_icon = self.north[self.env.vehicles[agent_idx].icon_type]
                    elif self.env.vehicles[agent_idx].direction == SnakeDirection.WEST:
                        rotated_icon = self.west[self.env.vehicles[agent_idx].icon_type]
                    elif self.env.vehicles[agent_idx].direction == SnakeDirection.SOUTH:
                        rotated_icon = self.south[self.env.vehicles[agent_idx].icon_type]
                    else:
                        rotated_icon = self.east[self.env.vehicles[agent_idx].icon_type]
                    self.env.vehicles[agent_idx].curr_icon = rotated_icon
                    # self.curr_icon = rotated_icon
                    # self.screen.blit(rotated_icon, cell_coords)
                    if self.env.vehicles[agent_idx].curr_head:
                        self.env.vehicles[agent_idx].last_head = self.env.vehicles[agent_idx].curr_head
                    self.env.vehicles[agent_idx].curr_head = [x, y]
            else:
                for idx in range(len(self.env.vehicles)):
                    if counter % self.speed_vehicle[idx] == 0:
                        if (x, y) == self.env.vehicles[idx].head:
                            if self.env.vehicles[idx].direction == SnakeDirection.NORTH:
                                rotated_icon = self.north[self.env.vehicles[idx].icon_type]
                            elif self.env.vehicles[idx].direction == SnakeDirection.WEST:
                                rotated_icon = self.west[self.env.vehicles[idx].icon_type]
                            elif self.env.vehicles[idx].direction == SnakeDirection.SOUTH:
                                rotated_icon = self.south[self.env.vehicles[idx].icon_type]
                            else:
                                rotated_icon = self.east[self.env.vehicles[idx].icon_type]
                            self.env.vehicles[idx].curr_icon = rotated_icon

                            if self.env.vehicles[idx].curr_head:
                                self.env.vehicles[idx].last_head = self.env.vehicles[idx].curr_head
                            self.env.vehicles[idx].curr_head = [x, y]

                for idx in range(len(self.env.pedestrians)):
                    if counter % self.speed_pedestrian[idx] == 0:
                        if (x, y) == self.env.pedestrians[idx].head:

                            if self.env.pedestrians[idx].curr_head:
                                self.env.pedestrians[idx].last_head = self.env.pedestrians[idx].curr_head
                            self.env.pedestrians[idx].curr_head = [x, y]

                            if not self.env.pedestrians[idx].lock:
                                self.env.pedestrians[idx].curr_icon = self.set_icon_scheme_pedestrian(idx)
                    # else:

        # if self.env.field[x, y] == CellType.SNAKE_HEAD:
        #     # pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)
        #     if self.env.snake.direction == SnakeDirection.NORTH:
        #         rotated_icon = self.north
        #     elif self.env.snake.direction == SnakeDirection.WEST:
        #         rotated_icon = self.west
        #     elif self.env.snake.direction == SnakeDirection.SOUTH:
        #         rotated_icon = self.south
        #     else:
        #         rotated_icon = self.east
        #     self.curr_icon = rotated_icon
        #     # self.screen.blit(rotated_icon, cell_coords)
        #     self.last_head = self.curr_head
        #     self.curr_head = [x, y]
        # if self.env.field[x, y]  == CellType.WALL:
        #     pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)
        # self.screen.blit(self.wall_icon, cell_coords)
        # else:
        #     pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)
        #     color = Colors.CELL_TYPE[self.env.field[x, y]]
        #     pygame.draw.rect(self.screen, color, cell_coords, 1)
        #     # internal_padding = self.CELL_SIZE // 6 * 2
        #     internal_square_coords = cell_coords.inflate((-self.internal_padding, -self.internal_padding))
        #     pygame.draw.rect(self.screen, color, internal_square_coords)

    def render(self, agent_idx=None, counter=None, head_update=False):
        """ Draw the entire game frame. """
        if not agent_idx:
            self.screen.fill(Colors.SCREEN_BACKGROUND)
            self.screen.blit(self.surface, (0, 0))

        num_font = pygame.font.Font("fonts/gyparody_tf.ttf", int(24 * (self.CELL_SIZE / 40.0)))

        text = "Traffic violation"
        # print(time_elapsed)
        disp_text = self.text_font.render(text, True, (0, 0, 0))
        self.screen.blit(disp_text, (self.screen_size[0] - 6.45 * self.CELL_SIZE, self.CELL_SIZE * 3 + 70))

        self.screen.blit(self.single_red, (self.screen_size[0] - 6 * self.CELL_SIZE, self.CELL_SIZE * 4.5 + 70))
        self.screen.blit(self.west[0], (self.screen_size[0] - 5 * self.CELL_SIZE, self.CELL_SIZE * 4.5 + 60))
        disp_num = num_font.render("-5  /", True, (0, 0, 0))
        self.screen.blit(disp_num, (self.screen_size[0] - 3.5 * self.CELL_SIZE, self.CELL_SIZE * 4.5 + 70))
        self.screen.blit(self.triangle_icon, (self.screen_size[0] - 2 * self.CELL_SIZE, self.CELL_SIZE * 4.5 + 70))

        self.screen.blit(self.single_red, (self.screen_size[0] - 6 * self.CELL_SIZE, self.CELL_SIZE * 6 + 70))
        self.screen.blit(self.man1_icon, (self.screen_size[0] - 5 * self.CELL_SIZE, self.CELL_SIZE * 6 + 55))
        disp_num = num_font.render("-5  /", True, (0, 0, 0))
        self.screen.blit(disp_num, (self.screen_size[0] - 3.5 * self.CELL_SIZE, self.CELL_SIZE * 6 + 70))
        self.screen.blit(self.circle_icon, (self.screen_size[0] - 2 * self.CELL_SIZE, self.CELL_SIZE * 6 + 70))

        text = "Accident"
        disp_text = self.text_font.render(text, True, (0, 0, 0))
        self.screen.blit(disp_text, (self.screen_size[0] - 6.45 * self.CELL_SIZE, self.CELL_SIZE * 8.5 + 70))

        accident_icon = pygame.transform.scale(self.accident_icon,
                                               tuple([int(1.2 * x) for x in self.accident_icon.get_size()]))
        self.screen.blit(accident_icon, (self.screen_size[0] - 6.3 * self.CELL_SIZE, self.CELL_SIZE * 10 + 40))
        self.screen.blit(self.west[1], (self.screen_size[0] - 6 * self.CELL_SIZE, self.CELL_SIZE * 10 + 70))
        self.screen.blit(self.west[0], (self.screen_size[0] - 5.4 * self.CELL_SIZE, self.CELL_SIZE * 10 + 70))
        self.screen.blit(self.square_icon, (self.screen_size[0] - 2 * self.CELL_SIZE, self.CELL_SIZE * 10 + 80))
        disp_num = num_font.render("-10  /", True, (0, 0, 0))
        self.screen.blit(disp_num, (self.screen_size[0] - 3.5 * self.CELL_SIZE, self.CELL_SIZE * 10 + 80))


        accident_icon = pygame.transform.scale(self.accident_icon,
                                               tuple([int(1.2 * x) for x in self.accident_icon.get_size()]))
        self.screen.blit(accident_icon, (self.screen_size[0] - 6.3 * self.CELL_SIZE, self.CELL_SIZE * 12 + 40))
        self.screen.blit(self.man1_icon_west, (self.screen_size[0] - 6.3 * self.CELL_SIZE, self.CELL_SIZE * 12 + 65))
        self.screen.blit(self.west[0], (self.screen_size[0] - 5.5 * self.CELL_SIZE, self.CELL_SIZE * 12 + 70))
        disp_num = num_font.render("-10  /", True, (0, 0, 0))
        self.screen.blit(disp_num, (self.screen_size[0] - 3.5 * self.CELL_SIZE, self.CELL_SIZE * 12 + 80))
        self.screen.blit(self.cross_icon, (self.screen_size[0] - 2 * self.CELL_SIZE, self.CELL_SIZE * 12 + 80))

        # disp_num = num_font.render(disp_num, True, (0, 0, 0))

        # icon_list = [self.good_fruit_icon, self.bad_fruit_icon, self.lava_icon]
        # text_fields = ["+" + str(self.env.rewards['good_fruit']), str(self.env.rewards['bad_fruit']),
        #                str(self.env.rewards['lava'])]
        #
        # for i in range(len(icon_list)):
        #     icon_list[i] = pygame.transform.scale(icon_list[i], tuple([int(0.8 * x) for x in icon_list[i].get_size()]))
        #
        # if self.collaborating_agent is None:
        #     text_fields = ["+" + str(self.env.rewards['good_fruit']), str(self.env.rewards['bad_fruit']),
        #                    str(self.env.rewards['lava'])]
        #
        # ct = 0
        # for text in text_fields:
        #     ct += 1
        #     disp_num = num_font.render(text, True, (0, 0, 0))
        #     self.screen.blit(disp_num, ((self.FIELD_SIZE + 3.0) * self.CELL_SIZE - disp_num.get_width(),
        #                                 self.screen_size[1] // 2 - (4.5 - ct) * self.screen_size[1] // 10))
        #     self.screen.blit(icon_list[ct - 1], ((self.FIELD_SIZE + 0.5) * self.CELL_SIZE,
        #                                          self.screen_size[1] // 2 - (4.5 - ct) * self.screen_size[1] // 10))

        for x in range(self.env.field.size):
            for y in range(self.env.field.size):
                self.render_cell(x, y, agent_idx, counter, head_update=head_update)

    def transition_animation(self, x_v, y_v, x0_v, y0_v, x_p, y_p, x0_p, y0_p, interpolate_idx, counter):

        # imm_coords.inflate_ip(-5, -5)  # Shrinks the rect by 10 pixels on both width and height
        # imm_coords.move_ip(1, 0.8)  # Moves the rect by (5, 5) pixels
        # pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, imm_coords)
        #     if imm_coords_collaborator is not None:
        #         pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, imm_coords_collaborator)

        # # restore icon from last frame if crashed
        # if interpolate_idx < self.intermediate_frames / 3 * 2:
        #     if reward == self.env.rewards['good_fruit']:  # good fruit
        #         cell_coords = pygame.Rect(
        #             x_v[i] * self.CELL_SIZE + 5,
        #             y_v[i] * self.CELL_SIZE + 5,
        #             self.CELL_SIZE,
        #             self.CELL_SIZE,
        #         )
        #         self.screen.blit(self.good_fruit_icon, cell_coords)
        #     elif reward == self.env.rewards['bad_fruit']:
        #         cell_coords = pygame.Rect(
        #             x_v[i] * self.CELL_SIZE,
        #             y_v[i] * self.CELL_SIZE,
        #             self.CELL_SIZE,
        #             self.CELL_SIZE,
        #         )
        #         self.screen.blit(self.bad_fruit_icon, cell_coords)
        #     elif reward == self.env.rewards['lava']:
        #
        #         cell_coords = pygame.Rect(
        #             x_v[i] * self.CELL_SIZE,
        #             y_v[i] * self.CELL_SIZE,
        #             self.CELL_SIZE,
        #             self.CELL_SIZE,
        #         )
        #         self.screen.blit(self.lava_icon, cell_coords)
        # else:
        #     if reward == self.env.rewards['good_fruit']:
        #         cell_coords = pygame.Rect(
        #             x_v[i] * self.CELL_SIZE + self.CELL_SIZE // 2 - self.CELL_SIZE // 6,
        #             y_v[i] * self.CELL_SIZE - 10,
        #             self.CELL_SIZE,
        #             self.CELL_SIZE,
        #         )
        #         self.screen.blit(self.reward_icon, cell_coords)
        #     elif reward == self.env.rewards['bad_fruit']:
        #         cell_coords = pygame.Rect(
        #             x_v[i] * self.CELL_SIZE + 5,
        #             y_v[i] * self.CELL_SIZE + 25,
        #             self.CELL_SIZE * 2 // 3,
        #             self.CELL_SIZE * 2 // 3,
        #         )
        #         self.screen.blit(self.small_crash_icon, cell_coords)
        #     elif reward == self.env.rewards['lava']:
        #         cell_coords = pygame.Rect(
        #             x_v[i] * self.CELL_SIZE,
        #             y_v[i] * self.CELL_SIZE + 20,
        #             self.CELL_SIZE,
        #             self.CELL_SIZE * 2 // 3,
        #         )
        #         self.screen.blit(self.big_crash_icon, cell_coords)

        # imm_coords = pygame.Rect(
        #     x0 * self.CELL_SIZE + (x - x0) * interpolate_idx * self.CELL_SIZE // self.intermediate_frames,
        #     y0 * self.CELL_SIZE + (y - y0) * interpolate_idx * self.CELL_SIZE // self.intermediate_frames,
        #     self.CELL_SIZE,
        #     self.CELL_SIZE,
        # )
        #
        # self.screen.blit(curr_icon, imm_coords)

        # # restore icon from last frame if crashed
        # if interpolate_idx < self.intermediate_frames / 3 * 2:
        #     if reward == self.env.rewards['good_fruit']:  # good fruit
        #         cell_coords = pygame.Rect(
        #             x * self.CELL_SIZE + 5,
        #             y * self.CELL_SIZE + 5,
        #             self.CELL_SIZE,
        #             self.CELL_SIZE,
        #         )
        #         self.screen.blit(self.good_fruit_icon, cell_coords)
        #     elif reward == self.env.rewards['bad_fruit']:
        #         cell_coords = pygame.Rect(
        #             x * self.CELL_SIZE,
        #             y * self.CELL_SIZE,
        #             self.CELL_SIZE,
        #             self.CELL_SIZE,
        #         )
        #         self.screen.blit(self.bad_fruit_icon, cell_coords)
        #     elif reward == self.env.rewards['lava']:
        #
        #         cell_coords = pygame.Rect(
        #             x * self.CELL_SIZE,
        #             y * self.CELL_SIZE,
        #             self.CELL_SIZE,
        #             self.CELL_SIZE,
        #         )
        #         self.screen.blit(self.lava_icon, cell_coords)
        # else:
        #     if reward == self.env.rewards['good_fruit']:
        #         cell_coords = pygame.Rect(
        #             x * self.CELL_SIZE + self.CELL_SIZE // 2 - self.CELL_SIZE // 6,
        #             y * self.CELL_SIZE - 10,
        #             self.CELL_SIZE,
        #             self.CELL_SIZE,
        #         )
        #         self.screen.blit(self.reward_icon, cell_coords)
        #     elif reward == self.env.rewards['bad_fruit']:
        #         cell_coords = pygame.Rect(
        #             x * self.CELL_SIZE + 5,
        #             y * self.CELL_SIZE + 25,
        #             self.CELL_SIZE * 2 // 3,
        #             self.CELL_SIZE * 2 // 3,
        #         )
        #         self.screen.blit(self.small_crash_icon, cell_coords)
        #     elif reward == self.env.rewards['lava']:
        #         cell_coords = pygame.Rect(
        #             x * self.CELL_SIZE,
        #             y * self.CELL_SIZE + 20,
        #             self.CELL_SIZE,
        #             self.CELL_SIZE * 2 // 3,
        #         )
        #         self.screen.blit(self.big_crash_icon, cell_coords)

        # vehicles_coords = []
        for i in range(len(self.env.vehicles)):
            if self.env.vehicles[i].lock:
                if self.env.vehicles[i].direction == SnakeDirection.NORTH:
                    new_coords = self.env.vehicles[i].lock_pos.move(-20, -70)
                elif self.env.vehicles[i].direction == SnakeDirection.WEST:
                    new_coords = self.env.vehicles[i].lock_pos.move(-50, -10)
                elif self.env.vehicles[i].direction == SnakeDirection.SOUTH:
                    new_coords = self.env.vehicles[i].lock_pos.move(10, 50)
                elif self.env.vehicles[i].direction == SnakeDirection.EAST:
                    new_coords = self.env.vehicles[i].lock_pos.move(50, -10)

                # if interpolate_idx % 60 < 30:
                if interpolate_idx % 20 < 10:
                    self.screen.blit(self.accident_icon, new_coords)
                if (interpolate_idx == 1) and (self.trialends_countdown_rounds == 7):
                    self.reward_vehicle = self.env.rewards["accident"]
                    self.accident_scoreboard = -10
                    self.accident_type = 'vehicle_accident'
                    self.timestep_result = self.env.timestep_vehicle(agent_index=i, rwd=self.reward_vehicle)
                    # self.render_scoreboard(self.env.stats.sum_episode_rewards, pygame.time.get_ticks() - self.pause_time,
                    #                        self.accident_scoreboard + self.violation_scoreboard)
                    self.dualsense.start_rumble_thread(self.dualsense.single_rumble, 0.5)

        pedestrians_coords = []
        # if counter % 2 == 0:
        for i in range(len(self.env.pedestrians)):
            if not self.env.pedestrians[i].lock:
                # if counter % self.speed_pedestrian[i] == 0:
                #     imm_coords = pygame.Rect(
                #         5 + x0_p[i] * self.CELL_SIZE + (x_p[i] - x0_p[i]) * interpolate_idx * self.CELL_SIZE // (self.intermediate_frames * 2),
                #         5 + y0_p[i] * self.CELL_SIZE + (y_p[i] - y0_p[i]) * interpolate_idx * self.CELL_SIZE // (self.intermediate_frames * 2),
                #         self.CELL_SIZE,
                #         self.CELL_SIZE,
                #     )
                # else:
                imm_coords = pygame.Rect(
                    5 + x0_p[i] * self.CELL_SIZE + (counter % self.speed_pedestrian[i]) * (x_p[i] - x0_p[i]) * self.CELL_SIZE // self.speed_pedestrian[i] + (x_p[i] - x0_p[i]) * interpolate_idx * self.CELL_SIZE // (self.intermediate_frames * self.speed_pedestrian[i]),
                    5 + y0_p[i] * self.CELL_SIZE + (counter % self.speed_pedestrian[i]) * (y_p[i] - y0_p[i]) * self.CELL_SIZE // self.speed_pedestrian[i] + (y_p[i] - y0_p[i]) * interpolate_idx * self.CELL_SIZE // (self.intermediate_frames * self.speed_pedestrian[i]),
                    self.CELL_SIZE,
                    self.CELL_SIZE,
                )
            else:
                self.accident_type = 'pedestrian_accident'  # to overwrite the vehicle accident if exists
                imm_coords = self.env.pedestrians[i].lock_pos

            # if self.visual_feedback_p == i:
            #     if interpolate_idx % 30 < 15:
            #         self.screen.blit(self.warning_icon, imm_coords.move(-80, 0))

            self.screen.blit(self.env.pedestrians[i].curr_icon, imm_coords)

            if self.env.pedestrians[i].lock:
                color = Colors.RED
            else:
                color = Colors.GREEN
            pygame.draw.rect(self.screen, color, imm_coords, width=5)
            font = pygame.font.Font(None, 28)
            name = self.human_schemes[i]
            name_surface = font.render(name, True, (255, 255, 255))
            self.screen.blit(name_surface, imm_coords.move(5, 5))
            if self.keep_flag[i] or self.env.pedestrians[i].lock:
                speed = '0'
            else:
                speed = str(int(40 / self.speed_pedestrian[i]))
            speed_surface = font.render(speed +' km/h', True, (255, 255, 255))
            self.screen.blit(speed_surface, imm_coords.move(5, 60))

            pedestrians_coords.append(imm_coords)
        # else:
        #     for i in range(len(self.env.pedestrians)):
        #         if not self.env.pedestrians[i].lock:
        #             imm_coords = pygame.Rect(
        #                 5 + x0_p[i] * self.CELL_SIZE + (x_p[i] - x0_p[i]) * self.CELL_SIZE // 2 + (
        #                         x_p[i] - x0_p[i]) * interpolate_idx // 2 * self.CELL_SIZE // (
        #                     self.intermediate_frames),
        #                 5 + y0_p[i] * self.CELL_SIZE + (y_p[i] - y0_p[i]) * self.CELL_SIZE // 2 + (
        #                         y_p[i] - y0_p[i]) * interpolate_idx // 2 * self.CELL_SIZE // (
        #                     self.intermediate_frames),
        #                 self.CELL_SIZE,
        #                 self.CELL_SIZE,
        #             )
        #         else:
        #             imm_coords = self.env.pedestrians[i].lock_pos
        #         if self.visual_feedback_p == i:
        #             if interpolate_idx % 30 < 15:
        #                 self.screen.blit(self.warning_icon, imm_coords.move(-80, 0))
        #         # try:
        #         self.screen.blit(self.env.pedestrians[i].curr_icon, imm_coords)
        #         # except TypeError:
        #         #     print("Error")
        #         pedestrians_coords.append(imm_coords)

        vehicles_coords = []
        for i in range(len(self.env.vehicles)):
            if not self.env.vehicles[i].lock:
                # if counter % self.speed_vehicle[i] == 0:
                #     imm_coords = pygame.Rect(
                #         x0_v[i] * self.CELL_SIZE + (x_v[i] - x0_v[i]) * interpolate_idx * self.CELL_SIZE // (self.intermediate_frames * self.speed_vehicle[i]),
                #         y0_v[i] * self.CELL_SIZE + (y_v[i] - y0_v[i]) * interpolate_idx * self.CELL_SIZE // (self.intermediate_frames * self.speed_vehicle[i]),
                #         self.CELL_SIZE,
                #         self.CELL_SIZE,
                #     )
                # else:
                imm_coords = pygame.Rect(
                    x0_v[i] * self.CELL_SIZE + (counter % self.speed_vehicle[i]) * (x_v[i] - x0_v[i]) * self.CELL_SIZE // self.speed_vehicle[i] + (x_v[i] - x0_v[i]) * interpolate_idx * self.CELL_SIZE // (self.intermediate_frames * self.speed_vehicle[i]),
                    y0_v[i] * self.CELL_SIZE + (counter % self.speed_vehicle[i]) * (y_v[i] - y0_v[i]) * self.CELL_SIZE // self.speed_vehicle[i] + (y_v[i] - y0_v[i]) * interpolate_idx * self.CELL_SIZE // (self.intermediate_frames * self.speed_vehicle[i]),
                    self.CELL_SIZE,
                    self.CELL_SIZE,
                )
            else:
                imm_coords = self.env.vehicles[i].lock_pos

            # if self.visual_feedback_v == i:
            #     if interpolate_idx % 30 < 15:
            #         self.screen.blit(self.warning_icon, imm_coords.move(-80, 0))
            self.screen.blit(self.env.vehicles[i].curr_icon, imm_coords)

            if self.speed_vehicle[i] == 1:
                color = Colors.RED
            elif self.speed_vehicle[i] == 2:
                color = Colors.YELLOW
            elif self.speed_vehicle[i] == 4:
                color = Colors.GREEN
            if self.env.vehicles[i].pause:
                color = Colors.GREEN
            if self.env.vehicles[i].lock:
                color = Colors.RED

            pygame.draw.rect(self.screen, color, imm_coords, width=5)

            if self.env.vehicles[i].pause or self.env.vehicles[i].lock:
                speed = '0'
            else:
                speed = str(int(40 / self.speed_vehicle[i]))

            font = pygame.font.Font(None, 28)
            name = self.car_schemes[self.env.vehicles[i].icon_type]
            name_surface = font.render(name, True, (255, 255, 255))
            self.screen.blit(name_surface, imm_coords.move(5, 5))
            speed_surface = font.render(speed + ' km/h', True, (255, 255, 255))
            self.screen.blit(speed_surface, imm_coords.move(5, 60))

            vehicles_coords.append(imm_coords)

        if (interpolate_idx == 1) and (self.trialends_countdown_rounds == 7):
            if self.accident_type == 'vehicle_accident':
                self.outlet.push_sample(["va"])
            elif self.accident_type == 'pedestrian_accident':
                self.outlet.push_sample(["pa"])

        return vehicles_coords, pedestrians_coords

    def pickup_animation(self, x, y, reward, is_collaborator):
        for i in range(len(self.env.vehicles)):
            if reward == self.env.rewards['good_fruit']:  # coin
                cell_coords = pygame.Rect(
                    x[i] * self.CELL_SIZE + self.CELL_SIZE // 2 - self.CELL_SIZE // 6,
                    y[i] * self.CELL_SIZE - 10,
                    self.CELL_SIZE,
                    self.CELL_SIZE,
                )
                self.screen.blit(self.reward_icon, cell_coords)

            if reward == self.env.rewards['bad_fruit']:  # small crash
                cell_coords = pygame.Rect(
                    x[i] * self.CELL_SIZE + 5,
                    y[i] * self.CELL_SIZE - 5,
                    self.CELL_SIZE,
                    self.CELL_SIZE,
                )
                self.screen.blit(self.small_crash_icon, cell_coords)

            if reward == self.env.rewards['lava']:  # big crash
                cell_coords = pygame.Rect(
                    x[i] * self.CELL_SIZE,
                    y[i] * self.CELL_SIZE + self.CELL_SIZE // 3,
                    self.CELL_SIZE,
                    self.CELL_SIZE * 2 // 3,
                )
                self.screen.blit(self.big_crash_icon, cell_coords)

    def map_key_to_snake_action(self, key):
        """ Convert a keystroke to an environment action. """
        actions = [
            SnakeAction.MAINTAIN_DIRECTION,
            SnakeAction.TURN_LEFT,
            SnakeAction.MAINTAIN_DIRECTION,
            SnakeAction.TURN_RIGHT,
        ]

        key_idx = self.SNAKE_CONTROL_KEYS.index(key)
        direction_idx = ALL_SNAKE_DIRECTIONS.index(self.env.vehicle.direction)
        return np.roll(actions, -key_idx)[direction_idx]

    def quit_game(self):
        self.env.is_game_over = True
        if self.env.verbose >= 1:
            stats_csv_line = self.env.stats.to_dataframe().to_csv(header=False, index=None)
            print(stats_csv_line, file=self.env.stats_file, end='', flush=True)
        if self.env.verbose >= 2:
            print(self.env.stats, file=self.env.debug_file)
        raise QuitRequestedError

    def handle_pause(self):

        if self.pause:
            temp = pygame.time.get_ticks()

        while self.pause:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.pause = False
                        self.pause_time += pygame.time.get_ticks() - temp
                        if self.frame_num == 0:
                            pygame.mixer.music.load("sound/background2.mp3")
                            pygame.mixer.music.set_volume(0.1)
                            pygame.mixer.music.play(-1, 0.0)
                    if event.key == pygame.K_ESCAPE:
                        self.quit_game()

                if event.type == pygame.QUIT:
                    self.quit_game()

    def interact_traffic_lights_vehicles(self, action, agent_idx):

        pos = self.env.vehicles[agent_idx].head
        pos = (pos[1], pos[0])
        # print("Current Position: ", pos)
        key = next((k for k, v in traffic_light_position_mapping().items() if pos in v), None)

        # key = next((k for k, v in traffic_light_position_mapping().items() if v == pos), None)
        # print("Current Traffic Light: ", key)
        # if key:
        #     print("Current Traffic Light Status: ", self.traffic_light_status[key])
        flag = False
        if pos == (2, 5):
            if self.traffic_light_status['01'] in [11, 21, 31] and action == 0:
                flag = True, 0
        if pos == (2, 6):
            if self.traffic_light_status['01'] in [11, 21, 31]:
                flag = True, 0
        if pos == (2, 7):
            if self.traffic_light_status['01'] in [11, 12, 13]:
                flag = True, 0
        if pos == (4, 11):
            if self.traffic_light_status['02'] in [11, 21, 31] and action == 0:
                flag = True, 0
        if pos == (5, 11):
            if self.traffic_light_status['02'] in [11, 21, 31]:
                flag = True, 0
        if pos == (6, 11):
            if self.traffic_light_status['02'] in [11, 12, 13]:
                flag = True, 0
        if pos == (10, 7):
            if self.traffic_light_status['03'] in [11, 12, 13]:
                flag = True, 0
        if pos == (10, 8):
            if self.traffic_light_status['03'] in [11, 21, 31]:
                flag = True, 0
        if pos == (10, 9):
            if self.traffic_light_status['03'] in [11, 21, 31] and action == 0:
                flag = True, 0
        if pos == (6, 3):
            if self.traffic_light_status['04'] in [11, 12, 13]:
                flag = True, 0
        if pos == (7, 3):
            if self.traffic_light_status['04'] in [11, 21, 31]:
                flag = True, 0
        if pos == (8, 3):
            if self.traffic_light_status['04'] in [11, 21, 31] and action == 0:
                flag = True, 0
        if pos == (4, 18):
            if self.traffic_light_status['09'] in [11, 21, 31] and action == 0:
                flag = True, 0
        if pos == (5, 18):
            if self.traffic_light_status['09'] in [11, 21, 31]:
                flag = True, 0
        if pos == (6, 18):
            if self.traffic_light_status['09'] in [11, 12, 13]:
                flag = True, 0
        if pos == (10, 15):
            if self.traffic_light_status['10'] == 1:
                flag = True, 0
        if pos == (10, 16):
            if self.traffic_light_status['10'] == 1 and action == 0:
                flag = True, 0
        if pos == (7, 13):
            if self.traffic_light_status['11'] == 1:
                flag = True, 0
        if pos == (8, 13):
            if self.traffic_light_status['11'] == 1:
                flag = True, 0
        if pos == (14, 6):
            if self.traffic_light_status['16'] == 1:
                flag = True, 0
        if pos == (16, 11):
            if self.traffic_light_status['17'] == 1 and action == 0:
                flag = True, 0
        if pos == (17, 11):
            if self.traffic_light_status['17'] == 1:
                flag = True, 0
        if pos == (17, 3):
            if self.traffic_light_status['18'] == 1:
                flag = True, 0
        if pos == (18, 3):
            if self.traffic_light_status['18'] == 1:
                flag = True, 0
        if pos == (16, 18):
            if self.traffic_light_status['22'] == 1 and action == 0:
                flag = True, 0
        if pos == (17, 18):
            if self.traffic_light_status['22'] == 1:
                flag = True, 0
        if pos == (20, 16):
            if self.traffic_light_status['23'] == 1 and action == 0:
                flag = True, 0
        if pos == (18, 13):
            if self.traffic_light_status['24'] == 1 and action == 0:
                flag = True, 0
        # print("Flag: ", flag)
        if flag:
            if (np.random.rand() < 0.4) and (not self.env.vehicles[agent_idx].pause):
                # print("Vehicle at {} running a red light with action {}!!!".format(pos, action))
                return False, 1
        return flag, 0


    def interact_traffic_lights_vehicles(self, action, agent_idx):

        pos = self.env.vehicles[agent_idx].head
        pos = (pos[1], pos[0])
        # print("Current Position: ", pos)
        key = next((k for k, v in traffic_light_position_mapping().items() if pos in v), None)

        # key = next((k for k, v in traffic_light_position_mapping().items() if v == pos), None)
        # print("Current Traffic Light: ", key)
        # if key:
        #     print("Current Traffic Light Status: ", self.traffic_light_status[key])
        flag = False
        if pos == (2, 5):
            if self.traffic_light_status['01'] in [11, 21, 31] and action == 0:
                flag = True, 0
        if pos == (2, 6):
            if self.traffic_light_status['01'] in [11, 21, 31]:
                flag = True, 0
        if pos == (2, 7):
            if self.traffic_light_status['01'] in [11, 12, 13]:
                flag = True, 0
        if pos == (4, 11):
            if self.traffic_light_status['02'] in [11, 21, 31] and action == 0:
                flag = True, 0
        if pos == (5, 11):
            if self.traffic_light_status['02'] in [11, 21, 31]:
                flag = True, 0
        if pos == (6, 11):
            if self.traffic_light_status['02'] in [11, 12, 13]:
                flag = True, 0
        if pos == (10, 7):
            if self.traffic_light_status['03'] in [11, 12, 13]:
                flag = True, 0
        if pos == (10, 8):
            if self.traffic_light_status['03'] in [11, 21, 31]:
                flag = True, 0
        if pos == (10, 9):
            if self.traffic_light_status['03'] in [11, 21, 31] and action == 0:
                flag = True, 0
        if pos == (6, 3):
            if self.traffic_light_status['04'] in [11, 12, 13]:
                flag = True, 0
        if pos == (7, 3):
            if self.traffic_light_status['04'] in [11, 21, 31]:
                flag = True, 0
        if pos == (8, 3):
            if self.traffic_light_status['04'] in [11, 21, 31] and action == 0:
                flag = True, 0
        if pos == (4, 18):
            if self.traffic_light_status['09'] in [11, 21, 31] and action == 0:
                flag = True, 0
        if pos == (5, 18):
            if self.traffic_light_status['09'] in [11, 21, 31]:
                flag = True, 0
        if pos == (6, 18):
            if self.traffic_light_status['09'] in [11, 12, 13]:
                flag = True, 0
        if pos == (10, 15):
            if self.traffic_light_status['10'] == 1:
                flag = True, 0
        if pos == (10, 16):
            if self.traffic_light_status['10'] == 1 and action == 0:
                flag = True, 0
        if pos == (7, 13):
            if self.traffic_light_status['11'] == 1:
                flag = True, 0
        if pos == (8, 13):
            if self.traffic_light_status['11'] == 1:
                flag = True, 0
        if pos == (14, 6):
            if self.traffic_light_status['16'] == 1:
                flag = True, 0
        if pos == (16, 11):
            if self.traffic_light_status['17'] == 1 and action == 0:
                flag = True, 0
        if pos == (17, 11):
            if self.traffic_light_status['17'] == 1:
                flag = True, 0
        if pos == (17, 3):
            if self.traffic_light_status['18'] == 1:
                flag = True, 0
        if pos == (18, 3):
            if self.traffic_light_status['18'] == 1:
                flag = True, 0
        if pos == (16, 18):
            if self.traffic_light_status['22'] == 1 and action == 0:
                flag = True, 0
        if pos == (17, 18):
            if self.traffic_light_status['22'] == 1:
                flag = True, 0
        if pos == (20, 16):
            if self.traffic_light_status['23'] == 1 and action == 0:
                flag = True, 0
        if pos == (18, 13):
            if self.traffic_light_status['24'] == 1 and action == 0:
                flag = True, 0
        # # print("Flag: ", flag)
        # if flag:
        #     if (np.random.rand() < 0.4) and (not self.env.vehicles[agent_idx].pause):
        #         # print("Vehicle at {} running a red light with action {}!!!".format(pos, action))
        #         return False, 1
        return flag, 0


    def interact_traffic_lights_pedestrians(self, action, agent_idx):

        pos = self.env.pedestrians[agent_idx].head
        pos = (pos[1], pos[0])
        # print("Current Position: ", pos)
        key = next((k for k, v in traffic_light_position_mapping().items() if pos in v), None)
        # print("Current Traffic Light: ", key)
        # if key:
        #     print("Current Traffic Light Status: ", self.traffic_light_status[key])
        flag = False
        direction = self.env.pedestrians[agent_idx].direction
        if pos == (3, 4):
            if self.traffic_light_status['05'] == 1:
                if ((direction == SnakeDirection.NORTH and action == 2) or
                        (direction == SnakeDirection.SOUTH and action == 1) or
                        (direction == SnakeDirection.EAST and action == 0)):
                    flag = True, 0
            if self.traffic_light_status['08'] == 1:
                if ((direction == SnakeDirection.WEST and action == 1) or
                        (direction == SnakeDirection.SOUTH and action == 0) or
                        (direction == SnakeDirection.EAST and action == 2)):
                    flag = True, 0
        if pos == (3, 10):
            if self.traffic_light_status['05'] == 1:
                if ((direction == SnakeDirection.NORTH and action == 1) or
                        (direction == SnakeDirection.SOUTH and action == 2) or
                        (direction == SnakeDirection.WEST and action == 0)):
                    flag = True, 0
            if self.traffic_light_status['06'] == 1:
                if ((direction == SnakeDirection.WEST and action == 1) or
                        (direction == SnakeDirection.SOUTH and action == 0) or
                        (direction == SnakeDirection.EAST and action == 2)):
                    flag = True, 0
        if pos == (9, 10):
            if self.traffic_light_status['06'] == 1:
                if ((direction == SnakeDirection.NORTH and action == 0) or
                        (direction == SnakeDirection.EAST and action == 1) or
                        (direction == SnakeDirection.WEST and action == 2)):
                    flag = True, 0
            if self.traffic_light_status['07'] == 1:
                if ((direction == SnakeDirection.WEST and action == 0) or
                        (direction == SnakeDirection.NORTH and action == 1) or
                        (direction == SnakeDirection.SOUTH and action == 2)):
                    flag = True, 0
        if pos == (9, 4):
            if self.traffic_light_status['07'] == 1:
                if ((direction == SnakeDirection.NORTH and action == 2) or
                        (direction == SnakeDirection.EAST and action == 0) or
                        (direction == SnakeDirection.SOUTH and action == 1)):
                    flag = True, 0
            if self.traffic_light_status['08'] == 1:
                if ((direction == SnakeDirection.WEST and action == 2) or
                        (direction == SnakeDirection.NORTH and action == 0) or
                        (direction == SnakeDirection.EAST and action == 1)):
                    flag = True, 0
        if pos == (3, 14):
            if self.traffic_light_status['12'] == 1:
                if ((direction == SnakeDirection.NORTH and action == 2) or
                        (direction == SnakeDirection.SOUTH and action == 1) or
                        (direction == SnakeDirection.EAST and action == 0)):
                    flag = True, 0
            if self.traffic_light_status['15'] == 1:
                if ((direction == SnakeDirection.WEST and action == 1) or
                        (direction == SnakeDirection.SOUTH and action == 0) or
                        (direction == SnakeDirection.EAST and action == 2)):
                    flag = True, 0
        if pos == (3, 17):
            if self.traffic_light_status['12'] == 1:
                if ((direction == SnakeDirection.NORTH and action == 1) or
                        (direction == SnakeDirection.SOUTH and action == 2) or
                        (direction == SnakeDirection.WEST and action == 0)):
                    flag = True, 0
            if self.traffic_light_status['13'] == 1:
                if ((direction == SnakeDirection.WEST and action == 1) or
                        (direction == SnakeDirection.SOUTH and action == 0) or
                        (direction == SnakeDirection.EAST and action == 2)):
                    flag = True, 0
        if pos == (9, 17):
            if self.traffic_light_status['13'] == 1:
                if ((direction == SnakeDirection.NORTH and action == 0) or
                        (direction == SnakeDirection.EAST and action == 1) or
                        (direction == SnakeDirection.WEST and action == 2)):
                    flag = True, 0
            if self.traffic_light_status['14'] == 1:
                if ((direction == SnakeDirection.WEST and action == 0) or
                        (direction == SnakeDirection.NORTH and action == 1) or
                        (direction == SnakeDirection.SOUTH and action == 2)):
                    flag = True, 0
        if pos == (9, 14):
            if self.traffic_light_status['14'] == 1:
                if ((direction == SnakeDirection.NORTH and action == 2) or
                        (direction == SnakeDirection.EAST and action == 0) or
                        (direction == SnakeDirection.SOUTH and action == 1)):
                    flag = True, 0
            if self.traffic_light_status['15'] == 1:
                if ((direction == SnakeDirection.WEST and action == 2) or
                        (direction == SnakeDirection.NORTH and action == 0) or
                        (direction == SnakeDirection.EAST and action == 1)):
                    flag = True, 0
        if pos == (15, 4):
            if self.traffic_light_status['19'] == 1:
                if ((direction == SnakeDirection.NORTH and action == 2) or
                        (direction == SnakeDirection.SOUTH and action == 1) or
                        (direction == SnakeDirection.EAST and action == 0)):
                    flag = True, 0
            if self.traffic_light_status['21'] == 1:
                if ((direction == SnakeDirection.WEST and action == 1) or
                        (direction == SnakeDirection.SOUTH and action == 0) or
                        (direction == SnakeDirection.EAST and action == 2)):
                    flag = True, 0
        if pos == (15, 10):
            if self.traffic_light_status['19'] == 1:
                if ((direction == SnakeDirection.NORTH and action == 1) or
                        (direction == SnakeDirection.SOUTH and action == 2) or
                        (direction == SnakeDirection.WEST and action == 0)):
                    flag = True, 0
            if self.traffic_light_status['20'] == 1:
                if ((direction == SnakeDirection.WEST and action == 1) or
                        (direction == SnakeDirection.SOUTH and action == 0) or
                        (direction == SnakeDirection.EAST and action == 2)):
                    flag = True, 0
        if pos == (19, 10):
            if self.traffic_light_status['20'] == 1:
                if ((direction == SnakeDirection.WEST and action == 2) or
                        (direction == SnakeDirection.NORTH and action == 0) or
                        (direction == SnakeDirection.EAST and action == 1)):
                    flag = True, 0
        if pos == (19, 4):
            if self.traffic_light_status['21'] == 1:
                if ((direction == SnakeDirection.WEST and action == 2) or
                        (direction == SnakeDirection.NORTH and action == 0) or
                        (direction == SnakeDirection.EAST and action == 1)):
                    flag = True, 0
        if pos == (15, 14):
            if self.traffic_light_status['25'] == 1:
                if ((direction == SnakeDirection.NORTH and action == 2) or
                        (direction == SnakeDirection.SOUTH and action == 1) or
                        (direction == SnakeDirection.EAST and action == 0)):
                    flag = True, 0
            if self.traffic_light_status['28'] == 1:
                if ((direction == SnakeDirection.WEST and action == 1) or
                        (direction == SnakeDirection.SOUTH and action == 0) or
                        (direction == SnakeDirection.EAST and action == 2)):
                    flag = True, 0
        if pos == (15, 17):
            if self.traffic_light_status['25'] == 1:
                if ((direction == SnakeDirection.NORTH and action == 1) or
                        (direction == SnakeDirection.SOUTH and action == 2) or
                        (direction == SnakeDirection.WEST and action == 0)):
                    flag = True, 0
            if self.traffic_light_status['26'] == 1:
                if ((direction == SnakeDirection.WEST and action == 1) or
                        (direction == SnakeDirection.SOUTH and action == 0) or
                        (direction == SnakeDirection.EAST and action == 2)):
                    flag = True, 0
        if pos == (19, 17):
            if self.traffic_light_status['26'] == 1:
                if ((direction == SnakeDirection.NORTH and action == 0) or
                        (direction == SnakeDirection.EAST and action == 1) or
                        (direction == SnakeDirection.WEST and action == 2)):
                    flag = True, 0
            if self.traffic_light_status['27'] == 1:
                if ((direction == SnakeDirection.WEST and action == 0) or
                        (direction == SnakeDirection.NORTH and action == 1) or
                        (direction == SnakeDirection.SOUTH and action == 2)):
                    flag = True, 0
        if pos == (19, 14):
            if self.traffic_light_status['27'] == 1:
                if ((direction == SnakeDirection.NORTH and action == 2) or
                        (direction == SnakeDirection.EAST and action == 0) or
                        (direction == SnakeDirection.SOUTH and action == 1)):
                    flag = True, 0
            if self.traffic_light_status['28'] == 1:
                if ((direction == SnakeDirection.WEST and action == 2) or
                        (direction == SnakeDirection.NORTH and action == 0) or
                        (direction == SnakeDirection.EAST and action == 1)):
                    flag = True, 0
        if flag:
            if np.random.rand() < 0.2:
                # print("Pedestrian at {} running a red light with action {}!!!".format(pos, action))
                return False, 1
        return flag, 0

    def interact_other_agents_vehicles(self, action, agent_idx):
        self.env.choose_action_multiagents(action, self.env.vehicles, agent_idx)
        next_move = self.env.vehicles[agent_idx].peek_next_move()
        if action == 1:
            self.env.choose_action_multiagents(2, self.env.vehicles, agent_idx)
        if action == 2:
            self.env.choose_action_multiagents(1, self.env.vehicles, agent_idx)
        # flag = False
        flag_v, flag_p = False, False
        for i in range(len(self.env.vehicles)):
            if i != agent_idx:
                if self.env.vehicles[i].head == next_move:
                    # print("Collision with other vehicles")
                    flag_v = True
        for i in range(len(self.env.pedestrians)):
            if i != agent_idx:
                if self.env.pedestrians[i].head == next_move:
                    # print("Collision with pedestrians")
                    flag_p = True
        if flag_v:
            if np.random.rand() < 0.1:
                # print("=====================================")
                # print("Vehicle accident will happen at {}!!!".format(next_move, action))
                # print("=====================================")
                if not self.env.vehicles[agent_idx].pause:
                    return (False, False), 1
                # stopped, not good if start for an accident
        if flag_p:
            if np.random.rand() < 0.3:
                # print("=====================================")
                # print("Pedestrian accident will happen at {}!!!".format(next_move, action))
                # print("=====================================")
                return (False, False), 1
        return (flag_v, flag_p), 0

    def validate_traffic_rule_vehicles(self, action, agent_idx):

        # print("=====================================")
        # print("Current Position: ", self.env.vehicles[agent_idx].head)
        if self.env.vehicles[agent_idx].direction == SnakeDirection.NORTH:
            # print("NORTH MAP")
            action_map = self.action_car_north
        if self.env.vehicles[agent_idx].direction == SnakeDirection.WEST:
            # print("WEST MAP")
            action_map = self.action_car_west
        if self.env.vehicles[agent_idx].direction == SnakeDirection.SOUTH:
            # print("SOUTH MAP")
            action_map = self.action_car_south
        if self.env.vehicles[agent_idx].direction == SnakeDirection.EAST:
            # print("EAST MAP")
            action_map = self.action_car_east

        current_head = self.env.vehicles[agent_idx].head
        current_head = (current_head[1], current_head[0])
        # print("Current Map", self.env.field)
        # print("Current action: ", action)
        # print("Current action map: ", action_map[current_head])
        if not action_map[current_head]:
            print("=====================================")

            raise IndexError("No action available for the current direction")
        if self.change_lane_back_vehicles[agent_idx]:
            self.change_lane_back_vehicles[agent_idx] = None
            # print("Last action is change lane, cannot change lane again")
            # print("=====================================")

            return 0, 0
        elif action not in action_map[current_head]:
            # choose a random action
            resampled_action = np.random.choice(action_map[current_head])
            # print("Resampled action: ", resampled_action)
            # print("=====================================")

            if self.is_in_the_lane(agent_idx) and resampled_action != 0:
                if np.random.uniform(0, 1) < 0.2:
                    # print("I MUST CHANGE LANE! I AM LUCKY")
                    # print("=====================================")

                    return resampled_action, 1
                else:
                    # print("=====================================")
                    return 0, 0
            return resampled_action, 0
        elif self.is_in_the_lane(agent_idx) and action != 0:
            if np.random.uniform(0, 1) < 0.2:
                # print("Using Action: {}, I MUST CHANGE LANE! I AM LUCKY".format(action))
                # print("=====================================")
                return action, 1
            else:
                # print("Using Action: 0, FAILED TO CHANGE LANE!")
                # print("=====================================")
                return 0, 0
        else:
            # print("Original Sampled Action: {} is valid, conduct".format(action))
            # print("=====================================")
            return action, 0

    def validate_traffic_rule_pedestrians(self, action, agent_idx):
        if self.env.pedestrians[agent_idx].direction == SnakeDirection.NORTH:
            # print("NORTH MAP")
            action_map = self.action_pedestrian_north
        if self.env.pedestrians[agent_idx].direction == SnakeDirection.WEST:
            # print("WEST MAP")
            action_map = self.action_pedestrian_west
        if self.env.pedestrians[agent_idx].direction == SnakeDirection.SOUTH:
            # print("SOUTH MAP")
            action_map = self.action_pedestrian_south
        if self.env.pedestrians[agent_idx].direction == SnakeDirection.EAST:
            # print("EAST MAP")
            action_map = self.action_pedestrian_east

        current_head = self.env.pedestrians[agent_idx].head
        current_head = (current_head[1], current_head[0])
        # print("Current Map", self.env.field)
        # print("Current action: ", action)
        # print("Current action map: ", action_map[current_head])
        if not action_map[current_head]:
            print("=====================================")

            raise IndexError("No action available for the current direction")

        elif action not in action_map[current_head]:
            # choose a random action
            resampled_action = np.random.choice(action_map[current_head])
            # print("Resampled action: ", resampled_action)
            # print("=====================================")
            return resampled_action
        else:
            # print("Original Sampled Action: {} is valid, conduct".format(action))
            # print("=====================================")
            return action

    def is_in_the_lane(self, agent_idx):
        lane_area = [((4, 1), (8, 3)),
                     ((1, 5), (2, 9)),
                     ((4, 11), (8, 13)),
                     ((4, 18), (8, 21)),
                     ((1, 15), (2, 16)),
                     ((10, 5), (14, 9)),
                     ((10, 15), (14, 16)),
                     ((16, 1), (18, 3)),
                     ((16, 11), (18, 13)),
                     ((16, 18), (18, 21)),
                     ((15, 20), (16, 21)),
                     ]
        current_head = self.env.vehicles[agent_idx].head
        current_head = (current_head[1], current_head[0])

        for lane in lane_area:
            if (lane[0][0] <= current_head[0] <= lane[1][0] and
                    lane[0][1] <= current_head[1] <= lane[1][1]):
                return True
        return False

    def is_on_the_zebra_crossing(self, agent_idx):
        lane_area = [((4, 4), (8, 4)),
                     ((4, 10), (8, 10)),
                     ((3, 5), (3, 9)),
                     ((9, 5), (9, 9)),

                     ((15, 5), (15, 9)),
                     ((16, 4), (18, 4)),
                     ((16, 10), (18, 10)),

                     ((15, 15), (15, 16)),
                     ((19, 15), (19, 16)),
                     ((16, 14), (18, 14)),
                     ((16, 17), (18, 17)),

                     ((3, 15), (3, 16)),
                     ((9, 15), (9, 16)),
                     ((4, 14), (8, 14)),
                     ((4, 17), (8, 17)),
                     ]
        current_head = self.env.vehicles[agent_idx].head
        current_head = (current_head[1], current_head[0])

        for lane in lane_area:
            if lane[0][0] <= current_head[0] <= lane[1][0] and lane[0][1] <= current_head[1] <= lane[1][1]:
                for i in range(len(self.env.pedestrians)):
                    p_head = self.env.pedestrians[i].head
                    p_head = (p_head[1], p_head[0])
                    if lane[0][0] <= p_head[0] <= lane[1][0] and lane[0][1] <= p_head[1] <= lane[1][1]:
                        return True
        return False

    def is_change_lane(self, action, agent_idx):
        if action != 0:
            # check if the next head is in the lane area
            if self.is_in_the_lane(agent_idx):
                if action == SnakeAction.TURN_LEFT:
                    return SnakeAction.TURN_RIGHT
                else:
                    return SnakeAction.TURN_LEFT
            # print("Not in the lane area, no need to change lane")
        else:
            # print("action is 0, no need to change lane")
            return None

    # async def agent_generation(self, snake):
    #     for interpolate_idx in range(4):
    #         cell_coords = pygame.Rect(
    #             snake.head[0] * self.CELL_SIZE,
    #             snake.head[1] * self.CELL_SIZE,
    #             self.CELL_SIZE,
    #             self.CELL_SIZE,
    #         )
    #         self.screen.blit(self.spawn_icon, cell_coords)
    #
    #         self.render_scoreboard(0, pygame.time.get_ticks() - self.pause_time, 0)
    #         pygame.display.update()
    #     print("Agent generation completed")
    #
    # async def agent_generation_main(self):
    #
    #     tasks = [self.agent_generation(snake) for snake in self.env.vehicles]
    #
    #     await asyncio.gather(*tasks)

    def agent_generation(self, snake):
        for interpolate_idx in range(10):
            cell_coords = pygame.Rect(
                snake.head[0] * self.CELL_SIZE,
                snake.head[1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE,
            )
            self.screen.blit(self.spawn_icon, cell_coords)

            self.render_scoreboard(self.env.stats.sum_episode_rewards, pygame.time.get_ticks() - self.pause_time,
                                   self.accident_scoreboard + self.violation_scoreboard)
            pygame.display.update()
        # print("Agent generation completed")

    def run(self, num_episodes=1, participant='test'):

        """ Run the GUI player for the specified number of episodes. """
        pygame.display.update()
        self.fps_clock = pygame.time.Clock()

        capture_thread1 = captureThread(0, participant=participant, data_dir='./user_study_data/', exp_id='original',
                                        test=self.test)
        capture_thread1.start()
        try:
            for episode in range(num_episodes):
                self.run_episode()
                pygame.time.wait(1500)
            capture_thread1.stop()
        except QuitRequestedError:
            capture_thread1.stop()

    def run_episode(self):
        """ Run the GUI player for a single episode. """
        self.outlet.push_sample(["TS"])
        global frame_ct
        # Initialize the environment.
        self.timestep_watch.reset()
        self.timestep_result = self.env.new_episode()
        # timestep_result_collaborator = timestep_result
        self.agent.begin_episode()


        self.base_update_interval = 8  # 1x speed  5km/h
        self.slow_update_interval = 4  # 1.5x speed  10km/h
        self.normal_update_interval = 2  # 2x speed  20km/h
        self.fast_update_interval = 1  # 4x speed  40km/h

        self.speed_vehicle = [self.slow_update_interval] * len(self.env.vehicles)
        self.speed_pedestrian = [self.base_update_interval] * len(self.env.pedestrians)
        self.change_lane_back_vehicles = [None] * len(self.env.vehicles)
        self.keep_action = [None] * len(self.env.pedestrians)
        self.keep_flag = [False] * len(self.env.pedestrians)

        is_human_agent = isinstance(self.agent, HumanAgent)
        self.timestep_delay = self.HUMAN_TIMESTEP_DELAY if is_human_agent else self.AI_TIMESTEP_DELAY

        """ Select car scheme first """

        # self.render()
        self.render_initial_traffic_lights()
        start_text_font = pygame.font.Font("fonts/gyparody_hv.ttf", int(22 * (self.CELL_SIZE / 40.0)))
        disp_text = start_text_font.render("Press <Space> to Start", True, (220, 220, 220))
        # self.screen.blit(disp_text, (self.screen_size[0] // 2 - disp_text.get_width() // 2 , self.screen_size[1] // 2 - disp_text.get_height()//2 ))
        self.screen.blit(disp_text, (self.CELL_SIZE * 22 + 40, self.CELL_SIZE * 20))
        pygame.display.update()

        # Main game loop.
        self.running = True
        action_selected = False
        episode_start = True
        counter = 0

        self.trialends_countdown = False
        self.trialends_countdown_rounds = 8
        violation_lock_countdown_rounds = 7
        visual_feedback_countdown_rounds_v = 3
        visual_feedback_countdown_rounds_p = 3
        self.accident_type = None
        while self.running:
            if self.visual_feedback_v is not None:
                visual_feedback_countdown_rounds_v -= 1
                # print("Visual feedback countdown: ", visual_feedback_countdown_rounds_v)
            if visual_feedback_countdown_rounds_v == 0:
                self.visual_feedback_v = None
                visual_feedback_countdown_rounds_v = 3

            if self.visual_feedback_p is not None:
                visual_feedback_countdown_rounds_p -= 1
                # print("Visual feedback countdown: ", visual_feedback_countdown_rounds_p)
            if visual_feedback_countdown_rounds_p == 0:
                self.visual_feedback_p = None
                visual_feedback_countdown_rounds_p = 3

            self.violation_scoreboard = 0
            self.accident_scoreboard = 0
            self.reward_vehicle = 0
            self.reward_pedestrian = 0
            # print("Accident type: ", self.accident_type)

            if self.event_lock:
                violation_lock_countdown_rounds -= 1
                print("Violation lock countdown: ", violation_lock_countdown_rounds)
                print("Accident type: ", self.accident_type)
            if violation_lock_countdown_rounds == 0:
                self.event_lock = False
                violation_lock_countdown_rounds = 7
                self.accident_type = None
                self.button_pressed = False
                self.button_pressed_value = None
            frame_ct = self.frame_num
            if not action_selected:
                action = SnakeAction.MAINTAIN_DIRECTION

                # Handle events.
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.pause = True

                    if event.key == pygame.K_ESCAPE:
                        self.quit_game()

                if event.type == pygame.QUIT:
                    self.quit_game()

            self.handle_pause()

            # Initial sound
            if self.frame_num == 0 and PLAY_SOUND:
                self.begin_sound.play()

            if episode_start:
                # asyncio.run(self.agent_generation_main())
                for vehicle in self.env.vehicles: self.agent_generation(vehicle)
                for pedestrian in self.env.pedestrians: self.agent_generation(pedestrian)
                episode_start = False
            elif (pygame.time.get_ticks() - self.pause_time) % 3 == 0 and len(self.env.vehicles) < 15:  # every 5 seconds generate a new agent

                if counter % self.slow_update_interval == 0:
                    self.env.new_agent_vehicle()
                    self.change_lane_back_vehicles.append(None)
                # if len(self.env.vehicles) % 3 == 0:
                #     self.speed_vehicle.append(self.normal_update_interval)

                # if len(self.env.vehicles) % 3 == 0:
                #     self.speed_vehicle.append(self.normal_update_interval)
                # else:
                self.speed_vehicle.append(self.slow_update_interval)

                if (counter % self.base_update_interval == 0) and len(self.env.pedestrians) < 6:
                    self.env.new_agent_pedestrian()
                    self.keep_action.append(None)
                    self.keep_flag.append(False)
                    self.speed_pedestrian.append(self.base_update_interval)

            # try to switch the speed of some vehicles during movement
            if counter % 4 == 0:  # make sure not the in the middle of the movement
                speed, agent_idx = np.random.choice([1, 4]), np.random.choice(range(len(self.env.vehicles)))
                old_speed = self.speed_vehicle[agent_idx]
                if old_speed == 1 or old_speed == 4:
                    self.speed_vehicle[agent_idx] = 2
                    print("Vehicle {} speed changed from {} to {}".format(agent_idx, old_speed, 2))
                else:
                    self.speed_vehicle[agent_idx] = speed
                    print("Vehicle {} speed changed from {} to {}".format(agent_idx, old_speed, speed))

            # if self.last_head == [0, 0]:  # generation of the agent icon
            # for interpolate_idx in range(4):
            #     cell_coords = pygame.Rect(
            #         self.env.snake.head[0] * self.CELL_SIZE,
            #         self.env.snake.head[1] * self.CELL_SIZE,
            #         self.CELL_SIZE,
            #         self.CELL_SIZE,
            #     )
            #     self.screen.blit(self.spawn_icon, cell_coords)
            #
            #     self.render_scoreboard(0, pygame.time.get_ticks() - self.pause_time, 0)
            #     pygame.display.update()

            # self.fps_clock.tick(self.intermediate_frames+5)
            # pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)

            # Update game state.
            timestep_timed_out = self.timestep_watch.time() >= self.timestep_delay
            # human_made_move = is_human_agent and action != SnakeAction.MAINTAIN_DIRECTION

            if timestep_timed_out:
                # print("Timestep timed out")
                self.timestep_watch.reset()
                self.frame_num = self.frame_num + 1
                self.sound_played = False

                agent2pop_vehicle = []
                for agent_idx in range(len(self.env.vehicles)):
                    if not self.env.vehicles[agent_idx].lock:
                        if counter % self.speed_vehicle[agent_idx] == 0:
                            action = self.agent.act(self.timestep_result.observation, self.timestep_result.reward)
                            action, switch_lane_by_force = self.validate_traffic_rule_vehicles(action, agent_idx)

                            wait4greenlight, redlight_force = self.interact_traffic_lights_vehicles(action, agent_idx)
                            collide_w_agents, accident_force = self.interact_other_agents_vehicles(action, agent_idx)
                            if accident_force or redlight_force:
                                if not self.event_lock and not self.trialends_countdown:
                                    wait4greenlight = False
                                    self.event_lock = True
                                    print("Vehicle Violation!")
                                    self.reward_vehicle = self.env.rewards["violation"]
                                    self.violation_scoreboard = -5
                                    self.accident_type = 'vehicle_violation'  # vv

                                    self.dualsense.start_rumble_thread(self.dualsense.multiple_rumble_with_interval, 0.02, 0.1)
                                    self.outlet.push_sample(["V Vio"])
                                    self.visual_feedback_v = agent_idx

                                    # agent2collide_vehicle.append(agent_idx)
                                    # if the_other_agent[0] == 'v': agent2collide_vehicle.append(the_other_agent[1])
                                    # if the_other_agent[0] == 'p': agent2collide_pedestrian.append(the_other_agent[1])
                                    # print("Accident force: ", accident_force)
                                elif accident_force:
                                    collide_w_agents = [True, True]
                                elif redlight_force:
                                    wait4greenlight = True

                            if not wait4greenlight and not (collide_w_agents[0] or collide_w_agents[1]):
                                if ((1 and (self.speed_vehicle[agent_idx] == 1)) or switch_lane_by_force or self.is_on_the_zebra_crossing(agent_idx)) and not self.event_lock and not self.trialends_countdown:
                                    if self.speed_vehicle[agent_idx] == 1: print("FAST")
                                    if switch_lane_by_force: print("Switch lane by force")
                                    if self.is_on_the_zebra_crossing(agent_idx): print("On the sidewalk")
                                    self.event_lock = True
                                    print("Vehicle Normal!")
                                    self.dualsense.start_rumble_thread(self.dualsense.multiple_rumble_with_interval, 0.02, 0.1)
                                    self.outlet.push_sample(["V Nor"])
                                    self.visual_feedback_v = agent_idx
                                    self.accident_type = 'normal'

                                self.change_lane_back_vehicles[agent_idx] = self.is_change_lane(action, agent_idx)
                                self.env.choose_action_multiagents(action, self.env.vehicles, agent_idx)
                                action_selected = False
                                self.timestep_result = self.env.timestep_vehicle(agent_index=agent_idx, rwd=self.reward_vehicle)
                                self.reward_vehicle = self.env.rewards["timestep"]
                                if self.timestep_result.is_episode_end and self.env.stats.termination_reason in ['hit_wall',                                                                                                          'hit_own_body']:
                                    self.timestep_result.is_episode_end = False
                                    self.env.is_game_over = False
                                    agent2pop_vehicle.append(agent_idx)
                                self.env.vehicles[agent_idx].pause = False
                            else:
                                self.env.vehicles[agent_idx].pause = True
                    else:  # countdown to stop this episode
                        self.trialends_countdown = True
                    # print(len(self.env.pedestrians))
                for i in range(len(agent2pop_vehicle) - 1, -1, -1):
                    self.env.vehicles.pop(agent2pop_vehicle[i])
                    self.change_lane_back_vehicles.pop(agent2pop_vehicle[i])
                    self.speed_vehicle.pop(agent2pop_vehicle[i])

                # if counter % 2 == 0:
                    # agent2collide_pedestrian = []
                for agent_idx in range(len(self.env.pedestrians)):
                    if not self.env.pedestrians[agent_idx].lock and not self.trialends_countdown:
                        if counter % self.speed_pedestrian[agent_idx] == 0:
                            if not self.keep_flag[agent_idx]:  # if is waiting for green light, then keep the action unchanged
                                action = self.agent.act(self.timestep_result.observation, self.timestep_result.reward)
                                action = self.validate_traffic_rule_pedestrians(action, agent_idx)
                            else:
                                action = self.keep_action[agent_idx]
                                self.keep_flag[agent_idx] = False
                            wait4greenlight, redlight_force = self.interact_traffic_lights_pedestrians(action, agent_idx)

                            if (not wait4greenlight and redlight_force == 0) or (redlight_force == 1 and not self.event_lock):
                                if redlight_force == 1:
                                    self.event_lock = True
                                    print("Pedestrian Violation!")
                                    self.reward_pedestrian = self.env.rewards["violation"]
                                    self.violation_scoreboard = -5
                                    self.accident_type = 'pedestrian_violation'

                                    self.dualsense.start_rumble_thread(self.dualsense.multiple_rumble_with_interval,
                                                                       0.02, 0.1)
                                    self.outlet.push_sample(["P Vio"])
                                    self.visual_feedback_p = agent_idx

                                # if (redlight_force == 0) and (np.random.rand() < 0.03) and not self.event_lock:
                                #     self.event_lock = True
                                #     print("Pedestrian Normal!")
                                #     self.dualsense.start_rumble_thread(self.dualsense.multiple_rumble_with_interval,
                                #                                        0.02, 0.1)
                                #     self.outlet.push_sample(["P Nor"])
                                #     self.visual_feedback_p = agent_idx
                                #     self.accident_type = 'normal'

                                self.env.choose_action_multiagents(action, self.env.pedestrians, agent_idx)
                                action_selected = False
                                self.timestep_result_p = self.env.timestep_pedestrians(agent_index=agent_idx,
                                                                                       rwd=self.reward_pedestrian)
                                self.reward_pedestrian = self.env.rewards["timestep"]

                                # if self.timestep_result_p.is_episode_end and self.env.stats.termination_reason in [
                                #     'hit_wall', 'hit_own_body']:
                                #     self.timestep_result_p.is_episode_end = False
                                #     self.env.is_game_over = False
                                #     agent2collide_pedestrian.append(agent_idx)
                            else:
                                self.keep_action[agent_idx] = action
                                self.keep_flag[agent_idx] = True

                    # for i in range(len(agent2collide_pedestrian) - 1, -1, -1):
                        # self.env.pedestrians.pop(agent2collide_pedestrian[i])
                        # self.keep_action.pop(agent2collide_pedestrian[i])
                        # self.keep_flag.pop(agent2collide_pedestrian[i])

                # action = self.agent.act(timestep_result.observation, timestep_result.reward)
                # action = self.check_traffic_rule_validity(action)
                #
                # wait4greenlight = self.interact_with_traffic_lights(action)
                # if not wait4greenlight:
                #     self.change_lane_back = self.is_change_lane(action)
                #     # action = np.random.randint(0, 3)
                #     self.env.choose_action_multiagents(action)
                #
                #     action_selected = False
                #
                #     if self.agent_name == 'mixed':
                #         timestep_result = self.env.timestep(self.punch_sound, self.good_sound, self.bad_sound,
                #                                             self.very_bad_sound, self.free_sound, self.agent.curr_agent)
                #     else:
                #         timestep_result = self.env.timestep(self.punch_sound, self.good_sound, self.bad_sound,
                #                                             self.very_bad_sound, self.free_sound)

                if self.save_frames:
                    if self.agent_name == 'a2c':
                        for a in range(3):
                            qval = self.agent.get_q_value(self.timestep_result.observation, a)
                            pygame.image.save(self.screen,
                                              'screenshots/frame-%03d_%02d_%.3f.png' % (self.frame_num, a, qval))
                    else:
                        pygame.image.save(self.screen, 'screenshots/frame-%03d.png' % (self.frame_num))

                if self.trialends_countdown: self.trialends_countdown_rounds -= 1
                if self.trialends_countdown_rounds < 0: self.timestep_result.is_episode_end = True
                if self.timestep_result.is_episode_end:
                    smaller_text_font = pygame.font.Font("fonts/gyparody_hv.ttf", int(22 * (self.CELL_SIZE / 40.0)))
                    disp_text = smaller_text_font.render("Round Finished", True, (220, 220, 220))
                    # self.screen.blit(disp_text, (15 + self.screen_size[0] // 2 - disp_text.get_width() // 2 , self.screen_size[1] * 2 // 3 + disp_text.get_height() // 2 ))
                    self.screen.blit(disp_text, (self.CELL_SIZE * 23 + 50, self.CELL_SIZE * 20))
                    pygame.display.update()
                    time.sleep(3)
                    self.agent.end_episode()
                    self.running = False

            # Render.
            if self.running:
                score = self.env.stats.sum_episode_rewards
                # asyncio.run(self.agent_running_main())
                self.render(counter=counter, head_update=True)

                # transition animation
                time_remaining = self.env.max_step_limit - self.env.timestep_index
                x_vehicle, y_vehicle = [], []
                x0_vehicle, y0_vehicle = [], []
                # imm_coords = []
                # print("Current Number of Vehicles: ", len(self.env.vehicles))
                for agent_idx in range(len(self.env.vehicles)):
                    x_vehicle.append(self.env.vehicles[agent_idx].curr_head[0])
                    y_vehicle.append(self.env.vehicles[agent_idx].curr_head[1])
                    x0_vehicle.append(self.env.vehicles[agent_idx].last_head[0])
                    y0_vehicle.append(self.env.vehicles[agent_idx].last_head[1])
                    # imm_coords.append(pygame.Rect(
                    #     self.env.vehicles[agent_idx].last_head[0] * self.CELL_SIZE,
                    #     self.env.vehicles[agent_idx].last_head[1] * self.CELL_SIZE,
                    #     self.CELL_SIZE,
                    #     self.CELL_SIZE,
                    # ))
                # if counter % 2 == 0:
                x_pedestrian, y_pedestrian = [], []
                x0_pedestrian, y0_pedestrian = [], []
                # print("Current Number of Pedestrians: ", len(self.env.pedestrians))
                for agent_idx in range(len(self.env.pedestrians)):
                    x_pedestrian.append(self.env.pedestrians[agent_idx].curr_head[0])
                    y_pedestrian.append(self.env.pedestrians[agent_idx].curr_head[1])
                    x0_pedestrian.append(self.env.pedestrians[agent_idx].last_head[0])
                    y0_pedestrian.append(self.env.pedestrians[agent_idx].last_head[1])
                    # imm_coords.append(pygame.Rect(
                    #     self.env.pedestrians[agent_idx].last_head[0] * self.CELL_SIZE,
                    #     self.env.pedestrians[agent_idx].last_head[1] * self.CELL_SIZE,
                    #     self.CELL_SIZE,
                    #     self.CELL_SIZE,
                    # ))
                # x, y = self.env.snakes[agent_idx].curr_head
                #
                # x0, y0 = self.env.snakes[agent_idx].last_head
                # imm_coords = pygame.Rect(
                #     x0 * self.CELL_SIZE,
                #     y0 * self.CELL_SIZE,
                #     self.CELL_SIZE,
                #     self.CELL_SIZE,
                # )

                for interpolate_idx in range(1, self.intermediate_frames - 1):
                    self.render(counter=counter)
                    self.render_traffic_lights(traffic_lights_position(self.CELL_SIZE),
                                               pygame.time.get_ticks() - self.pause_time)

                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            # if is_human_agent and event.key in self.SNAKE_CONTROL_KEYS:
                            #     action = self.map_key_to_snake_action(event.key)
                            #     action_selected = True
                            if event.key == pygame.K_SPACE:
                                self.pause = True
                            if event.key == pygame.K_ESCAPE:
                                self.quit_game()

                        if event.type == pygame.QUIT:
                            self.quit_game()
                    self.handle_pause()
                    # try:
                    vehicles_coords, pedestrian_coords = \
                        self.transition_animation(x_vehicle, y_vehicle, x0_vehicle, y0_vehicle, x_pedestrian,
                                                  y_pedestrian, x0_pedestrian, y0_pedestrian, interpolate_idx,
                                                  counter)

                    collision = self.compare_bounding_boxes(vehicles_coords, pedestrian_coords)
                    if collision:
                        for pair in collision:
                            agent1_idx = int(pair[0].split('_')[1])
                            self.env.vehicles[agent1_idx].lock_pos = vehicles_coords[agent1_idx]
                            self.env.vehicles[agent1_idx].lock = True

                            agent2_name = pair[1].split('_')[0]
                            agent2_idx = int(pair[1].split('_')[1])
                            if agent2_name == 'vehicle':
                                self.env.vehicles[agent2_idx].lock_pos = vehicles_coords[agent2_idx]
                                self.env.vehicles[agent2_idx].lock = True

                            else:
                                self.env.pedestrians[agent2_idx].lock_pos = pedestrian_coords[agent2_idx]
                                self.lock_pos_adjustment(agent1_idx, agent2_idx)
                                self.env.pedestrians[agent2_idx].curr_icon = self.set_icon_scheme_pedestrian(agent2_idx,
                                                                                                             self.env.vehicles[
                                                                                                                 agent1_idx].direction)
                                self.env.pedestrians[agent2_idx].lock = True

                        # print("Collision happened.")

                    # except IndexError:
                    #     print("=====================================")
                    #     print("Agent Index: ", agent_idx)
                    #     print("Total Agents: ", len(self.env.vehicles))
                    #     print("=====================================")
                    #
                    #     pass
                    if not self.button_pressed:
                        self.render_scoreboard(score, pygame.time.get_ticks() - self.pause_time,
                                               self.accident_scoreboard + self.violation_scoreboard)
                    else:
                        self.render_scoreboard(score, pygame.time.get_ticks() - self.pause_time,
                                               self.button_pressed_value)
                    pygame.display.set_caption(
                        f'Robotaxi [Score: {score:01d}]   |   [Steps Remaining: {time_remaining:01d}]')

                    pygame.display.update()
                    # self.fps_clock.tick(self.intermediate_frames+5)
                    # print(self.fps_clock.get_fps())
                for agent_idx in range(len(self.env.vehicles)):
                    if counter % self.speed_vehicle[agent_idx] == 0:
                        if self.change_lane_back_vehicles[agent_idx]:
                            self.env.choose_action_multiagents(self.change_lane_back_vehicles[agent_idx], self.env.vehicles,
                                                               agent_idx)

                # pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, imm_coords)

                # for agent_idx in range(len(self.env.vehicles)):
                #     # final pose
                #     cell_coords = pygame.Rect(x_vehicle[agent_idx] * self.CELL_SIZE, y_vehicle[agent_idx] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                #     self.screen.blit(self.env.vehicles[agent_idx].curr_icon, cell_coords)

                # for agent_idx in range(len(self.env.pedestrians)):
                #     # final pose
                #     cell_coords = pygame.Rect(x_pedestrian[agent_idx] * self.CELL_SIZE, y_pedestrian[agent_idx] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                #     self.screen.blit(self.env.pedestrians[agent_idx].curr_icon, cell_coords)

                # # final pose
                # cell_coords = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                # self.screen.blit(self.env.snakes[agent_idx].curr_icon, cell_coords)

                # Pick up item animation
                # self.pickup_animation(x_vehicle, y_vehicle, self.timestep_result.reward, False)

                time_remaining = self.env.max_step_limit - self.env.timestep_index

                pygame.display.set_caption(
                    f'Robotaxi [Score: {score:01d}]   |   [Steps Remaining: {time_remaining:01d}]')
                if not self.button_pressed:
                    self.render_scoreboard(score, pygame.time.get_ticks() - self.pause_time,
                                           self.accident_scoreboard + self.violation_scoreboard)
                else:
                    self.render_scoreboard(score, pygame.time.get_ticks() - self.pause_time,
                                           self.button_pressed_value)
                pygame.display.update()
                # self.fps_clock.tick(self.FPS_LIMIT)
            counter += 1

        self.outlet.push_sample(["TE"])

    # async def agent_running_main(self):
    #     tasks = [self.agent_running(agent_idx) for agent_idx in range(len(self.env.snakes))]
    #     await asyncio.gather(*tasks)
    #
    #
    # async def agent_running(self, agent_idx):
    #     self.render(agent_idx)
    #
    #     score = self.env.stats.sum_episode_rewards
    #
    #     # transition animation
    #     time_remaining = self.env.max_step_limit - self.env.timestep_index
    #     x, y = self.env.snakes[agent_idx].curr_head
    #
    #     x0, y0 = self.env.snakes[agent_idx].last_head
    #     imm_coords = pygame.Rect(
    #         x0 * self.CELL_SIZE,
    #         y0 * self.CELL_SIZE,
    #         self.CELL_SIZE,
    #         self.CELL_SIZE,
    #     )
    #
    #     for interpolate_idx in range(1, self.intermediate_frames - 1):
    #         self.render(agent_idx)
    #         self.render_traffic_lights(self.dict_traffic_lights_pos(),
    #                                    pygame.time.get_ticks() - self.pause_time)
    #
    #         for event in pygame.event.get():
    #             if event.type == pygame.KEYDOWN:
    #                 # if is_human_agent and event.key in self.SNAKE_CONTROL_KEYS:
    #                 #     action = self.map_key_to_snake_action(event.key)
    #                 #     action_selected = True
    #                 if event.key == pygame.K_SPACE:
    #                     self.pause = True
    #                 if event.key == pygame.K_ESCAPE:
    #                     self.quit_game()
    #
    #             if event.type == pygame.QUIT:
    #                 self.quit_game()
    #         self.handle_pause()
    #
    #         imm_coords = self.transition_animation(imm_coords, x, y, x0, y0, self.timestep_result.reward,
    #                                                self.env.snakes[agent_idx].curr_icon, interpolate_idx, False)
    #         self.render_scoreboard(score, pygame.time.get_ticks() - self.pause_time, self.timestep_result.reward)
    #         pygame.display.set_caption(
    #             f'Robotaxi [Score: {score:01d}]   |   [Steps Remaining: {time_remaining:01d}]')
    #
    #         pygame.display.update()
    #         # self.fps_clock.tick(self.intermediate_frames+5)
    #         # print(self.fps_clock.get_fps())
    #     if self.change_lane_back[agent_idx]:
    #         self.env.choose_action_multiagents(self.change_lane_back[agent_idx], agent_idx)
    #
    #     # pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, imm_coords)
    #
    #     # final pose
    #     cell_coords = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
    #     self.screen.blit(self.env.snakes[agent_idx].curr_icon, cell_coords)
    #
    #     # Pick up item animation
    #     self.pickup_animation(x, y, self.timestep_result.reward, False)
    def compare_bounding_boxes(self, array1, array2):
        results = []
        # Compare all boxes in array1
        for i in range(len(array1)):
            for j in range(i + 1, len(array1)):
                if calculate_overlap_area(array1[i], array1[j]):
                    results.append((f"vehicle_{i}", f"vehicle_{j}"))

        # # Compare all boxes in array2
        # for i in range(len(array2)):
        #     for j in range(i + 1, len(array2)):
        #         if calculate_overlap_area(array2[i], array2[j]):
        #             results.append((f"b{i}", f"b{j}"))

        # Compare boxes between array1 and array2
        for i in range(len(array1)):
            for j in range(len(array2)):
                if calculate_overlap_area(array1[i], array2[j]):
                    results.append((f"vehicle_{i}", f"pedestrian_{j}"))

        return results

    def lock_pos_adjustment(self, agent1_idx, agent2_idx):

        if not self.env.pedestrians[agent2_idx].lock:
            if self.env.vehicles[agent1_idx].direction == SnakeDirection.NORTH:
                self.env.pedestrians[agent2_idx].lock_pos = self.env.vehicles[agent1_idx].lock_pos.move(-20, -70)
                # self.env.pedestrians[agent2_idx].lock_pos.move_ip(0, -20)
            elif self.env.vehicles[agent1_idx].direction == SnakeDirection.WEST:
                self.env.pedestrians[agent2_idx].lock_pos = self.env.vehicles[agent1_idx].lock_pos.move(-50, -10)
                # self.env.pedestrians[agent2_idx].lock_pos.move_ip(-20, 0)
            elif self.env.vehicles[agent1_idx].direction == SnakeDirection.SOUTH:
                self.env.pedestrians[agent2_idx].lock_pos = self.env.vehicles[agent1_idx].lock_pos.move(10, 50)
                # self.env.pedestrians[agent2_idx].lock_pos.move_ip(0, 20)
            elif self.env.vehicles[agent1_idx].direction == SnakeDirection.EAST:
                self.env.pedestrians[agent2_idx].lock_pos = self.env.vehicles[agent1_idx].lock_pos.move(50, -10)
                # self.env.pedestrians[agent2_idx].lock_pos.move_ip(20, 0)


class Stopwatch(object):
    """ Measures the time elapsed since the last checkpoint. """

    def __init__(self):
        self.start_time = pygame.time.get_ticks()

    def reset(self):
        """ Set a new checkpoint. """
        self.start_time = pygame.time.get_ticks()

    def time(self):
        """ Get time (in milliseconds) since the last checkpoint. """
        return pygame.time.get_ticks() - self.start_time


class Colors:
    SCREEN_BACKGROUND = (119, 119, 119)
    SCORE = (120, 100, 84)
    SCORE_GOOD = (50, 205, 50)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    SCORE_BAD = (255, 255, 33)
    SCORE_VERY_BAD = (205, 20, 50)
    SELECTION = (215, 215, 215)
    CELL_TYPE = {
        CellType.WALL: (26, 26, 26),
        CellType.SNAKE_BODY: (82, 154, 255),
        CellType.SNAKE_HEAD: (65, 132, 255),
        CellType.GOOD_EVENT: (85, 242, 240),
        CellType.BAD_EVENT: (177, 242, 85),
        CellType.LAVA: (150, 53, 219),
        CellType.PIT: (179, 179, 179),
    }


class QuitRequestedError(RuntimeError):
    """ Gets raised whenever the user wants to quit the game. """
    pass


class captureThread(threading.Thread):

    def __init__(self, threadID, participant='test', data_dir='./user_study_data/', exp_id='collaborative', test=False):
        super(captureThread, self).__init__()
        # threading.Thread.__init__(self)
        self._stop_event = threading.Event()
        self.threadID = threadID
        self.prefix = participant + '_' + exp_id
        self.data_dir = data_dir
        self.participant = participant
        self.exp_id = exp_id

        if not test:
            self.store_dir = data_dir + 'webcam_imgs/' + participant + '/' + exp_id
        else:
            self.store_dir = data_dir + 'webcam_imgs/' + participant + '/' + exp_id + '_test'
        if not os.path.exists(data_dir + 'webcam_imgs/' + participant):
            os.makedirs(data_dir + 'webcam_imgs/' + participant)
        if not os.path.exists(self.store_dir):
            os.makedirs(self.store_dir)

    def run(self):
        global frame_ct
        print("Starting " + self.name + ' at ' + time.ctime(time.time()))
        cap = cv2.VideoCapture(0)
        while frame_ct == -1:
            time.sleep(0.01)
        img_ct = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                cv2.imwrite(self.store_dir + '/' + self.prefix + '_' + str(round(time.time() * 1000)) + '_' + str(
                    frame_ct) + '_' + str(img_ct) + '.jpg', frame)
                img_ct += 1
                if self._stop_event.is_set(): break

        cap.release()
        cv2.destroyAllWindows()
        print("Exiting cam")

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


def traffic_light_position_mapping():
    return {'01': [(2, 5), (2, 6), (2, 7)],
            '02': [(4, 11), (5, 11), (6, 11)],
            '03': [(10, 7), (10, 8), (10, 9)],
            '04': [(6, 3), (7, 3), (8, 3)],
            '05': [(3, 4), (3, 10)],
            '06': [(3, 10), (9, 10)],
            '07': [(9, 10), (9, 4)],
            '08': [(3, 4), (9, 4)],
            '09': [(4, 18), (5, 18), (6, 18)],
            '10': [(10, 15), (10, 16)],
            '11': [(7, 13), (8, 13)],
            '12': [(3, 14), (3, 17)],
            '13': [(3, 17), (9, 17)],
            '14': [(9, 17), (9, 14)],
            '15': [(9, 14), (3, 14)],
            '16': [(14, 5), (14, 6)],
            '17': [(16, 11), (17, 11)],
            '18': [(17, 3), (18, 3)],
            '19': [(15, 4), (15, 10)],
            '20': [(15, 10), (19, 10)],
            '21': [(15, 4), (19, 4)],
            '22': [(16, 18), (17, 18)],
            '23': [(20, 16)],
            '24': [(18, 13)],
            '25': [(15, 14), (15, 17)],
            '26': [(15, 17), (19, 17)],
            '27': [(19, 17), (19, 14)],
            '28': [(19, 14), (15, 14)]
            }


def traffic_lights_position(CELL_SIZE):
    return {'01': (CELL_SIZE * 6 - 10, CELL_SIZE * 4 + 10),
            '02': (CELL_SIZE * 9 + 15, CELL_SIZE * 5 - 10),
            '03': (CELL_SIZE * 8 - 10, CELL_SIZE * 8 + 20),
            '04': (CELL_SIZE * 5 + 10, CELL_SIZE * 7 - 10),
            '05': (CELL_SIZE * 7 + 60, CELL_SIZE * 3 + 17),
            '06': (CELL_SIZE * 10 + 17, CELL_SIZE * 6 + 60),
            '07': (CELL_SIZE * 6 + 55, CELL_SIZE * 9 + 17),
            '08': (CELL_SIZE * 4 + 17, CELL_SIZE * 5 + 60),

            '09': (CELL_SIZE * 16 + 15, CELL_SIZE * 5 - 10),
            '10': (CELL_SIZE * 16 - 30, CELL_SIZE * 8 + 35),
            '11': (CELL_SIZE * 15, CELL_SIZE * 8 - 25),
            '12': (CELL_SIZE * 15 + 60, CELL_SIZE * 3 + 17),
            '13': (CELL_SIZE * 17 + 17, CELL_SIZE * 6 + 60),
            '14': (CELL_SIZE * 15 + 60, CELL_SIZE * 9 + 17),
            '15': (CELL_SIZE * 14 + 17, CELL_SIZE * 6 + 60),

            '16': (CELL_SIZE * 6 - 25, CELL_SIZE * 16),
            '17': (CELL_SIZE * 9 + 30, CELL_SIZE * 17 - 30),
            '18': (CELL_SIZE * 5, CELL_SIZE * 18 - 25),
            '19': (CELL_SIZE * 6 + 55, CELL_SIZE * 15 + 17),
            '20': (CELL_SIZE * 10 + 17, CELL_SIZE * 17 + 60),
            '21': (CELL_SIZE * 4 + 17, CELL_SIZE * 16 + 60),

            '22': (CELL_SIZE * 16 + 30, CELL_SIZE * 17 - 30),
            '23': (CELL_SIZE * 16 + 15, CELL_SIZE * 18 + 35),
            '24': (CELL_SIZE * 15, CELL_SIZE * 18 + 15),
            '25': (CELL_SIZE * 15 + 60, CELL_SIZE * 15 + 17),
            '26': (CELL_SIZE * 17 + 17, CELL_SIZE * 17 + 60),
            '27': (CELL_SIZE * 15 + 60, CELL_SIZE * 19 + 17),
            '28': (CELL_SIZE * 14 + 17, CELL_SIZE * 17 + 60)}


def initialize_traffic_lights_status():
    """
    Initialize the traffic lights status.
    Single light
    1: red
    2: green constant
    3: green blinking
    Two lights (two digits) --> left green constant and right red then 21
    1: left red
    2: left green constant
    3: left green blinking
    +
    1: right red
    2: right green constant
    3: right green blinking
    """
    return {'01': 11, '02': 11, '03': 11, '04': 11, '05': 1, '06': 1, '07': 1, '08': 1,
            '09': 11, '10': 1, '11': 1, '12': 1, '13': 1, '14': 1, '15': 1,
            '16': 1, '17': 1, '18': 1, '19': 1, '20': 1, '21': 1,
            '22': 1, '23': 1, '24': 1, '25': 1, '26': 1, '27': 1, '28': 1}


def calculate_overlap_area(box1, box2):
    # Extract coordinates
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the coordinates of the intersection rectangle
    inter_left = max(x1, x2)
    inter_top = max(y1, y2)
    inter_right = min(x1 + w1, x2 + w2)
    inter_bottom = min(y1 + h1, y2 + h2)

    # Calculate the width and height of the intersection rectangle
    inter_width = max(0, inter_right - inter_left)
    inter_height = max(0, inter_bottom - inter_top)

    # Calculate the area of the intersection rectangle
    inter_area = inter_width * inter_height

    # Calculate the area of both bounding boxes
    area1 = w1 * h1
    area2 = w2 * h2

    # Check if the overlap is greater than half the area of one of the boxes
    if inter_area > 0.5 * area1 or inter_area > 0.5 * area2:
        return True
    else:
        return False
