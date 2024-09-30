import numpy as np
import pygame
import time
import cv2
import os
import threading
from robotaxi_integration.robotaxi.agent import HumanAgent
from robotaxi_integration.robotaxi.gameplay.entities import (CellType, SnakeAction, SnakeDirection, ALL_SNAKE_DIRECTIONS)
from robotaxi_integration.robotaxi.gameplay.environment import PLAY_SOUND, TimestepResult
from robotaxi_integration.robotaxi.utils.next_step_choice import get_next_step_choices_car_smooth_turn, get_next_step_choices_pedestrian
import ctypes
from dualsense import DualSense
from pylsl import StreamInfo, StreamOutlet
from robotaxi_integration.robotaxi.utils.pathfinding import pathfinding
from robotaxi_integration.robotaxi.utils.utils import add_frame_to_buffer, save_buffered_frames, get_screenshot
import queue
import multiprocessing
ctypes.windll.user32.SetProcessDPIAware()

frame_ct = -1

LANE_PICKUP = [((10, 4), (13, 4)),
               ((9, 11), (9, 11)),
               ((11, 10), (14, 10)),
               ((15, 12), (15, 13)),
               ((19, 11), (19, 12)),
               ((11, 14), (14, 14)),
               ((12, 17), (14, 17)),
               ]
LANE_ZEBRA_CROSSING_P = [((4, 4), (8, 4)),
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
LANE_ZEBRA_CROSSING_V = [((4, 4), (8, 4)),
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
LANE = [((4, 1), (5, 4)),
        ((6, 1), (8, 3)),

        ((1, 5), (2, 7)),
        ((1, 8), (3, 9)),

        ((4, 11), (6, 14)),
        ((7, 10), (8, 13)),

        ((4, 18), (6, 21)),
        ((7, 17), (8, 21)),

        ((1, 15), (3, 16)),

        ((9, 5), (14, 6)),
        ((10, 7), (15, 9)),

        ((10, 15), (15, 16)),

        ((16, 1), (18, 3)),

        ((16, 11), (17, 14)),
        ((18, 10), (18, 13)),

        ((16, 18), (18, 21)),

        ((20, 15), (21, 16)),
        ]
BUILDING_POS = {
    # 'shop':              (self.CELL_SIZE * 10 + 25, self.CELL_SIZE * 20 + 20),
    # 'court':             (self.CELL_SIZE * 18 + 30, self.CELL_SIZE * 20 + 20),
    # 'school':            (self.CELL_SIZE * 18 + 30, self.CELL_SIZE * 1 + 20),
    # 'park':              (self.CELL_SIZE * 11 + 40, self.CELL_SIZE * 10 + 20),
    # 'office':            (self.CELL_SIZE * 11 + 40, self.CELL_SIZE * 12 + 60),
    # 'hospital':          (self.CELL_SIZE * 18 + 30, self.CELL_SIZE * 10 + 20),
    # 'restaurant':        (self.CELL_SIZE * 1 + 60, self.CELL_SIZE * 12 + 100),
    # 'police_station':    (self.CELL_SIZE * 11 + 60, self.CELL_SIZE * 1 + 20),
    # 'stadium':           (self.CELL_SIZE * 1 + 40, self.CELL_SIZE * 1 + 10),
    #
    # 'taxi_building_top': (self.CELL_SIZE * 1 + 30, self.CELL_SIZE * 20 + 20),
    # 'taxi_building_mid': (self.CELL_SIZE * 1 + 60, self.CELL_SIZE * 10 + 20),
    # 'taxi_building_bot': (self.CELL_SIZE * 20 + 10, self.CELL_SIZE * 12 + 100)
    'shop': (10, 20),
    'court': (18, 20),
    'school': (18, 1),
    'park': (11, 10),
    'office': (11, 13),
    'hospital': (18, 10),
    'restaurant': (2, 13),
    'police_station': (12, 1),
    'stadium': (1, 1),

    'taxi_building_top': (1, 20),
    'taxi_building_mid': (2, 10),
    'taxi_building_bot': (20, 13)
}
BUILDING_POS_PARKING = {
    'shop': (11, 18),
    'court': (18, 18),
    'school': (16, 2),
    'park': (11, 8),
    'office': (9, 14),
    'hospital': (18, 8),
    'restaurant': (2, 16),
    'police_station': (9, 2),
    'stadium': (3, 4),
}
PEDESTRIAN_GETOFF_POS = {
    'shop': (11, 19),
    'court': (18, 19),
    'school': (17, 2),
    'park': (11, 9),
    'office': (10, 14),
    'hospital': (18, 9),
    'restaurant': (2, 15),
    'police_station': (10, 2),
    'stadium': (3, 3),
}
BUILDING_POS_PARKING_DIRECTION = {

    'shop': SnakeDirection.EAST,
    'court': SnakeDirection.EAST,
    'school': SnakeDirection.NORTH,
    'park': SnakeDirection.EAST,
    'office': SnakeDirection.NORTH,
    'hospital': SnakeDirection.EAST,
    'restaurant': SnakeDirection.WEST,
    'police_station': SnakeDirection.NORTH,
    'stadium': SnakeDirection.WEST,
}

image_buffer1 = []
image_buffer2 = []


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

    def __init__(self, q_rcv, q_send, save_frames=False, field_size=8, test=False, mode=None):
        # pygame.mixer.pre_init(44100, -16, 2, 32)
        pygame.init()
        pygame.mixer.init()

        pygame.mouse.set_visible(True)
        self.logger = multiprocessing.get_logger()
        self.q_rcv = q_rcv
        self.q_send = q_send
        self.mode = mode
        self.intermediate_frames = 15
        self.punch_sound = pygame.mixer.Sound('robotaxi_integration/sound/punch.wav')
        self.begin_sound = pygame.mixer.Sound('robotaxi_integration/sound/begin.wav')
        self.good_sound = pygame.mixer.Sound('robotaxi_integration/sound/good.wav')
        self.bad_sound = pygame.mixer.Sound('robotaxi_integration/sound/road_block_crash.wav')
        self.very_bad_sound = pygame.mixer.Sound('robotaxi_integration/sound/car_crash.wav')
        self.stuck_sound = pygame.mixer.Sound('robotaxi_integration/sound/woop.wav')
        self.free_sound = pygame.mixer.Sound('robotaxi_integration/sound/restart.wav')
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

        self.car_schemes = ["robotaxi", "pickup", "truck", "bulldozer"]
        self.human_schemes = []
        self.selected_icon_scheme = 0  # default
        self.set_icon_scheme_vehicle(self.selected_icon_scheme)
        # self.set_icon_scheme_pedestrian()
        self.selected_icon_scheme_collaborator = 0
        self.set_icon_scheme_collaborator(self.selected_icon_scheme_collaborator)
        self.selected = False
        self.trees = True
        self.test = test
        self.load_icon = True
        if self.load_icon:
            self.load_icons()

        self.traffic_light_status = initialize_traffic_lights_status()

        self.action_car_north, self.action_car_south, self.action_car_east, self.action_car_west = get_next_step_choices_car_smooth_turn()
        self.action_pedestrian_north, self.action_pedestrian_south, self.action_pedestrian_east, self.action_pedestrian_west = get_next_step_choices_pedestrian()

        self.post_action_head_list = [None]
        self.curr_head = [0, 0]
        self.last_head = [0, 0]
        self.curr_head_collaborator = [0, 0]
        self.last_head_collaborator = [0, 0]
        self.internal_padding = self.CELL_SIZE // 5
        self.text_font = pygame.font.Font("robotaxi_integration/fonts/gyparody_hv.ttf", int(23 * (self.CELL_SIZE / 40.0)))
        self.num_font = pygame.font.Font("robotaxi_integration/fonts/gyparody_tf.ttf", int(36 * (self.CELL_SIZE / 40.0)))
        self.marker_font = pygame.font.Font("robotaxi_integration/fonts/OpenSans-Bold.ttf", int(12 * (self.CELL_SIZE / 40.0)))

        self.warning_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/question1.png"),
                                                   (self.CELL_SIZE, self.CELL_SIZE))
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
            'ps_pressed': self.ps_pressed_callback,
            'touchpad_pressed': self.touchpad_pressed_callback,
        }

        self.bus_turn_icon = {}
        self.pickup_turn_icon = {}
        self.truck_turn_icon = {}
        self.bulldozer_turn_icon = {}
        # self.vehicle_turn_icon_resize()
        self.vehicle_turn_icon_load()
        self.dualsense = DualSense(callbacks=self.callbacks)
        self.event_lock = False  # lock to prevent multiple violations from being triggered at the same time
        self.button_pressed = False
        self.button_pressed_value = None

        info = StreamInfo("Robotaxi surveillance", "Markers", 1, 0, "string", "myuid000")

        # next make an outlet
        self.outlet = StreamOutlet(info)

        pygame.display.set_caption('Robotaxi Surveillance')

    def load_icons(self):
        self.spawn_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/wave.png"), (self.CELL_SIZE, self.CELL_SIZE))
        self.wall_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/forest.png"), (self.CELL_SIZE, self.CELL_SIZE))
        self.good_fruit_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/man.png"),
                                                      (self.CELL_SIZE * 2 // 3, self.CELL_SIZE * 2 // 3))
        self.man1_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/man/man1.png"),
                                                (self.CELL_SIZE * 2 // 1.9, self.CELL_SIZE * 2 // 1.9))
        self.man1_icon_west = pygame.transform.rotate(self.man1_icon, 45)
        self.man1_icon_east = pygame.transform.rotate(self.man1_icon, -45)
        self.man1_icon_south = pygame.transform.rotate(self.man1_icon, -135)

        self.cop_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/man/cop.png"),
                                               (self.CELL_SIZE * 2 // 2, self.CELL_SIZE * 2 // 2))
        self.cop_icon_west = pygame.transform.rotate(self.cop_icon, 45)
        self.cop_icon_east = pygame.transform.rotate(self.cop_icon, -45)
        self.cop_icon_south = pygame.transform.rotate(self.cop_icon, -135)

        self.man2_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/man/man2.png"),
                                                (self.CELL_SIZE * 2 // 2, self.CELL_SIZE * 2 // 2))
        self.man2_icon_west = pygame.transform.rotate(self.man2_icon, 45)
        self.man2_icon_east = pygame.transform.rotate(self.man2_icon, -45)
        self.man2_icon_south = pygame.transform.rotate(self.man2_icon, -135)

        self.man3_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/man/man3.png"),
                                                (self.CELL_SIZE * 2 // 2, self.CELL_SIZE * 2 // 2))
        self.man3_icon_west = pygame.transform.rotate(self.man3_icon, 45)
        self.man3_icon_east = pygame.transform.rotate(self.man3_icon, -45)
        self.man3_icon_south = pygame.transform.rotate(self.man3_icon, -135)

        self.woman1_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/man/woman1.png"),
                                                  (self.CELL_SIZE * 2 // 2, self.CELL_SIZE * 2 // 2))
        self.woman1_icon_west = pygame.transform.rotate(self.woman1_icon, 45)
        self.woman1_icon_east = pygame.transform.rotate(self.woman1_icon, -45)
        self.woman1_icon_south = pygame.transform.rotate(self.woman1_icon, -135)

        self.woman2_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/man/woman2.png"),
                                                  (self.CELL_SIZE * 2 // 2, self.CELL_SIZE * 2 // 2))
        self.woman2_icon_west = pygame.transform.rotate(self.woman2_icon, 45)
        self.woman2_icon_east = pygame.transform.rotate(self.woman2_icon, -45)
        self.woman2_icon_south = pygame.transform.rotate(self.woman2_icon, -135)

        self.bad_fruit_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/road_block.png"),
                                                     (self.CELL_SIZE * 2 // 3, self.CELL_SIZE * 2 // 3))
        self.lava_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/purple_car.png"),
                                                (self.CELL_SIZE, self.CELL_SIZE))
        self.small_crash_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/road_block_broken.png"),
                                                       (self.CELL_SIZE * 2 // 3, self.CELL_SIZE * 2 // 3))
        self.big_crash_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/broken_purple_car.png"),
                                                     (self.CELL_SIZE, self.CELL_SIZE * 2 // 3))
        self.reward_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/dollar.png"),
                                                  (self.CELL_SIZE // 3, self.CELL_SIZE // 3))
        self.curr_icon = None
        self.question1_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/question1.png"),
                                                     (self.CELL_SIZE * 2 // 3, self.CELL_SIZE * 2 // 3))
        self.question2_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/question2.png"),
                                                     (self.CELL_SIZE * 2 // 3, self.CELL_SIZE * 2 // 3))
        self.question3_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/question3.png"),
                                                     (self.CELL_SIZE * 2 // 3, self.CELL_SIZE * 2 // 3))
        self.pit_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/stopped.png"), (self.CELL_SIZE, self.CELL_SIZE))
        self.stop_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/stopped.png"),
                                                (self.CELL_SIZE // 3, self.CELL_SIZE // 3))
        self.accident_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/accident.png"),
                                                    (self.CELL_SIZE * 1.5, self.CELL_SIZE * 1.5))
        # self.head_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/snake.png"),(self.CELL_SIZE, self.CELL_SIZE))
        # self.body_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/body.png"),(self.CELL_SIZE, self.CELL_SIZE))
        self.punch_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/scary_tree.png"),
                                                 (self.CELL_SIZE, self.CELL_SIZE))
        self.thought_bubble_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/thought-bubble.png"),
                                                          (self.CELL_SIZE, self.CELL_SIZE))

        self.arrow_left_north = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/left_north.png"),
                                                       (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_right_north = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/right_north.png"),
                                                        (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_straight_north = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/straight_north.png"),
                                                           (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_left_south = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/left_south.png"),
                                                       (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_right_south = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/right_south.png"),
                                                        (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_straight_south = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/straight_south.png"),
                                                           (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_left_east = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/left_east.png"),
                                                      (self.CELL_SIZE, self.CELL_SIZE // 2))
        self.arrow_right_east = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/right_east.png"),
                                                       (self.CELL_SIZE, self.CELL_SIZE // 2))
        self.arrow_straight_east = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/straight_east.png"),
                                                          (self.CELL_SIZE, self.CELL_SIZE // 2))
        self.arrow_left_west = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/left_west.png"),
                                                      (self.CELL_SIZE, self.CELL_SIZE // 2))
        self.arrow_right_west = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/right_west.png"),
                                                       (self.CELL_SIZE, self.CELL_SIZE // 2))
        self.arrow_straight_west = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/straight_west.png"),
                                                          (self.CELL_SIZE, self.CELL_SIZE // 2))
        self.arrow_all_north = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/all_north.png"),
                                                      (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_all_south = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/all_south.png"),
                                                      (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_all_east = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/all_east.png"),
                                                     (self.CELL_SIZE, self.CELL_SIZE // 2))
        self.arrow_all_west = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/all_west.png"),
                                                     (self.CELL_SIZE, self.CELL_SIZE // 2))
        self.arrow_noright_north = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/noright_north.png"),
                                                          (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_noright_south = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/noright_south.png"),
                                                          (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_noright_east = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/noright_east.png"),
                                                         (self.CELL_SIZE, self.CELL_SIZE // 1.3))
        self.arrow_noright_west = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/noright_west.png"),
                                                         (self.CELL_SIZE, self.CELL_SIZE // 2))
        self.arrow_rightonly_south = pygame.transform.flip(self.arrow_left_south, 1, 0)
        self.arrow_uturn_north = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/uturn.png"),
                                                        (self.CELL_SIZE // 2, self.CELL_SIZE))
        self.arrow_uturn_west = pygame.transform.rotate(self.arrow_uturn_north, 90)
        self.arrow_uturn_east = pygame.transform.rotate(self.arrow_uturn_west, 180)

        self.arrow_uturn_left_south = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Arrows/uturn_left.png"),
                                                             (self.CELL_SIZE // 1.8, self.CELL_SIZE))

        self.house_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/house.png"),
                                                 (self.CELL_SIZE * 1.5, self.CELL_SIZE * 1.5))
        self.school_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/school.png"),
                                                  (self.CELL_SIZE * 1.5, self.CELL_SIZE * 1.5))
        self.park_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/park.png"),
                                                (self.CELL_SIZE * 2, self.CELL_SIZE * 2))
        self.hospital_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/hospital.png"),
                                                    (self.CELL_SIZE * 1.5, self.CELL_SIZE * 1.5))
        self.shop_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/shop.png"),
                                                (self.CELL_SIZE * 1.5, self.CELL_SIZE * 1.5))
        self.office_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/construction.png"),
                                                  (self.CELL_SIZE * 2, self.CELL_SIZE * 2))
        self.police_station_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/police-station.png"),
                                                          (self.CELL_SIZE * 1.5, self.CELL_SIZE * 1.5))
        self.taxibuilding_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/taxi-building.png"),
                                                        (self.CELL_SIZE * 1.5, self.CELL_SIZE * 1.5))
        self.tree_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/tree.png"),
                                                (self.CELL_SIZE, self.CELL_SIZE))
        self.court_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/court.png"),
                                                 (self.CELL_SIZE * 1.5, self.CELL_SIZE * 1.5))
        self.restaurant_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/restaurant.png"),
                                                      (self.CELL_SIZE * 1.5, self.CELL_SIZE * 1.5))
        self.stadium_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/stadium.png"),
                                                   (self.CELL_SIZE * 2, self.CELL_SIZE * 2))

        self.thought_school_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/thought_school.png"),
                                                          (self.CELL_SIZE, self.CELL_SIZE))
        self.thought_park_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/thought_park.png"),
                                                        (self.CELL_SIZE, self.CELL_SIZE))
        self.thought_hospital_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/thought_hospital.png"),
                                                            (self.CELL_SIZE, self.CELL_SIZE))
        self.thought_shop_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/thought_shop.png"),
                                                        (self.CELL_SIZE, self.CELL_SIZE))
        self.thought_office_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/thought_construction.png"),
                                                          (self.CELL_SIZE, self.CELL_SIZE))
        self.thought_police_station_icon = pygame.transform.scale(
            pygame.image.load("robotaxi_integration/icon/Mine/thought_police-station.png"),
            (self.CELL_SIZE, self.CELL_SIZE))
        self.thought_court_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/thought_court.png"),
                                                         (self.CELL_SIZE, self.CELL_SIZE))
        self.thought_restaurant_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/thought_restaurant.png"),
                                                              (self.CELL_SIZE, self.CELL_SIZE))
        self.thought_stadium_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/thought_stadium.png"),
                                                           (self.CELL_SIZE, self.CELL_SIZE))
        self.thought_car_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/thought_car.png"),
                                                       (self.CELL_SIZE * 1.2, self.CELL_SIZE * 1.2))

        self.triangle_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/ps_triangle.png"),
                                                    (self.CELL_SIZE / 1.38, self.CELL_SIZE / 1.38))
        self.circle_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/ps_circle.png"),
                                                  (self.CELL_SIZE / 1.38, self.CELL_SIZE / 1.38))
        self.square_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/ps_square.png"),
                                                  (self.CELL_SIZE / 1.44, self.CELL_SIZE / 1.44))
        self.cross_icon = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/Mine/ps_cross.png"),
                                                 (self.CELL_SIZE / 1.44, self.CELL_SIZE / 1.44))

        self.two_all_red_north = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/trafficlight/2_all_red.png"),
                                                        (self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        self.two_left_red_north = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/trafficlight/2_left_red.png"),
                                                         (self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        self.two_straight_red_north = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/trafficlight/2_straight_red.png"),
                                                             (self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        self.two_left_red_off_north = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/trafficlight/2_left_red_off.png"),
                                                             (self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        self.two_straight_red_off_north = pygame.transform.scale(
            pygame.image.load("robotaxi_integration/icon/trafficlight/2_straight_red_off.png"),
            (self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        # self.two_off_north = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/trafficlight/2_off.png"),(self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        # self.three_all_red_north = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/trafficlight/3_all_red.png"),(self.CELL_SIZE // 2, self.CELL_SIZE // 2))
        # self.three_off_north = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/trafficlight/3_off.png"),(self.CELL_SIZE // 2, self.CELL_SIZE // 2))
        self.single_red = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/trafficlight/single_red.png"),
                                                 (self.CELL_SIZE // 1.5, self.CELL_SIZE // 1.5))
        self.single_green = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/trafficlight/single_green.png"),
                                                   (self.CELL_SIZE // 1.5, self.CELL_SIZE // 1.5))
        self.single_off = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/trafficlight/single_off.png"),
                                                 (self.CELL_SIZE // 1.5, self.CELL_SIZE // 1.5))

        self.pedestrian_red_north = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/trafficlight/stop.png"),
                                                           (self.CELL_SIZE // 1.7, self.CELL_SIZE // 1.7))
        self.pedestrian_green_north = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/trafficlight/pass.png"),
                                                             (self.CELL_SIZE // 1.7, self.CELL_SIZE // 1.7))
        self.pedestrian_green_off = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/trafficlight/pass_off.png"),
                                                           (self.CELL_SIZE // 1.7, self.CELL_SIZE // 1.7))

        self.uturn_all_red = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/trafficlight/uturn_all_red.png"),
                                                    (self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        self.uturn_all_red_west = pygame.transform.rotate(self.uturn_all_red, 90)
        self.uturn_uturn_red = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/trafficlight/uturn_uturn_red.png"),
                                                      (self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        self.uturn_uturn_red_west = pygame.transform.rotate(self.uturn_uturn_red, 90)
        self.uturn_uturn_red_off = pygame.transform.scale(
            pygame.image.load("robotaxi_integration/icon/trafficlight/uturn_uturn_red_off.png"),
            (self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        self.uturn_uturn_red_off_west = pygame.transform.rotate(self.uturn_uturn_red_off, 90)
        self.uturn_straight_red = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/trafficlight/uturn_straight_red.png"),
                                                         (self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
        self.uturn_straight_red_west = pygame.transform.rotate(self.uturn_straight_red, 90)
        self.uturn_straight_red_off = pygame.transform.scale(
            pygame.image.load("robotaxi_integration/icon/trafficlight/uturn_straight_red_off.png"),
            (self.CELL_SIZE * 1.33, self.CELL_SIZE / 1.38))
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

    def vehicle_turn_icon_resize(self):
        # name = ['bus']
        name = ['bus', 'truck', 'pickup', 'bulldozer']

        for n in name:
            for i in range(5, 95, 5):

            # for i in range(-175, -90, 5):
                path = "robotaxi_integration/icon/turn/{}/{}.png".format(n, i)
                icon = pygame.transform.scale(pygame.image.load(path), (self.CELL_SIZE, self.CELL_SIZE))

                pygame.image.save(icon, path)

                icon = pygame.transform.flip(icon, 1, 0)
                # save the icon
                path = "robotaxi_integration/icon/turn/{}/{}.png".format(n, abs(i))

                pygame.image.save(icon, path)

    def vehicle_turn_icon_load(self):
        name = ['bus', 'truck', 'pickup', 'bulldozer']
        for n in name:
            for i in range(-180, 185, 5):
                path = "./robotaxi_integration/icon/turn/{}/{}.png".format(n, i)
                attribute_name = f"{n}_turn_icon"
                getattr(self, attribute_name)[i] = pygame.transform.scale(pygame.image.load(path), (self.CELL_SIZE, self.CELL_SIZE - 5))
                # self.bus_turn_icon[i] = pygame.transform.scale(pygame.image.load(path),
                #                                                (self.CELL_SIZE, self.CELL_SIZE - 5))

        # print("Bus turn icons loaded")

    def touchpad_pressed_callback(self):

        if self.pause:
            self.temp = pygame.time.get_ticks()

        self.pause = not self.pause

        if not self.pause:
            self.pause_time += pygame.time.get_ticks() - self.temp
            if self.frame_num == 0:
                # pygame.mixer.music.load("robotaxi_integration/sound/background2.mp3")
                # pygame.mixer.music.set_volume(0.1)
                # pygame.mixer.music.play(-1, 0.0)
                pass

    def triangle_pressed_callback(self):
        # print("Triangle button was pressed!")
        if self.running and (self.event_type is not None) and (not self.button_pressed):
            print(self.event_type)
            self.outlet.push_sample(["R 1"])
            self.q_send.put(get_screenshot(self.screen, 3))
            if self.event_type == "vehicle_violation":
                result = TimestepResult(observation=self.env.get_observation(), reward=10, is_episode_end=False)
                self.env.record_timestep_stats(result, 0)
                self.button_pressed_value = 10

            else:
                result = TimestepResult(observation=self.env.get_observation(), reward=-10, is_episode_end=False)
                self.env.record_timestep_stats(result, 0)
                self.button_pressed_value = -10
            self.button_pressed = True
        elif self.event_type is None:
            print("No violation to report")

    def circle_pressed_callback(self):
        # print("Triangle button was pressed!")
        if self.running and (self.event_type is not None) and (not self.button_pressed):
            print(self.event_type)
            self.outlet.push_sample(["R 2"])
            self.q_send.put(get_screenshot(self.screen, 3))
            if self.event_type == "pedestrian_violation":
                result = TimestepResult(observation=self.env.get_observation(), reward=10, is_episode_end=False)
                self.env.record_timestep_stats(result, 0)
                self.button_pressed_value = 10
            else:
                result = TimestepResult(observation=self.env.get_observation(), reward=-10, is_episode_end=False)
                self.env.record_timestep_stats(result, 0)
                self.button_pressed_value = -10
            self.button_pressed = True
        elif self.event_type is None:
            print("No violation to report")

    def square_pressed_callback(self):
        # print("Triangle button was pressed!")
        if self.running and (self.event_type is not None) and (not self.button_pressed):
            print(self.event_type)
            self.outlet.push_sample(["R 3"])
            self.q_send.put(get_screenshot(self.screen, 3))
            if self.event_type == "vehicle_accident":
                result = TimestepResult(observation=self.env.get_observation(), reward=20, is_episode_end=False)
                self.env.record_timestep_stats(result, 0)
                self.button_pressed_value = 20
            else:
                result = TimestepResult(observation=self.env.get_observation(), reward=-10, is_episode_end=False)
                self.env.record_timestep_stats(result, 0)
                self.button_pressed_value = -10
            self.button_pressed = True
        elif self.event_type is None:
            print("No violation to report")

    def cross_pressed_callback(self):
        # print("Triangle button was pressed!")
        if self.running and (self.event_type is not None) and (not self.button_pressed):
            print(self.event_type)
            self.outlet.push_sample(["R 4"])
            self.q_send.put(get_screenshot(self.screen, 3))
            if self.event_type == "pedestrian_accident":
                result = TimestepResult(observation=self.env.get_observation(), reward=20, is_episode_end=False)
                self.env.record_timestep_stats(result, 0)
                self.button_pressed_value = 20
            else:
                result = TimestepResult(observation=self.env.get_observation(), reward=-10, is_episode_end=False)
                self.env.record_timestep_stats(result, 0)
                self.button_pressed_value = -10
            self.button_pressed = True
        elif self.event_type is None:
            print("No violation to report")

    def ps_pressed_callback(self):
        pass

    def receive_send_rumble(self):
        try:
            value = self.q_rcv.get(timeout=0.01)  # Non-blocking mode
            # print(f"Received: {value}")
            if self.dualsense.is_initialized():
                self.dualsense.start_rumble_thread(self.dualsense.general_rumble, *value)
                self.outlet.push_sample(["S 2"])
            else:
                print("try to send rumble but no dualsense connect")
        except queue.Empty:
            print("Queue is empty, no values available.")

    def set_icon_scheme_vehicle(self, idx):

        # scheme = self.car_schemes[idx]

        self.south, self.north, self.east, self.west = [], [], [], []
        for i in range(len(self.car_schemes)):
            scheme = self.car_schemes[i]
            self.south.append(pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/" + scheme + "_south.png"),
                                                     (self.CELL_SIZE, self.CELL_SIZE - 5)))
            self.north.append(pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/" + scheme + "_north.png"),
                                                     (self.CELL_SIZE, self.CELL_SIZE - 5)))
            self.east.append(pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/" + scheme + "_east.png"),
                                                    (self.CELL_SIZE, self.CELL_SIZE - 5)))
            self.west.append(
                pygame.transform.flip(pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/" + scheme + "_east.png"),
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
        self.south_collaborator = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/" + scheme + "_south.png"),
                                                         (self.CELL_SIZE, self.CELL_SIZE - 5))
        self.north_collaborator = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/" + scheme + "_north.png"),
                                                         (self.CELL_SIZE, self.CELL_SIZE - 5))
        self.east_collaborator = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/" + scheme + "_east.png"),
                                                        (self.CELL_SIZE, self.CELL_SIZE - 5))
        self.west_collaborator = pygame.transform.flip(self.east_collaborator, 1, 0)

    def set_fixed_icon_scheme_collaborator(self):
        scheme = 'bulldozer'
        self.south_collaborator = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/" + scheme + "_south.png"),
                                                         (self.CELL_SIZE, self.CELL_SIZE - 5))
        self.north_collaborator = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/" + scheme + "_north.png"),
                                                         (self.CELL_SIZE, self.CELL_SIZE - 5))
        self.east_collaborator = pygame.transform.scale(pygame.image.load("robotaxi_integration/icon/" + scheme + "_east.png"),
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

            # grid_surface.blit(self.arrow_straight_east, (cell_size * 13 - 5, cell_size * 7 + 25))
            grid_surface.blit(self.arrow_uturn_east, (cell_size * 13 - 5, cell_size * 7 + 20))
            grid_surface.blit(self.arrow_noright_east, (cell_size * 13 - 5, cell_size * 8 + 5))

            grid_surface.blit(self.arrow_right_west, (cell_size * 11 + 15, cell_size * 17 + 20))
            grid_surface.blit(self.arrow_uturn_west, (cell_size * 11 + 15, cell_size * 16 + 30))
            grid_surface.blit(self.arrow_straight_west, (cell_size * 18 + 15, cell_size * 17 + 20))
            grid_surface.blit(self.arrow_right_west, (cell_size * 18 + 15, cell_size * 16 + 20))

            # grid_surface.blit(self.arrow_left_south, (cell_size * 6 + 20, cell_size * 14 - 15))
            grid_surface.blit(self.arrow_uturn_left_south, (cell_size * 6 + 20, cell_size * 14 - 15))

            grid_surface.blit(self.arrow_rightonly_south, (cell_size * 5 + 20, cell_size * 14 - 15))
        if create_building:
            grid_surface.blit(self.taxibuilding_icon, (cell_size * 1 + 30, cell_size * 20 + 20))
            grid_surface.blit(self.shop_icon, (cell_size * 10 + 25, cell_size * 20 + 20))
            grid_surface.blit(self.court_icon, (cell_size * 18 + 30, cell_size * 20 + 20))

            grid_surface.blit(self.school_icon, (cell_size * 18 + 30, cell_size * 1 + 20))
            grid_surface.blit(self.park_icon, (cell_size * 11 + 40, cell_size * 10 + 20))
            grid_surface.blit(self.office_icon, (cell_size * 11 + 40, cell_size * 12 + 60))
            grid_surface.blit(self.hospital_icon, (cell_size * 18 + 30, cell_size * 10 + 20))
            grid_surface.blit(self.taxibuilding_icon, (cell_size * 1 + 60, cell_size * 10 + 20))
            grid_surface.blit(self.restaurant_icon, (cell_size * 1 + 60, cell_size * 12 + 100))
            grid_surface.blit(self.taxibuilding_icon, (cell_size * 20 + 10, cell_size * 12 + 100))

            grid_surface.blit(self.police_station_icon, (cell_size * 11 + 60, cell_size * 1 + 20))
            grid_surface.blit(self.stadium_icon, (cell_size * 1 + 40, cell_size * 1 + 10))

            if self.trees:
                grid_surface.blit(self.tree_icon, (cell_size * 20, cell_size * 1 + 10))
                grid_surface.blit(self.tree_icon, (cell_size * 20, cell_size * 1 + 30))
                grid_surface.blit(self.tree_icon, (cell_size * 20, cell_size * 1 + 50))
                grid_surface.blit(self.tree_icon, (cell_size * 20, cell_size * 1 + 70))

                grid_surface.blit(self.tree_icon, (cell_size * 21, cell_size * 1 + 10))
                grid_surface.blit(self.tree_icon, (cell_size * 21, cell_size * 1 + 30))
                grid_surface.blit(self.tree_icon, (cell_size * 21, cell_size * 1 + 50))
                grid_surface.blit(self.tree_icon, (cell_size * 21, cell_size * 1 + 70))

                # restaurant tree
                grid_surface.blit(self.tree_icon, (cell_size * 3 + 0, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 3 - 20, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 3 - 40, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 3 - 60, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 3 - 80, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 3 - 100, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 3 - 120, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 3 - 140, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 3 - 160, cell_size * 10 + 160))

                # hospital tree
                grid_surface.blit(self.tree_icon, (cell_size * 20, cell_size * 10 + 20))
                grid_surface.blit(self.tree_icon, (cell_size * 20, cell_size * 10 + 40))
                grid_surface.blit(self.tree_icon, (cell_size * 20, cell_size * 10 + 60))
                grid_surface.blit(self.tree_icon, (cell_size * 20, cell_size * 10 + 80))
                grid_surface.blit(self.tree_icon, (cell_size * 20, cell_size * 10 + 100))
                grid_surface.blit(self.tree_icon, (cell_size * 20, cell_size * 10 + 120))
                grid_surface.blit(self.tree_icon, (cell_size * 20, cell_size * 10 + 140))
                grid_surface.blit(self.tree_icon, (cell_size * 20, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 21, cell_size * 10 + 20))
                grid_surface.blit(self.tree_icon, (cell_size * 21, cell_size * 10 + 40))
                grid_surface.blit(self.tree_icon, (cell_size * 21, cell_size * 10 + 60))
                grid_surface.blit(self.tree_icon, (cell_size * 21, cell_size * 10 + 80))
                grid_surface.blit(self.tree_icon, (cell_size * 21, cell_size * 10 + 100))
                grid_surface.blit(self.tree_icon, (cell_size * 21, cell_size * 10 + 120))
                grid_surface.blit(self.tree_icon, (cell_size * 21, cell_size * 10 + 140))
                grid_surface.blit(self.tree_icon, (cell_size * 21, cell_size * 10 + 160))

                grid_surface.blit(self.tree_icon, (cell_size * 20 + 80, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 20 + 60, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 20 + 40, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 20 + 20, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 20, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 20 - 20, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 20 - 40, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 20 - 60, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 20 - 80, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 20 - 100, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 20 - 120, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 20 - 140, cell_size * 10 + 160))

                grid_surface.blit(self.tree_icon, (cell_size * 18, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 18, cell_size * 10 + 180))
                grid_surface.blit(self.tree_icon, (cell_size * 18, cell_size * 10 + 200))
                grid_surface.blit(self.tree_icon, (cell_size * 18, cell_size * 10 + 220))
                grid_surface.blit(self.tree_icon, (cell_size * 18, cell_size * 10 + 240))
                grid_surface.blit(self.tree_icon, (cell_size * 18, cell_size * 10 + 260))
                grid_surface.blit(self.tree_icon, (cell_size * 18, cell_size * 10 + 280))
                grid_surface.blit(self.tree_icon, (cell_size * 18, cell_size * 10 + 300))
                grid_surface.blit(self.tree_icon, (cell_size * 18, cell_size * 10 + 320))

                grid_surface.blit(self.tree_icon, (cell_size * 19, cell_size * 10 + 160))
                grid_surface.blit(self.tree_icon, (cell_size * 19, cell_size * 10 + 180))
                grid_surface.blit(self.tree_icon, (cell_size * 19, cell_size * 10 + 200))
                grid_surface.blit(self.tree_icon, (cell_size * 19, cell_size * 10 + 220))
                grid_surface.blit(self.tree_icon, (cell_size * 19, cell_size * 10 + 240))
                grid_surface.blit(self.tree_icon, (cell_size * 19, cell_size * 10 + 260))
                grid_surface.blit(self.tree_icon, (cell_size * 19, cell_size * 10 + 280))
                grid_surface.blit(self.tree_icon, (cell_size * 19, cell_size * 10 + 300))
                grid_surface.blit(self.tree_icon, (cell_size * 19, cell_size * 10 + 320))

                # court tree

                grid_surface.blit(self.tree_icon, (cell_size * 20, cell_size * 19 + 100))
                grid_surface.blit(self.tree_icon, (cell_size * 20, cell_size * 19 + 120))
                grid_surface.blit(self.tree_icon, (cell_size * 20, cell_size * 19 + 140))
                grid_surface.blit(self.tree_icon, (cell_size * 20, cell_size * 19 + 160))

                grid_surface.blit(self.tree_icon, (cell_size * 21, cell_size * 19 + 100))
                grid_surface.blit(self.tree_icon, (cell_size * 21, cell_size * 19 + 120))
                grid_surface.blit(self.tree_icon, (cell_size * 21, cell_size * 19 + 140))
                grid_surface.blit(self.tree_icon, (cell_size * 21, cell_size * 19 + 160))

                # shop tree
                grid_surface.blit(self.tree_icon, (cell_size * 12, cell_size * 19 + 100))
                grid_surface.blit(self.tree_icon, (cell_size * 12, cell_size * 19 + 120))
                grid_surface.blit(self.tree_icon, (cell_size * 12, cell_size * 19 + 140))
                grid_surface.blit(self.tree_icon, (cell_size * 12, cell_size * 19 + 160))

                grid_surface.blit(self.tree_icon, (cell_size * 13, cell_size * 19 + 100))
                grid_surface.blit(self.tree_icon, (cell_size * 13, cell_size * 19 + 120))
                grid_surface.blit(self.tree_icon, (cell_size * 13, cell_size * 19 + 140))
                grid_surface.blit(self.tree_icon, (cell_size * 13, cell_size * 19 + 160))

                grid_surface.blit(self.tree_icon, (cell_size * 9, cell_size * 19 + 100))
                grid_surface.blit(self.tree_icon, (cell_size * 9, cell_size * 19 + 120))
                grid_surface.blit(self.tree_icon, (cell_size * 9, cell_size * 19 + 140))
                grid_surface.blit(self.tree_icon, (cell_size * 9, cell_size * 19 + 160))

                grid_surface.blit(self.tree_icon, (cell_size * 8, cell_size * 19 + 100))
                grid_surface.blit(self.tree_icon, (cell_size * 8, cell_size * 19 + 120))
                grid_surface.blit(self.tree_icon, (cell_size * 8, cell_size * 19 + 140))
                grid_surface.blit(self.tree_icon, (cell_size * 8, cell_size * 19 + 160))

                grid_surface.blit(self.tree_icon, (cell_size * 7, cell_size * 19 + 100))
                grid_surface.blit(self.tree_icon, (cell_size * 7, cell_size * 19 + 120))
                grid_surface.blit(self.tree_icon, (cell_size * 7, cell_size * 19 + 140))
                grid_surface.blit(self.tree_icon, (cell_size * 7, cell_size * 19 + 160))

                grid_surface.blit(self.tree_icon, (cell_size * 6, cell_size * 19 + 100))
                grid_surface.blit(self.tree_icon, (cell_size * 6, cell_size * 19 + 120))
                grid_surface.blit(self.tree_icon, (cell_size * 6, cell_size * 19 + 140))
                grid_surface.blit(self.tree_icon, (cell_size * 6, cell_size * 19 + 160))

                grid_surface.blit(self.tree_icon, (cell_size * 5, cell_size * 19 + 100))
                grid_surface.blit(self.tree_icon, (cell_size * 5, cell_size * 19 + 120))
                grid_surface.blit(self.tree_icon, (cell_size * 5, cell_size * 19 + 140))
                grid_surface.blit(self.tree_icon, (cell_size * 5, cell_size * 19 + 160))

                grid_surface.blit(self.tree_icon, (cell_size * 4, cell_size * 19 + 100))
                grid_surface.blit(self.tree_icon, (cell_size * 4, cell_size * 19 + 120))
                grid_surface.blit(self.tree_icon, (cell_size * 4, cell_size * 19 + 140))
                grid_surface.blit(self.tree_icon, (cell_size * 4, cell_size * 19 + 160))

                grid_surface.blit(self.tree_icon, (cell_size * 3, cell_size * 19 + 100))
                grid_surface.blit(self.tree_icon, (cell_size * 3, cell_size * 19 + 120))
                grid_surface.blit(self.tree_icon, (cell_size * 3, cell_size * 19 + 140))
                grid_surface.blit(self.tree_icon, (cell_size * 3, cell_size * 19 + 160))

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

    def render(self, agent_idx=None, counter=None, head_update=False):
        """ Draw the entire game frame. """
        if not agent_idx:
            self.screen.fill(Colors.SCREEN_BACKGROUND)
            self.screen.blit(self.surface, (0, 0))

        num_font = pygame.font.Font("robotaxi_integration/fonts/gyparody_tf.ttf", int(24 * (self.CELL_SIZE / 40.0)))

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

        for x in range(self.env.field.size):
            for y in range(self.env.field.size):
                self.render_cell(x, y, agent_idx, counter, head_update=head_update)

    def transition_animation(self, x_v, y_v, x0_v, y0_v, x_p, y_p, x0_p, y0_p, interpolate_idx, counter):

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
                if (interpolate_idx == 1) and (self.trialends_countdown_rounds == 19):
                    self.button_pressed = False
                    self.button_pressed_value = None
                    self.reward_vehicle = self.env.rewards["accident"]
                    self.accident_scoreboard = -10
                    self.event_type = 'vehicle_accident'
                    self.timestep_result = self.env.timestep_vehicle(agent_index=i, rwd=self.reward_vehicle)


        pedestrians_coords = []
        # if counter % 2 == 0:
        for i in range(len(self.env.pedestrians)):
            if self.is_on_the_zebra_crossing_pedestrian(coord=self.env.pedestrians[i].curr_head):
                self.speed_pedestrian[i] = 2

            if not self.env.pedestrians[i].lock:
                imm_coords = pygame.Rect(
                    5 + x0_p[i] * self.CELL_SIZE + (counter % self.speed_pedestrian[i]) * (
                            x_p[i] - x0_p[i]) * self.CELL_SIZE // self.speed_pedestrian[i] + (
                            x_p[i] - x0_p[i]) * interpolate_idx * self.CELL_SIZE // (
                            self.intermediate_frames * self.speed_pedestrian[i]),
                    5 + y0_p[i] * self.CELL_SIZE + (counter % self.speed_pedestrian[i]) * (
                            y_p[i] - y0_p[i]) * self.CELL_SIZE // self.speed_pedestrian[i] + (
                            y_p[i] - y0_p[i]) * interpolate_idx * self.CELL_SIZE // (
                            self.intermediate_frames * self.speed_pedestrian[i]),
                    self.CELL_SIZE,
                    self.CELL_SIZE,
                )
            else:
                self.event_type = 'pedestrian_accident'  # to overwrite the vehicle accident if exists
                imm_coords = self.env.pedestrians[i].lock_pos

            self.screen.blit(self.env.pedestrians[i].curr_icon, imm_coords)

            if self.speed_pedestrian[i] == 1:
                color = Colors.RED
            elif self.speed_pedestrian[i] == 2:
                color = Colors.RED
            elif self.speed_pedestrian[i] == 4:
                color = Colors.GREEN
            elif self.speed_pedestrian[i] == 8:
                color = Colors.GREEN
            elif self.env.pedestrians[i].lock:
                color = Colors.RED

            pygame.draw.rect(self.screen, color, imm_coords, width=5)
            font = pygame.font.Font(None, 28)
            name = self.human_schemes[i]
            name_surface = font.render(name, True, (255, 255, 255))
            self.screen.blit(name_surface, imm_coords.move(5, 5))
            if self.keep_flag[i] or self.env.pedestrians[i].lock:
                speed = '0'
            else:
                speed = str(int(40 / self.speed_pedestrian[i]))
            speed_surface = font.render(speed + ' km/h', True, (255, 255, 255))
            self.screen.blit(speed_surface, imm_coords.move(5, 60))

            pedestrians_coords.append(imm_coords)
        vehicles_coords = []

        for i in range(len(self.env.vehicles)):
            if not self.env.vehicles[i].lock:
                imm_coords = pygame.Rect(
                    x0_v[i] * self.CELL_SIZE + (counter % self.speed_vehicle[i]) * (
                            x_v[i] - x0_v[i]) * self.CELL_SIZE // self.speed_vehicle[i] + (
                            x_v[i] - x0_v[i]) * interpolate_idx * self.CELL_SIZE // (
                            self.intermediate_frames * self.speed_vehicle[i]),
                    y0_v[i] * self.CELL_SIZE + (counter % self.speed_vehicle[i]) * (
                            y_v[i] - y0_v[i]) * self.CELL_SIZE // self.speed_vehicle[i] + (
                            y_v[i] - y0_v[i]) * interpolate_idx * self.CELL_SIZE // (
                            self.intermediate_frames * self.speed_vehicle[i]),
                    self.CELL_SIZE,
                    self.CELL_SIZE,
                )
                # import math
                # def circular_arc_interpolation(x0, y0, x, y, radius, angle_start, angle_end, t):
                #     # Calculate the current angle based on interpolation factor 't'
                #     angle_current = angle_start + t * (angle_end - angle_start)
                #
                #     # Calculate the x, y position along the arc
                #     arc_x = -radius * math.cos(angle_current)
                #     arc_y = -radius * math.sin(angle_current)
                #
                #     # Shift the arc to the starting point
                #     new_x = (x0 + arc_x) * self.CELL_SIZE
                #     new_y = (y0 + arc_y) * self.CELL_SIZE
                #     print("arc_x, arc_y", arc_x, arc_y)
                #     return new_x, new_y
                #
                # def get_arc_parameters(x0, y0, x, y):
                #     # Calculate the midpoint between the start and end points
                #     mid_x = (x0 + x) / 2
                #     mid_y = (y0 + y) / 2
                #
                #     # Assume the center of the arc is perpendicular to the line connecting start and end
                #     # (This is a simplification, you can improve the center calculation if necessary)
                #     # if x0 < x and y0 < y:
                #     #     center_x = mid_x - 0.2
                #     #     center_y = mid_y + 0.2
                #     # elif x0 < x and y0 > y:
                #     #     center_x = mid_x - 0.2
                #     #     center_y = mid_y - 0.2
                #     # elif x0 > x and y0 < y:
                #     #     center_x = mid_x + 0.2
                #     #     center_y = mid_y + 0.2
                #     # elif x0 > x and y0 > y:
                #     #     center_x = mid_x + 0.2
                #     #     center_y = mid_y - 0.2
                #     center_x = mid_x
                #     center_y = mid_y # Adjust this based on the curve tightness you want
                #
                #     # Calculate the radius of the arc
                #     radius = math.sqrt((center_x - x0) ** 2 + (center_y - y0) ** 2)
                #
                #     # Calculate the start and end angles in radians
                #     angle_start = math.atan2(y0 - center_y, x0 - center_x)
                #     angle_end = math.atan2(y - center_y, x - center_x)
                #
                #     return center_x, center_y, radius, angle_start, angle_end
                # if (x0_v[i] != x_v[i] and y0_v[i] != y_v[i]) and not self.is_in_the_lane(coord=(x0_v[i], y0_v[i])):
                #     center_x, center_y, radius, angle_start, angle_end = get_arc_parameters(x0_v[i], y0_v[i], x_v[i], y_v[i])
                #
                #     # Calculate the interpolation factor (t between 0 and 1)
                #     t = (interpolate_idx + (counter % self.speed_vehicle[i]) * self.intermediate_frames) / (self.intermediate_frames * self.speed_vehicle[i])
                #     # Get the new position along the arc
                #     new_x, new_y = circular_arc_interpolation(center_x, center_y, x_v[i], y_v[i], radius, angle_start, angle_end, t)
                #     # Create the pygame Rect for the vehicle's position
                #     imm_coords = pygame.Rect(new_x, new_y, self.CELL_SIZE, self.CELL_SIZE)

                # else:
                #     imm_coords = pygame.Rect(
                #         x0_v[i] * self.CELL_SIZE + (counter % self.speed_vehicle[i]) * (x_v[i] - x0_v[i]) * self.CELL_SIZE // self.speed_vehicle[i] + (x_v[i] - x0_v[i]) * interpolate_idx * self.CELL_SIZE // (self.intermediate_frames * self.speed_vehicle[i]),
                #         y0_v[i] * self.CELL_SIZE + (counter % self.speed_vehicle[i]) * (y_v[i] - y0_v[i]) * self.CELL_SIZE // self.speed_vehicle[i] + (y_v[i] - y0_v[i]) * interpolate_idx * self.CELL_SIZE // (self.intermediate_frames * self.speed_vehicle[i]),
                #         self.CELL_SIZE,
                #         self.CELL_SIZE,
                #     )

            else:
                imm_coords = self.env.vehicles[i].lock_pos

            num_turn_img = 18
            total_frame = self.intermediate_frames * self.speed_vehicle[i]
            curr_frame = interpolate_idx + (counter % self.speed_vehicle[i]) * self.intermediate_frames

            t = total_frame // num_turn_img

            curr_turn_img_idx = curr_frame // t
            if curr_turn_img_idx >= num_turn_img: curr_turn_img_idx = num_turn_img - 1
            # print('curr_frame ', curr_frame, ' t', t, ' curr_frame // t', curr_turn_img_idx)

            north_right = []
            north_left = []
            west_right = []
            west_left = []
            east_left = []
            east_right = []
            south_right = []
            south_left = []

            if self.env.vehicles[i].icon_type == 0:
                turn_icon = self.bus_turn_icon
            elif self.env.vehicles[i].icon_type == 1:
                turn_icon = self.pickup_turn_icon
            elif self.env.vehicles[i].icon_type == 2:
                turn_icon = self.truck_turn_icon
            elif self.env.vehicles[i].icon_type == 3:
                turn_icon = self.bulldozer_turn_icon
            for j in range(5, 95, 5):
                north_right.append(turn_icon[j])
            for j in range(-5, -95, -5):
                north_left.append(turn_icon[j])
            for j in range(-85, 5, 5):
                west_right.append(turn_icon[j])
            for j in range(-95, -185, -5):
                west_left.append(turn_icon[j])
            for j in range(95, 185, 5):
                east_right.append(turn_icon[j])
            for j in range(85, -5, -5):
                east_left.append(turn_icon[j])
            for j in range(175, 85, -5):
                south_left.append(turn_icon[j])
            for j in range(-175, -85, 5):
                south_right.append(turn_icon[j])

            if self.env.vehicles[i].pause or self.trialends_countdown:
                curr_turn_img_idx = 17

            if self.action_vehicle[i] == 2 and self.env.vehicles[
                i].direction == SnakeDirection.EAST and not self.is_in_the_lane(coord=(x0_v[i], y0_v[i])):
                self.screen.blit(north_right[curr_turn_img_idx], imm_coords)
            elif self.action_vehicle[i] == 1 and self.env.vehicles[
                i].direction == SnakeDirection.WEST and not self.is_in_the_lane(coord=(x0_v[i], y0_v[i])):
                self.screen.blit(north_left[curr_turn_img_idx], imm_coords)
            elif self.action_vehicle[i] == 1 and self.env.vehicles[
                i].direction == SnakeDirection.NORTH and not self.is_in_the_lane(coord=(x0_v[i], y0_v[i])):
                self.screen.blit(east_left[curr_turn_img_idx], imm_coords)
            elif self.action_vehicle[i] == 2 and self.env.vehicles[
                i].direction == SnakeDirection.NORTH and not self.is_in_the_lane(coord=(x0_v[i], y0_v[i])):
                self.screen.blit(west_right[curr_turn_img_idx], imm_coords)
            elif self.action_vehicle[i] == 2 and self.env.vehicles[
                i].direction == SnakeDirection.SOUTH and not self.is_in_the_lane(coord=(x0_v[i], y0_v[i])):
                self.screen.blit(east_right[curr_turn_img_idx], imm_coords)
            elif self.action_vehicle[i] == 1 and self.env.vehicles[
                i].direction == SnakeDirection.SOUTH and not self.is_in_the_lane(coord=(x0_v[i], y0_v[i])):
                self.screen.blit(west_left[curr_turn_img_idx], imm_coords)
            elif self.action_vehicle[i] == 1 and self.env.vehicles[
                i].direction == SnakeDirection.EAST and not self.is_in_the_lane(coord=(x0_v[i], y0_v[i])):
                self.screen.blit(south_left[curr_turn_img_idx], imm_coords)
            elif self.action_vehicle[i] == 2 and self.env.vehicles[
                i].direction == SnakeDirection.WEST and not self.is_in_the_lane(coord=(x0_v[i], y0_v[i])):
                self.screen.blit(south_right[curr_turn_img_idx], imm_coords)
            else:
                self.screen.blit(self.env.vehicles[i].curr_icon, imm_coords)


            if self.speed_vehicle[i] == 1:
                color = Colors.RED
            elif self.speed_vehicle[i] == 2:
                color = Colors.RED
            elif self.speed_vehicle[i] == 4:
                color = Colors.GREEN
            if self.env.vehicles[i].pause:
                color = Colors.GREEN

            if self.pickup_vehicle[i]:
                color = Colors.BLUE

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
            if self.pickup_vehicle[i]:
                name1 = "Reserved"
                name_surface = font.render(name1, True, (255, 255, 255))
                self.screen.blit(name_surface, imm_coords.move(5, -15))
            speed_surface = font.render(speed + ' km/h', True, (255, 255, 255))
            self.screen.blit(speed_surface, imm_coords.move(5, 60))

            vehicles_coords.append(imm_coords)

        if (interpolate_idx == 1) and (self.trialends_countdown_rounds == 19):
            if self.event_type == 'vehicle_accident':
                print("E 4 Vehicle-Vehicle Collision")
                self.outlet.push_sample(["E 4"])
                self.receive_send_rumble()
            elif self.event_type == 'pedestrian_accident':
                print("E 5 Pedestrian-Vehicle Collision")
                self.outlet.push_sample(["E 5"])
                self.receive_send_rumble()

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
                            # pygame.mixer.music.load("robotaxi_integration/sound/background2.mp3")
                            # pygame.mixer.music.set_volume(0.1)
                            # pygame.mixer.music.play(-1, 0.0)
                            pass
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

        visit_list = []
        old_direction = self.env.vehicles[agent_idx].direction
        self.env.choose_action_multiagents(action, self.env.vehicles, agent_idx)

        if action == 0:
            visit_list.append(self.env.vehicles[agent_idx].peek_next_move())
        if action == 1:
            visit_list.append(self.env.vehicles[agent_idx].head + old_direction)
            visit_list.append(self.env.vehicles[agent_idx].peek_next_move() + old_direction)
            self.env.choose_action_multiagents(2, self.env.vehicles, agent_idx)
        if action == 2:
            visit_list.append(self.env.vehicles[agent_idx].head + old_direction)
            visit_list.append(self.env.vehicles[agent_idx].peek_next_move() + old_direction)
            self.env.choose_action_multiagents(1, self.env.vehicles, agent_idx)

        # flag = False
        flag_v, flag_p = False, False
        for i in range(len(self.env.vehicles)):
            if i != agent_idx:
                if self.env.vehicles[i].head in visit_list:
                    # print("Collision with other vehicles")
                    flag_v = True
        for i in range(len(self.env.pedestrians)):
            if i != agent_idx:
                if self.env.pedestrians[i].head in visit_list:
                    # print("Collision with pedestrians")
                    flag_p = True
        if flag_v:
            if np.random.rand() < 0.1:  # 0.1
                # print("=====================================")
                # print("Vehicle accident will happen at {}!!!".format(next_move, action))
                # print("=====================================")
                if not self.env.vehicles[agent_idx].pause:
                    return (False, False), 1
                # stopped, not good if start for an accident
        if flag_p:
            if np.random.rand() < 0.1:  # 0.3
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
        # last_head = self.env.vehicles[agent_idx].last_head
        current_head = self.env.vehicles[agent_idx].head
        current_head = (current_head[1], current_head[0])
        # print("Current Map", self.env.field)
        # print("Current action: ", action)
        # print("Current action map: ", action_map[current_head])
        if not action_map[current_head]:
            print("=====================================")

            raise IndexError("No action available for the current direction")
        if self.post_action_head_list[agent_idx]:
            self.post_action_head_list[agent_idx] = None
            # print("Last action is change lane, cannot change lane again")
            # print("=====================================")

            return 0, 0
        elif action not in action_map[current_head]:
            # choose a random action
            resampled_action = np.random.choice(action_map[current_head])
            # print("Resampled action: ", resampled_action)
            # print("=====================================")

            if self.is_in_the_lane(agent_idx) and resampled_action != 0:
                if (np.random.uniform(0, 1) < 0.2) or (0 not in action_map[current_head]):
                    # print("I MUST CHANGE LANE! I AM LUCKY")
                    # print("=====================================")

                    return resampled_action, 1
                else:
                    # print("=====================================")
                    return 0, 0
            return resampled_action, 0
        elif self.is_in_the_lane(agent_idx) and action != 0:
            if len(action_map[current_head]) == 1:  # very rare case
                # print("Only one action available, I MUST CHANGE LANE! I AM LUCKY")
                # print("=====================================")
                return action, 1
            elif np.random.uniform(0, 1) < 0.2:
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

    def is_in_the_lane(self, agent_idx=None, coord=None):

        if not coord:
            current_head = self.env.vehicles[agent_idx].head
            current_head = (current_head[1], current_head[0])

            for lane in LANE:
                if (lane[0][0] <= current_head[0] <= lane[1][0] and
                        lane[0][1] <= current_head[1] <= lane[1][1]):
                    return True
        else:
            coord = (coord[1], coord[0])
            for lane in LANE:
                if (lane[0][0] <= coord[0] <= lane[1][0] and
                        lane[0][1] <= coord[1] <= lane[1][1]):
                    return True
        return False

    def is_on_the_zebra_crossing_vehicle(self, agent_idx):

        current_head = self.env.vehicles[agent_idx].head
        current_head = (current_head[1], current_head[0])

        for lane in LANE_ZEBRA_CROSSING_V:
            if lane[0][0] <= current_head[0] <= lane[1][0] and lane[0][1] <= current_head[1] <= lane[1][1]:
                for i in range(len(self.env.pedestrians)):
                    p_head = self.env.pedestrians[i].head
                    p_head = (p_head[1], p_head[0])
                    if lane[0][0] <= p_head[0] <= lane[1][0] and lane[0][1] <= p_head[1] <= lane[1][1]:
                        return True
        return False

    def is_on_the_zebra_crossing_pedestrian(self, agent_idx=None, coord=None):

        if agent_idx:
            current_head = self.env.pedestrians[agent_idx].head
            current_head = (current_head[1], current_head[0])

            for lane in LANE_ZEBRA_CROSSING_P:
                if (lane[0][0] <= current_head[0] <= lane[1][0] and
                        lane[0][1] <= current_head[1] <= lane[1][1]):
                    return True
        elif coord:
            coord = (coord[1], coord[0])
            for lane in LANE_ZEBRA_CROSSING_P:
                if (lane[0][0] <= coord[0] <= lane[1][0] and
                        lane[0][1] <= coord[1] <= lane[1][1]):
                    return True
        return False

    def is_on_the_pickup_area(self, agent_idx):

        current_head = self.env.pedestrians[agent_idx].head  # last head
        current_head = (current_head[1], current_head[0])

        for lane in LANE_PICKUP:
            if lane[0][0] <= current_head[0] <= lane[1][0] and lane[0][1] <= current_head[1] <= lane[1][1]:
                for i in range(len(self.env.pedestrians)):
                    p_head = self.env.pedestrians[i].head
                    p_head = (p_head[1], p_head[0])
                    if lane[0][0] <= p_head[0] <= lane[1][0] and lane[0][1] <= p_head[1] <= lane[1][1]:
                        return True
        return False

    def find_pickup_car(self, agent_idx):

        des = 1000
        dis = -1
        for i in range(len(self.env.vehicles)):
            if self.env.vehicles[i].icon_type == 0:
                new_dis = np.sqrt(
                    (self.env.vehicles[i].head[0] - self.env.pedestrians[agent_idx].curr_head[0]) ** 2
                    + (self.env.vehicles[i].head[1] - self.env.pedestrians[agent_idx].curr_head[1]) ** 2)
                if new_dis < des:
                    des = new_dis
                    dis = i

        if dis != -1:
            self.pickup_vehicle[dis] = True
            self.pickup_vehicle_found = True

    def solve_pickup2destination_path(self, agent_idx):

        for i in range(len(self.env.pedestrians)):
            if self.env.pedestrians[i].lock and self.env.pedestrians[i].destination:
                des = self.env.pedestrians[i].destination

        curr_head = self.env.vehicles[agent_idx].head
        curr_head = (curr_head[1], curr_head[0])
        curr_direction = self.env.vehicles[agent_idx].direction

        dest_head = BUILDING_POS_PARKING[des]
        dest_head = (dest_head[1], dest_head[0])
        dest_direction = BUILDING_POS_PARKING_DIRECTION[des]

        solution = pathfinding((curr_head, curr_direction), (dest_head, dest_direction), agent='vehicle')

        return solution[1] if solution else None

    def solve_pickup_path_vehicle(self, agent_idx):

        for i in range(len(self.env.pedestrians)):
            if self.env.pedestrians[i].lock and self.env.pedestrians[i].destination:
                p_pos = self.env.pedestrians[i].head
        try:
            if 10 <= p_pos[1] <= 13 and p_pos[0] == 4:
                dest_head = p_pos + SnakeDirection.EAST
                dest_direction = SnakeDirection.SOUTH
            if p_pos[1] == 9 and 11 <= p_pos[0] <= 12:
                dest_head = p_pos + SnakeDirection.NORTH
                dest_direction = SnakeDirection.EAST
            if 11 <= p_pos[1] <= 14 and p_pos[0] == 10:
                dest_head = p_pos + SnakeDirection.WEST
                dest_direction = SnakeDirection.NORTH
            if p_pos[1] == 15 and 12 <= p_pos[0] <= 13:
                dest_head = p_pos + SnakeDirection.SOUTH
                dest_direction = SnakeDirection.WEST
            if p_pos[1] == 19 and 11 <= p_pos[0] <= 12:
                dest_head = p_pos + SnakeDirection.NORTH
                dest_direction = SnakeDirection.EAST
            if 11 <= p_pos[1] <= 14 and p_pos[0] == 14:
                dest_head = p_pos + SnakeDirection.EAST
                dest_direction = SnakeDirection.NORTH
            if 11 <= p_pos[1] <= 14 and p_pos[0] == 17:
                dest_head = p_pos + SnakeDirection.WEST
                dest_direction = SnakeDirection.NORTH
        except UnboundLocalError:
            print("Pedestrian not found")

        dest_head = (dest_head[1], dest_head[0])

        curr_head = self.env.vehicles[agent_idx].head
        curr_head = (curr_head[1], curr_head[0])
        curr_direction = self.env.vehicles[agent_idx].direction

        solution = pathfinding((curr_head, curr_direction), (dest_head, dest_direction), agent='vehicle')

        return solution[1] if solution else None

    def solve_pickup_path_pedestrian(self, agent_idx):

        curr_head = self.env.pedestrians[agent_idx].head
        curr_head = (curr_head[1], curr_head[0])
        curr_direction = self.env.pedestrians[agent_idx].direction

        des = 1000
        for lane in LANE_PICKUP:
            new_dis = np.sqrt((lane[0][0] - curr_head[0]) ** 2 + (lane[0][1] - curr_head[1]) ** 2)
            if new_dis < des:
                des = new_dis
                dest_head1 = lane[0]
                dest_head2 = lane[1]

        # dest_head = ((np.ceil(dest_head1[0] + dest_head2[0]) / 2), np.ceil((dest_head1[1] + dest_head2[1]) / 2))
        dest_head = ((dest_head1[0] + dest_head2[0]) // 2, (dest_head1[1] + dest_head2[1]) // 2)

        dest_head = (dest_head[1], dest_head[0])
        if 10 <= dest_head[1] <= 13 and dest_head[0] == 4:
            dest_direction = SnakeDirection.SOUTH
        if dest_head[1] == 9 and 11 <= dest_head[0] <= 12:
            dest_direction = SnakeDirection.EAST
        if 11 <= dest_head[1] <= 14 and dest_head[0] == 10:
            dest_direction = SnakeDirection.NORTH
        if dest_head[1] == 15 and 12 <= dest_head[0] <= 13:
            dest_direction = SnakeDirection.WEST
        if dest_head[1] == 19 and 11 <= dest_head[0] <= 12:
            dest_direction = SnakeDirection.EAST
        if 11 <= dest_head[1] <= 14 and dest_head[0] == 14:
            dest_direction = SnakeDirection.NORTH
        if 11 <= dest_head[1] <= 14 and dest_head[0] == 17:
            dest_direction = SnakeDirection.NORTH
        dest_head = (dest_head[1], dest_head[0])

        solution = pathfinding((curr_head, curr_direction), (dest_head, dest_direction), agent='pedestrian')

        return solution[1] if solution else None

    def post_action_head_adjustment(self, action, agent_idx):
        if action != 0:
            self.env.vehicles[agent_idx].peek_next_move()  # first move one step forward and then turn head
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

        start_text_font = pygame.font.Font("robotaxi_integration/fonts/gyparody_hv.ttf", int(21 * (self.CELL_SIZE / 10.0)))
        disp_text = start_text_font.render("Press <TouchPad> to Start", True, (220, 220, 220))
        self.screen.blit(disp_text, (self.CELL_SIZE * 1 + 20, self.CELL_SIZE * 10))
        pygame.display.update()
        self.handle_pause()

        # capture_thread1 = captureThread(0, participant=participant, data_dir='./user_study_data/', exp_id='original',
        #                                 test=self.test)
        # capture_thread1.start()
        try:
            for episode in range(num_episodes):
                print("============Episode: ", episode, "============")
                self.run_episode(episode)
                self.outlet.push_sample(["T 3"])
                self.screen.fill(Colors.SCREEN_BACKGROUND)
                self.screen.blit(self.surface, (0, 0))
                start_text_font = pygame.font.Font("robotaxi_integration/fonts/gyparody_hv.ttf",
                                                   int(21 * (self.CELL_SIZE / 10.0)))
                disp_text = start_text_font.render("Rest", True, (220, 220, 220))
                self.screen.blit(disp_text, (self.CELL_SIZE * 10 + 20, self.CELL_SIZE * 10))
                pygame.display.update()
                pygame.time.wait(10000)
                self.outlet.push_sample(["T 4"])            # capture_thread1.stop()
        except QuitRequestedError:
            pass
            # capture_thread1.stop()

    def run_episode(self, episode_num):
        """ Run the GUI player for a single episode. """
        self.outlet.push_sample(["T 1"])
        global frame_ct
        # Initialize the environment.
        self.timestep_watch.reset()
        self.timestep_result = self.env.new_episode()
        self.agent.begin_episode()

        self.base_update_interval = 8  # 1x speed  5km/h
        self.slow_update_interval = 4  # 1.5x speed  10km/h
        self.normal_update_interval = 2  # 2x speed  20km/h
        self.fast_update_interval = 1  # 4x speed  40km/h

        # self.speed_vehicle = [self.slow_update_interval] * len(self.env.vehicles)
        self.speed_vehicle = [self.slow_update_interval] * len(self.env.vehicles)
        self.speed_pedestrian = [self.base_update_interval] * len(self.env.pedestrians)

        self.action_vehicle = [None] * len(self.env.vehicles)
        self.action_pedestrian = [None] * len(self.env.pedestrians)

        self.pickup_vehicle = [False] * len(self.env.vehicles)
        self.pickup_pedestrian = [False] * len(self.env.pedestrians)

        self.post_action_head_list = [None] * len(self.env.vehicles)
        self.keep_action = [None] * len(self.env.pedestrians)
        self.keep_flag = [False] * len(self.env.pedestrians)

        is_human_agent = isinstance(self.agent, HumanAgent)
        self.timestep_delay = self.HUMAN_TIMESTEP_DELAY if is_human_agent else self.AI_TIMESTEP_DELAY

        self.render_initial_traffic_lights()
        start_text_font = pygame.font.Font("robotaxi_integration/fonts/gyparody_hv.ttf", int(21 * (self.CELL_SIZE / 40.0)))
        disp_text = start_text_font.render("Press <TouchPad> to Start", True, (220, 220, 220))
        # self.screen.blit(disp_text, (self.screen_size[0] // 2 - disp_text.get_width() // 2 , self.screen_size[1] // 2 - disp_text.get_height()//2 ))
        self.screen.blit(disp_text, (self.CELL_SIZE * 22 + 20, self.CELL_SIZE * 20))
        pygame.display.update()

        # Main game loop.
        self.running = True
        action_selected = False
        episode_start = True
        counter = 0

        self.trialends_countdown = False
        self.trialends_countdown_rounds = 20
        self.event_type = None
        self.pickup_vehicle_found = False
        self.pickup_path_vehicle = None
        self.destination_path_vehicle = None
        self.pickup_path_pedestrian = None
        self.head_to_destination = False
        self.pickup_countdown_vehicle = 0
        self.pickup_countdown_pedestrian = 0
        self.taskdone_countdown_vehicle = -1
        violation_lock_countdown_rounds = 20
        visual_feedback_countdown_rounds_v = 3
        visual_feedback_countdown_rounds_p = 3
        total_frames = 0
        self.reward_vehicle = 0
        self.reward_pedestrian = 0
        self.violation_scoreboard = 0
        self.accident_scoreboard = 0
        self.button_pressed = False
        self.button_pressed_value = None

        self.buffer = None
        if not image_buffer1:
            self.buffer = image_buffer1
            print("Buffer to list 1")
        elif not image_buffer2:
            self.buffer = image_buffer2
            print("Buffer to list 2")

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

            self.taxi_service_done = False
            # print("Accident type: ", self.accident_type)
            if self.event_lock:
                violation_lock_countdown_rounds -= 1
                # print("Violation lock countdown: ", violation_lock_countdown_rounds)
                # print("Accident type: ", self.accident_type)
            if violation_lock_countdown_rounds == 0 and not self.trialends_countdown:
                self.event_lock = False
                violation_lock_countdown_rounds = 20
                self.event_type = None
                self.button_pressed = False
                self.button_pressed_value = None
            frame_ct = self.frame_num

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
                for vehicle in self.env.vehicles: self.agent_generation(vehicle)
                for pedestrian in self.env.pedestrians: self.agent_generation(pedestrian)
                episode_start = False
            elif (pygame.time.get_ticks() - self.pause_time) % 3 == 0 and len(
                    self.env.vehicles) < 10:  # every 5 seconds generate a new agent

                if counter % self.slow_update_interval == 0:
                    self.env.new_agent_vehicle()
                    self.post_action_head_list.append(None)
                    self.action_vehicle.append(None)
                    self.pickup_vehicle.append(False)
                    # self.speed_vehicle.append(self.fast_update_interval)
                    self.speed_vehicle.append(self.slow_update_interval)

                if (counter % self.base_update_interval == 0) and len(self.env.pedestrians) < 6:
                    self.env.new_agent_pedestrian()
                    self.keep_action.append(None)
                    self.keep_flag.append(False)
                    self.speed_pedestrian.append(self.base_update_interval)
                    self.pickup_pedestrian.append(False)

            # try to switch the speed of some vehicles during movement
            if counter % 20 == 0:  # make sure not the in the middle of the movement
                speed, agent_idx = np.random.choice([1, 4]), np.random.choice(range(len(self.env.vehicles)))
                old_speed = self.speed_vehicle[agent_idx]
                if old_speed == 1 or old_speed == 4:
                    self.speed_vehicle[agent_idx] = 2
                    # print("Vehicle {} speed changed from {} to {}".format(agent_idx, old_speed, 2))
                else:
                    self.speed_vehicle[agent_idx] = speed
                    # print("Vehicle {} speed changed from {} to {}".format(agent_idx, old_speed, speed))

            timestep_timed_out = self.timestep_watch.time() >= self.timestep_delay
            if timestep_timed_out:
                self.timestep_watch.reset()
                self.frame_num = self.frame_num + 1
                total_frames += 1
                self.sound_played = False

                agent2pop_vehicle = []
                for agent_idx in range(len(self.env.vehicles)):
                    if not self.env.vehicles[agent_idx].lock and self.trialends_countdown_rounds > 4:
                        # if self.pickup_vehicle[agent_idx]:
                        #     self.speed_vehicle[agent_idx] = self.fast_update_interval
                        #     if not self.pickup_path_vehicle and not self.head_to_destination and not self.taxi_service_done:
                        #         self.pickup_path_vehicle = self.solve_pickup_path_vehicle(agent_idx)
                        #         if not self.pickup_path_vehicle:
                        #             self.pickup_vehicle[agent_idx] = False
                        #             self.pickup_vehicle_found = False
                        #             self.speed_vehicle[agent_idx] = self.slow_update_interval
                        #     if not self.destination_path_vehicle and self.head_to_destination and self.taskdone_countdown_vehicle == -1:
                        #         self.destination_path_vehicle = self.solve_pickup2destination_path(agent_idx)

                        if counter % self.speed_vehicle[agent_idx] == 0:
                            # if self.pickup_vehicle[agent_idx] and self.pickup_countdown_vehicle:
                            #     self.pickup_countdown_vehicle -= 1
                            #     self.pickup_countdown_pedestrian -= 1
                            #     continue
                            # if self.pickup_vehicle[
                            #     agent_idx] and self.taskdone_countdown_vehicle != 0 and self.taskdone_countdown_vehicle != -1:
                            #     self.taskdone_countdown_vehicle -= 1
                            #     continue

                            # if self.taskdone_countdown_vehicle == 0:
                            #     self.pickup_vehicle[agent_idx] = False
                            #     self.head_to_destination = False
                            #     self.taxi_service_done = True
                            #     self.trialends_countdown = True
                            self.action_vehicle[agent_idx] = self.agent.act(self.timestep_result.observation,
                                                                            self.timestep_result.reward)
                            self.action_vehicle[agent_idx], switch_lane_by_force = self.validate_traffic_rule_vehicles(
                                self.action_vehicle[agent_idx], agent_idx)

                            # if self.pickup_path_vehicle and self.pickup_vehicle[agent_idx]:
                            #     self.action_vehicle[agent_idx] = self.pickup_path_vehicle[0]
                            #
                            # if self.destination_path_vehicle and self.head_to_destination and self.pickup_vehicle[
                            #     agent_idx]:
                            #     self.action_vehicle[agent_idx] = self.destination_path_vehicle[0]

                            wait4greenlight, redlight_force = self.interact_traffic_lights_vehicles(
                                self.action_vehicle[agent_idx], agent_idx)
                            collide_w_agents, accident_force = self.interact_other_agents_vehicles(
                                self.action_vehicle[agent_idx], agent_idx)

                            # if (self.pickup_path_vehicle or self.destination_path_vehicle) and self.pickup_vehicle[
                            #     agent_idx]:
                            #     redlight_force = 0
                            #     accident_force = 0

                            if accident_force or redlight_force:
                                if not self.event_lock and not self.trialends_countdown:
                                    wait4greenlight = False
                                    self.event_lock = True
                                    print("E 2 Signal Violation (Vehicle)")
                                    # self.logger.info("E 2 Signal Violation (Vehicle)")
                                    self.reward_vehicle = self.env.rewards["violation"]
                                    self.violation_scoreboard = -5
                                    self.event_type = 'vehicle_violation'  # vv
                                    self.outlet.push_sample(["E 2"])
                                    self.receive_send_rumble()

                                    self.visual_feedback_v = agent_idx
                                elif accident_force:
                                    collide_w_agents = [True, True]
                                elif redlight_force:
                                    wait4greenlight = True

                            if not wait4greenlight and not (collide_w_agents[0] or collide_w_agents[1]):
                                if (self.speed_vehicle[agent_idx] == 1) and not self.event_lock and not self.trialends_countdown:
                                    # if self.speed_vehicle[agent_idx] == 1: print("Fast driving")
                                    # if switch_lane_by_force: print("Switch lane by force")
                                    # if self.is_on_the_zebra_crossing_vehicle(agent_idx): print("On the zebra crossing")
                                    self.event_lock = True
                                    print("E 1 Speed Violation (Vehicle)")
                                    # self.logger.info("E 1 Speed Violation (Vehicle)")
                                    self.outlet.push_sample(["E 1"])
                                    self.receive_send_rumble()

                                    self.visual_feedback_v = agent_idx
                                    self.event_type = 'normal'

                                # if self.pickup_path_vehicle and self.pickup_vehicle[agent_idx]:
                                #     self.action_vehicle[agent_idx] = self.pickup_path_vehicle.pop(0)
                                #     if not self.pickup_path_vehicle:
                                #         self.head_to_destination = True
                                #         self.pickup_countdown_vehicle = 2
                                #         self.pickup_countdown_pedestrian = 2
                                #         # self.env.vehicles[agent_idx].pause = True
                                # if self.destination_path_vehicle and self.pickup_vehicle[agent_idx]:
                                #     self.action_vehicle[agent_idx] = self.destination_path_vehicle.pop(0)
                                #     if not self.destination_path_vehicle:
                                #         self.taskdone_countdown_vehicle = 3

                                self.post_action_head_list[agent_idx] = self.post_action_head_adjustment(
                                    self.action_vehicle[agent_idx], agent_idx)
                                self.timestep_result = self.env.timestep_vehicle(agent_index=agent_idx,
                                                                                 rwd=self.reward_vehicle,
                                                                                 action=self.action_vehicle[agent_idx])
                                self.reward_vehicle = self.env.rewards["timestep"]
                                if self.post_action_head_list[agent_idx]:
                                    self.env.choose_action_multiagents(self.post_action_head_list[agent_idx],
                                                                       self.env.vehicles, agent_idx)

                                if self.timestep_result.is_episode_end and self.env.stats.termination_reason in [
                                    'hit_wall', 'hit_own_body']:
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
                    self.post_action_head_list.pop(agent2pop_vehicle[i])
                    self.speed_vehicle.pop(agent2pop_vehicle[i])
                    self.action_vehicle.pop(agent2pop_vehicle[i])
                    self.pickup_vehicle.pop(agent2pop_vehicle[i])

                for agent_idx in range(len(self.env.pedestrians)):
                    if not self.env.pedestrians[agent_idx].lock and self.trialends_countdown_rounds > 4:
                        # if counter >= taxi_round and agent_idx == taxi_pedestrian and self.env.pedestrians[
                        #     agent_idx].curr_head:
                        #     if des == 0:
                        #         for idx, (k, v) in enumerate(BUILDING_POS.items()):
                        #             new_dis = np.sqrt((BUILDING_POS[k][0] -
                        #                                self.env.pedestrians[agent_idx].curr_head[0]) ** 2 + (
                        #                                       BUILDING_POS[k][1] -
                        #                                       self.env.pedestrians[agent_idx].curr_head[1]) ** 2)
                        #             if new_dis > dis:
                        #                 dis = new_dis
                        #                 des = k
                        #             if k == "stadium": break  # not include the taxi buildings
                        #         self.env.pedestrians[agent_idx].destination = des
                        #         self.speed_pedestrian[agent_idx] = self.slow_update_interval
                        #         self.pickup_pedestrian[agent_idx] = True
                        # if self.pickup_pedestrian[agent_idx]:
                        #     if not self.pickup_path_pedestrian:
                        #         self.pickup_path_pedestrian = self.solve_pickup_path_pedestrian(agent_idx)

                        if counter % 8 == 0:
                            if self.is_on_the_zebra_crossing_pedestrian(
                                    coord=self.env.pedestrians[agent_idx].curr_head):
                                self.speed_pedestrian[agent_idx] = self.normal_update_interval
                            elif self.env.pedestrians[agent_idx].destination:
                                self.speed_pedestrian[agent_idx] = self.slow_update_interval
                            else:
                                self.speed_pedestrian[agent_idx] = self.base_update_interval

                        if counter % self.speed_pedestrian[agent_idx] == 0:
                            if self.is_on_the_pickup_area(agent_idx) and self.env.pedestrians[
                                agent_idx].destination and not self.pickup_path_pedestrian:  # waiting for pickup
                                # if self.env.pedestrians[agent_idx].destination and self.pickup_pedestrian[agent_idx] and not self.pickup_path_pedestrian:  # waiting for pickup
                                self.env.pedestrians[agent_idx].lock = True
                                self.env.pedestrians[agent_idx].lock_pos = pygame.Rect(
                                    self.CELL_SIZE * self.env.pedestrians[agent_idx].curr_head[0],
                                    self.CELL_SIZE * self.env.pedestrians[agent_idx].curr_head[1],
                                    self.CELL_SIZE,
                                    self.CELL_SIZE
                                )

                            if not self.keep_flag[
                                agent_idx]:  # if is waiting for green light, then keep the action unchanged
                                action = self.agent.act(self.timestep_result.observation, self.timestep_result.reward)
                                action = self.validate_traffic_rule_pedestrians(action, agent_idx)
                            else:
                                action = self.keep_action[agent_idx]
                                self.keep_flag[agent_idx] = False

                            wait4greenlight, redlight_force = self.interact_traffic_lights_pedestrians(action,
                                                                                                       agent_idx)

                            # if self.pickup_pedestrian[agent_idx]:
                            #     redlight_force = 0

                            if ((not wait4greenlight and redlight_force == 0) or (redlight_force == 1 and not self.event_lock)) and not self.env.pedestrians[agent_idx].lock:

                                if redlight_force == 1:
                                    self.event_lock = True
                                    print("E 3 Signal Violation (Pedestrian)")
                                    # self.logger.info("E 3 Signal Violation (Pedestrian)")
                                    self.reward_pedestrian = self.env.rewards["violation"]
                                    self.violation_scoreboard = -5
                                    self.event_type = 'pedestrian_violation'

                                    self.outlet.push_sample(["E 3"])
                                    self.receive_send_rumble()

                                    self.visual_feedback_p = agent_idx
                                # if self.pickup_path_pedestrian and self.pickup_pedestrian[agent_idx]:
                                #     action = self.pickup_path_pedestrian.pop(0)
                                #     print('Current pedestrians head', self.env.pedestrians[agent_idx].head)

                                self.env.choose_action_multiagents(action, self.env.pedestrians, agent_idx)
                                action_selected = False
                                self.timestep_result_p = self.env.timestep_pedestrians(agent_index=agent_idx,
                                                                                       rwd=self.reward_pedestrian)
                                self.reward_pedestrian = self.env.rewards["timestep"]
                            else:
                                self.keep_action[agent_idx] = action
                                self.keep_flag[agent_idx] = True

                    # elif self.taxi_service_done and self.env.pedestrians[agent_idx].lock and self.env.pedestrians[
                    #     agent_idx].destination:
                    #     new_head = PEDESTRIAN_GETOFF_POS[self.env.pedestrians[agent_idx].destination]
                    #     self.env.pedestrians[agent_idx].body = deque([Point(new_head[0], new_head[1])])
                    #     self.env.pedestrians[agent_idx].direction = BUILDING_POS_PARKING_DIRECTION[
                    #         self.env.pedestrians[agent_idx].destination]
                    #     self.env.field_pedestrian.update_snake_footprint(
                    #         Point(self.env.pedestrians[agent_idx].last_head[0],
                    #               self.env.pedestrians[agent_idx].last_head[1]), self.env.pedestrians[agent_idx].head)
                    #
                    #     (x, y) = self.env.pedestrians[agent_idx].head
                    #     self.env.pedestrians[agent_idx].last_head = [x, y]
                    #     self.env.pedestrians[agent_idx].curr_head = [x, y]
                    #
                    #     self.env.pedestrians[agent_idx].lock = False
                    #     self.env.pedestrians[agent_idx].destination = None
                    #     self.env.pedestrians[agent_idx].lock_pos = None
                    #     self.pickup_pedestrian[agent_idx] = False
                    #     self.pickup_path_pedestrian = None
                    #     self.speed_pedestrian[agent_idx] = self.base_update_interval
                    #     self.keep_action[agent_idx] = None
                    #     self.keep_flag[agent_idx] = False
                    #
                    # elif not self.trialends_countdown and not self.pickup_vehicle_found:
                    #     self.find_pickup_car(agent_idx)

                if self.save_frames and self.frame_num % 5 == 0:
                    threading.Thread(target=add_frame_to_buffer, args=(self.buffer, self.screen, 3)).start()

                    # # Save every 100 frames, for example
                    # if len(image_buffer) >= 100:
                    #     save_buffered_frames(image_buffer, output_dir='screenshots', mode=self.mode,
                    #                          start_frame_num=self.frame_num)
                    #     # Increment frame number to reflect saved frames
                    #     self.frame_num += len(image_buffer)

                if self.trialends_countdown: self.trialends_countdown_rounds -= 1
                if self.trialends_countdown_rounds < 0: self.timestep_result.is_episode_end = True
                if self.timestep_result.is_episode_end:
                    # time.sleep(3)
                    self.agent.end_episode()
                    self.running = False

            # Render.
            if self.running:
                score = self.env.stats.sum_episode_rewards
                self.render(counter=counter, head_update=True)

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

                x_pedestrian, y_pedestrian = [], []
                x0_pedestrian, y0_pedestrian = [], []
                # print("Current Number of Pedestrians: ", len(self.env.pedestrians))
                for agent_idx in range(len(self.env.pedestrians)):
                    x_pedestrian.append(self.env.pedestrians[agent_idx].curr_head[0])
                    y_pedestrian.append(self.env.pedestrians[agent_idx].curr_head[1])
                    x0_pedestrian.append(self.env.pedestrians[agent_idx].last_head[0])
                    y0_pedestrian.append(self.env.pedestrians[agent_idx].last_head[1])

                for interpolate_idx in range(self.intermediate_frames):
                    self.render(counter=counter)
                    self.render_traffic_lights(traffic_lights_position(self.CELL_SIZE),
                                               pygame.time.get_ticks() - self.pause_time)

                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
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

                    if not self.button_pressed:
                        self.render_scoreboard(score, pygame.time.get_ticks() - self.pause_time,
                                               self.accident_scoreboard + self.violation_scoreboard)
                    else:
                        self.render_scoreboard(score, pygame.time.get_ticks() - self.pause_time,
                                               self.button_pressed_value)
                    pygame.display.set_caption(
                        f'Robotaxi Surveillance [Score: {score:01d}]   |   [Steps Remaining: {time_remaining:01d}]')

                    if self.trialends_countdown:
                        smaller_text_font = pygame.font.Font("robotaxi_integration/fonts/gyparody_hv.ttf",
                                                             int(22 * (self.CELL_SIZE / 40.0)))
                        disp_text = smaller_text_font.render("Round Finished", True, (220, 220, 220))
                        self.screen.blit(disp_text, (self.CELL_SIZE * 23 + 50, self.CELL_SIZE * 20))

                    pygame.display.update()

                time_remaining = self.env.max_step_limit - self.env.timestep_index

                pygame.display.set_caption(
                    f'Robotaxi Surveillance [Score: {score:01d}]   |   [Steps Remaining: {time_remaining:01d}]')
                if not self.button_pressed:
                    self.render_scoreboard(score, pygame.time.get_ticks() - self.pause_time,
                                           self.accident_scoreboard + self.violation_scoreboard)
                else:
                    self.render_scoreboard(score, pygame.time.get_ticks() - self.pause_time,
                                           self.button_pressed_value)

                if self.trialends_countdown:
                    smaller_text_font = pygame.font.Font("robotaxi_integration/fonts/gyparody_hv.ttf",
                                                         int(22 * (self.CELL_SIZE / 40.0)))
                    disp_text = smaller_text_font.render("Round Finished", True, (220, 220, 220))
                    self.screen.blit(disp_text, (self.CELL_SIZE * 23 + 50, self.CELL_SIZE * 20))

                pygame.display.update()
            counter += 1
        self.outlet.push_sample(["T 2"])

        if self.save_frames:
            threading.Thread(target=save_buffered_frames, args=(self.buffer, 'screenshots', self.mode, self.frame_num // 5 - len(self.buffer), episode_num)).start()

    def compare_bounding_boxes(self, array1, array2):
        results = []
        # Compare all boxes in array1
        for i in range(len(array1)):
            for j in range(i + 1, len(array1)):
                if calculate_overlap_area(array1[i], array1[j]):
                    results.append((f"vehicle_{i}", f"vehicle_{j}"))
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
            elif self.env.vehicles[agent1_idx].direction == SnakeDirection.WEST:
                self.env.pedestrians[agent2_idx].lock_pos = self.env.vehicles[agent1_idx].lock_pos.move(-50, -10)
            elif self.env.vehicles[agent1_idx].direction == SnakeDirection.SOUTH:
                self.env.pedestrians[agent2_idx].lock_pos = self.env.vehicles[agent1_idx].lock_pos.move(10, 50)
            elif self.env.vehicles[agent1_idx].direction == SnakeDirection.EAST:
                self.env.pedestrians[agent2_idx].lock_pos = self.env.vehicles[agent1_idx].lock_pos.move(50, -10)


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
    SKY_BLUE = (135, 206, 235)
    BLUE = (0, 0, 255)
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