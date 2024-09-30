import json


class SnakeAction(object):
    """ Defines all possible actions the agent can take in the environment. """

    MAINTAIN_DIRECTION = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


ALL_SNAKE_ACTIONS = [
    SnakeAction.MAINTAIN_DIRECTION,
    SnakeAction.TURN_LEFT,
    SnakeAction.TURN_RIGHT,
]


def get_next_step_choices_car():
    """ Returns a list of possible actions the agent can take in the current state. """
    level_filename = 'robotaxi_integration/robotaxi/levels/23x23-obstacles.json'
    with open(level_filename) as cfg:
        env_config = json.load(cfg)
    field = env_config['field']

    # point_map = {}
    # for i in range(len(field)):
    #     for j in range(len(field[i])):
    #         point_map[(i, j)] = []
    #         # if field[i][j] != '#':
    #         #     point_map[(i, j)].append('0')
    #         if i == 1:
    #             point_map[(i, j)].append((i + 1, j))
    #             if j not in [7, 8]:
    #                 point_map[(i, j)].append((i, j + 1))
    #                 point_map[(i, j)].append((i, j - 1))
    #             if j == 7:
    #                 point_map[(i, j)].append((i, j - 1))
    #             if j == 8:
    #                 point_map[(i, j)].append((i, j + 1))
    #         if i in [2, 3]:
    #             point_map[(i, j)].append((i - 1, j))
    #             point_map[(i, j)].append((i + 1, j))
    #             if i == 2:
    #                 if j != 8:
    #                     point_map[(i, j)].append((i, j + 1))
    #                     point_map[(i, j)].append((i, j - 1))
    #                 if j == 8:
    #                     point_map[(i, j)].append((i, j + 1))
    #         if i in range(4, 8):
    #             if j in [1, 2, 3, 4, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21]:
    #                 point_map[(i, j)].append((i, j + 1))
    #                 point_map[(i, j)].append((i, j - 1))

    point_map_north = {}
    point_map_south = {}
    point_map_east = {}
    point_map_west = {}
    north, south, east, west = True, True, True, True
    for i in range(len(field)):
        for j in range(len(field[i])):
            if north:
                point_map_north[(i, j)] = []
                if i in range(1, 4):
                    if j in [8, 9, 15, 16]:
                        point_map_north[(i, j)].append(0)
                    if i in [1, 2]:
                        if j in [8, 15]:
                            point_map_north[(i, j)].append(2)
                        if j in [9, 16]:
                            point_map_north[(i, j)].append(1)
                if i in range(4, 6):
                    if j in [8, 9, 16]:
                        point_map_north[(i, j)].append(0)
                    if j == 7:
                        point_map_north[(i, j)].append(1)
                        if i != 4:
                            point_map_north[(i, j)].append(0)
                    if j == 15:
                        point_map_north[(i, j)].append(1)
                        if i != 4:
                            point_map_north[(i, j)].append(0)
                if i in range(6, 9):
                    if j in [7, 8, 9, 15, 16]:
                        point_map_north[(i, j)].append(0)
                    if j == 9 and i != 6:
                        point_map_north[(i, j)].append(2)
                    if j == 15 and i == 6:
                        point_map_north[(i, j)].append(1)
                    if j == 16 and i != 6:
                        point_map_north[(i, j)].append(2)
                if i in range(9, 16):
                    if j in [7, 8, 9, 15, 16]:
                        point_map_north[(i, j)].append(0)
                if i in range(11, 15):
                    if j in [7, 8, 15]:
                        point_map_north[(i, j)].append(2)
                    if j in [8, 9, 16]:
                        point_map_north[(i, j)].append(1)
                if i in [16, 17]:
                    if j in [7, 8, 9, 15, 16]:
                        point_map_north[(i, j)].append(0)
                if i == 18:
                    if j in [15, 16]:
                        point_map_north[(i, j)].append(0)
                    if j == 16:
                        point_map_north[(i, j)].append(2)
                if i in range(19, 22):
                    if j == 16:
                        point_map_north[(i, j)].append(0)
            if south:
                point_map_south[(i, j)] = []
                if i in range(1, 4):
                    if j in [5, 6, 7]:
                        point_map_south[(i, j)].append(0)
                    if i == 1:
                        if j in [6, 7]:
                            point_map_south[(i, j)].append(2)
                        if j in [5, 6]:
                            point_map_south[(i, j)].append(1)
                if i in range(4, 8):
                    if j in [5, 6, 7]:
                        point_map_south[(i, j)].append(0)
                    if j == 5:
                        if i in [4, 5]:
                            point_map_south[(i, j)].append(2)
                    if j == 7 and i == 7:
                        point_map_south[(i, j)].append(1)
                    if j == 16 and i == 7:
                        point_map_south[(i, j)].append(1)
                if i == 8:
                    if j in [5, 6]:
                        point_map_south[(i, j)].append(0)
                    if j == 7:
                        point_map_south[(i, j)].append(1)
                if i in range(9, 16):
                    if j in [5, 6]:
                        point_map_south[(i, j)].append(0)
                if i in range(10, 15):
                    if j == 5:
                        point_map_south[(i, j)].append(1)
                    if j == 6:
                        point_map_south[(i, j)].append(2)
                if i == 16:
                    if j == 5:
                        point_map_south[(i, j)].append(2)
                    if j == 6:
                        point_map_south[(i, j)].append(0)
                        point_map_south[(i, j)].append(1)
                if i == 17:
                    if j == 6:
                        point_map_south[(i, j)].append(0)
                if i == 18:
                    if j == 6:
                        point_map_south[(i, j)].append(1)
                    # if j == 9:
                    #     point_map_south[(i, j)].append(1)
                if i in range(19, 22):
                    if j == 15:
                        point_map_south[(i, j)].append(0)
            if east:
                point_map_east[(i, j)] = []
                if j in range(1, 5):
                    if i in [6, 7, 8, 17, 18]:
                        point_map_east[(i, j)].append(0)
                    if j in [1, 2]:
                        if i in [6, 7, 17]:
                            point_map_east[(i, j)].append(2)
                        if i in [7, 8, 18]:
                            point_map_east[(i, j)].append(1)
                if j in range(5, 9):
                    if i in [6, 7, 8, 17, 18]:
                        point_map_east[(i, j)].append(0)
                    if i == 6 and j == 8:
                        point_map_east[(i, j)].append(1)
                    if i == 8:
                        if j in [5, 6]:
                            point_map_east[(i, j)].append(2)
                    if i == 17:
                        if j in [7, 8]:
                            point_map_east[(i, j)].append(1)
                if j == 9:
                    if i in [7, 8, 18]:
                        point_map_east[(i, j)].append(0)
                    if i == 6:
                        point_map_east[(i, j)].append(1)
                    if i == 17:
                        point_map_east[(i, j)].append(1)
                if j in range(10, 15):
                    if i in [7, 8, 18]:
                        point_map_east[(i, j)].append(0)

                if j == 15 and i == 7:
                    point_map_east[(i, j)].append(1)
                if j in [11, 12]:
                    if i == 7:
                        point_map_east[(i, j)].append(2)
                    if i == 8:
                        point_map_east[(i, j)].append(1)
                if j == 15 and i == 7:
                    point_map_east[(i, j)].append(1)
                if j in [15, 16]:
                    if i in [8, 18]:
                        point_map_east[(i, j)].append(0)
                    if i == 18:
                        point_map_east[(i, j)].append(1)
                        if j == 15:
                            point_map_east[(i, j)].append(2)
                if j in range(17, 22):
                    if i in [7, 8, 18]:
                        point_map_east[(i, j)].append(0)
                    if j > 17:
                        if i == 7:
                            point_map_east[(i, j)].append(2)
                        if i == 8:
                            point_map_east[(i, j)].append(1)
            if west:
                point_map_west[(i, j)] = []
                if j in range(1, 5):
                    if i in [4, 5, 16]:
                        point_map_west[(i, j)].append(0)
                    if j in [1, 2, 3]:
                        if i == 4:
                            point_map_west[(i, j)].append(1)
                        if i == 5:
                            point_map_west[(i, j)].append(2)

                ##############################  if i in [4, 5, 16]:
                if j in range(5, 10):
                    if i in [4, 5]:
                        point_map_west[(i, j)].append(0)

                    if i == 6 and j != 5:
                        point_map_west[(i, j)].append(0)

                if j in [8, 9] and i == 17:
                    point_map_west[(i, j)].append(0)

                if j in [5, 6] and i == 6:
                    point_map_west[(i, j)].append(1)

                ##############################  if j in [7, 8, 9] and i == 16:
                if j in [7, 8, 9] and i == 17:
                    point_map_west[(i, j)].append(2)

                if j in [8, 9] and i == 4:
                    point_map_west[(i, j)].append(2)

                ##############################  if j == 9 and i == 17:
                if j == 9 and i == 16:
                    point_map_west[(i, j)].append(1)


                if j in range(10, 15):
                    if i in [4, 5, 6, 16, 17]:
                        point_map_west[(i, j)].append(0)
                if j in [12, 13]:
                    if i in [4, 5, 16]:
                        point_map_west[(i, j)].append(1)
                    if i in [5, 6, 17]:
                        point_map_west[(i, j)].append(2)
                if j in [15, 16]:
                    if i in [4, 5, 16, 17]:
                        point_map_west[(i, j)].append(0)
                    if i in [4, 16]:
                        point_map_west[(i, j)].append(2)
                    if j == 16 and i == 6:
                        point_map_west[(i, j)].append(1)
                if j in range(17, 22):
                    if i in [4, 5, 6, 16, 17]:
                        point_map_west[(i, j)].append(0)
                if j in range(18, 22) and i == 6:
                    point_map_west[(i, j)].append(0)

                if j > 18:
                    if i in [4, 5, 16]:
                        point_map_west[(i, j)].append(1)
                    if i in [5, 6, 17]:
                        point_map_west[(i, j)].append(2)

    return point_map_north, point_map_south, point_map_east, point_map_west


def get_next_step_choices_pedestrian():

    level_filename = 'robotaxi_integration/robotaxi/levels/23x23-obstacles.json'
    with open(level_filename) as cfg:
        env_config = json.load(cfg)
    field = env_config['field_pedestrian']

    point_map_north = {}
    point_map_south = {}
    point_map_east = {}
    point_map_west = {}
    north, south, east, west = True, True, True, True

    for i in range(len(field)):
        for j in range(len(field[i])):
            if north:
                point_map_north[(i, j)] = []
                if j in [4, 10, 14, 17]:
                    if i in range(1, 19):
                        point_map_north[(i, j)].append(0)
                    if i in [3, 9, 15, 19]:
                        point_map_north[(i, j)].append(1)
                        point_map_north[(i, j)].append(2)
                if j in [14, 17] and i in [19, 20, 21]:
                    point_map_north[(i, j)].append(0)
                if j in [4, 10] and i == 19:
                    point_map_north[(i, j)].append(0)

            if south:
                point_map_south[(i, j)] = []
                if j in [4, 10, 14, 17]:
                    if i in range(1, 19):
                        point_map_south[(i, j)].append(0)
                    if i in [3, 9, 15, 19]:
                        point_map_south[(i, j)].append(1)
                        point_map_south[(i, j)].append(2)
                if j in [14, 17] and i in [18, 19, 20, 21]:
                    point_map_south[(i, j)].append(0)
                # if j in [4, 10] and i == 19:
                #     point_map_south[(i, j)].append(1)
                #     point_map_south[(i, j)].append(2)

            if east:
                point_map_east[(i, j)] = []
                if i in [3, 9, 15, 19]:
                    if j in range(1, 22):
                        point_map_east[(i, j)].append(0)
                    if j in [4, 10, 14, 17]:
                        point_map_east[(i, j)].append(1)
                if i in [3, 9, 15] and j in [4, 10, 14, 17]:
                        point_map_east[(i, j)].append(2)
                if i == 19 and j in [14, 17]:
                    point_map_east[(i, j)].append(2)

            if west:
                point_map_west[(i, j)] = []
                if i in [3, 9, 15, 19]:
                    if j in range(1, 22):
                        point_map_west[(i, j)].append(0)
                    if j in [4, 10, 14, 17]:
                        point_map_west[(i, j)].append(2)
                if i in [3, 9, 15] and j in [4, 10, 14, 17]:
                    point_map_west[(i, j)].append(1)
                if i == 19 and j in [14, 17]:
                    point_map_west[(i, j)].append(1)

                    # if j in [4, 10]:
                    #     point_map_west[(i, j)].append(2)

    return point_map_north, point_map_south, point_map_east, point_map_west


def get_next_step_choices_car_smooth_turn():

    point_map_north, point_map_south, point_map_east, point_map_west = get_next_step_choices_car()
    point_map_north_smooth = {}
    point_map_south_smooth = {}
    point_map_east_smooth = {}
    point_map_west_smooth = {}

    for i in range(23):
        for j in range(23):
            point_map_north_smooth[(i, j)] = []
            point_map_south_smooth[(i, j)] = []
            point_map_east_smooth[(i, j)] = []
            point_map_west_smooth[(i, j)] = []

    for i in range(23):
        for j in range(23):
            if 0 in point_map_north[(i, j)]:
                try:
                    if 0 in point_map_north[(i - 1, j)]:
                        point_map_north_smooth[(i, j)].append(0)
                    # point_map_north_smooth[(i, j)].append(0)
                except:
                    pass
                if i == 1:
                    point_map_north_smooth[(i, j)].append(0)
            if 1 in point_map_north[(i, j)]:
                try:
                    point_map_north_smooth[(i + 1, j)].append(1)
                except:
                    pass
            if 2 in point_map_north[(i, j)]:
                try:
                    point_map_north_smooth[(i + 1, j)].append(2)
                except:
                    pass
            if 0 in point_map_south[(i, j)]:
                try:
                    if 0 in point_map_south[(i + 1, j)]:
                        point_map_south_smooth[(i, j)].append(0)
                except:
                    pass
                if i == 21:
                    point_map_south_smooth[(i, j)].append(0)
            if 1 in point_map_south[(i, j)]:
                try:
                    point_map_south_smooth[(i - 1, j)].append(1)
                except:
                    pass
            if 2 in point_map_south[(i, j)]:
                try:
                    point_map_south_smooth[(i - 1, j)].append(2)
                except:
                    pass
            if 0 in point_map_east[(i, j)]:
                try:
                    if 0 in point_map_east[(i, j + 1)]:
                        point_map_east_smooth[(i, j)].append(0)
                except:
                    pass
                if j == 21:
                    point_map_east_smooth[(i, j)].append(0)
            if 1 in point_map_east[(i, j)]:
                try:
                    point_map_east_smooth[(i, j - 1)].append(1)
                except:
                    pass
            if 2 in point_map_east[(i, j)]:
                try:
                    point_map_east_smooth[(i, j - 1)].append(2)
                except:
                    pass
            if 0 in point_map_west[(i, j)]:
                try:
                    if 0 in point_map_west[(i, j - 1)]:
                        point_map_west_smooth[(i, j)].append(0)
                except:
                    pass
                if j == 1:
                    point_map_west_smooth[(i, j)].append(0)
            if 1 in point_map_west[(i, j)]:
                try:
                    point_map_west_smooth[(i, j + 1)].append(1)
                except:
                    pass
            if 2 in point_map_west[(i, j)]:
                try:
                    point_map_west_smooth[(i, j + 1)].append(2)
                except:
                    pass
    point_map_south_smooth[(7, 16)].append(1)
    point_map_south_smooth[(17, 9)].append(1)
    point_map_east_smooth[(16, 7)].append(1)
    point_map_north_smooth[(9, 9)].append(2)
    point_map_west_smooth[(6, 7)].append(1)
    # remove 0 from point_map_south_smooth[(17, 12)]
    # point_map_west_smooth[(17, 12)].append(2)
    # point_map_west_smooth[(17, 13)].remove(0)
    # point_map_west_smooth[(17, 12)].remove(0)
    # point_map_west_smooth[(16, 13)].remove(1)

    return point_map_north_smooth, point_map_south_smooth, point_map_east_smooth, point_map_west_smooth


# def get_next_step_choices_pedestrian():
#
#     level_filename = 'robotaxi_integration/levels/23x23-obstacles.json'
#     with open(level_filename) as cfg:
#         env_config = json.load(cfg)
#     print(env_config)
#     field = env_config['field_pedestrian']
#
#     point_map_north = {}
#     point_map_south = {}
#     point_map_east = {}
#     point_map_west = {}
#     north, south, east, west = True, True, True, True
#
#     for i in range(len(field)):
#         for j in range(len(field[i])):
#             if north:
#                 point_map_north[(i, j)] = []
#                 if j in [4, 10, 14, 17]:
#                     if i in range(1, 19):
#                         point_map_north[(i, j)].append(0)
#                     if i in [3, 9, 15, 19]:
#                         point_map_north[(i, j)].append(1)
#                         point_map_north[(i, j)].append(2)
#                 if j in [14, 17] and i in [19, 20, 21]:
#                     point_map_north[(i, j)].append(0)
#                 if j in [4, 10] and i == 19:
#                     point_map_north[(i, j)].append(0)
#
#             if south:
#                 point_map_south[(i, j)] = []
#                 if j in [4, 10, 14, 17]:
#                     if i in range(1, 19):
#                         point_map_south[(i, j)].append(0)
#                     if i in [3, 9, 15, 19]:
#                         point_map_south[(i, j)].append(1)
#                         point_map_south[(i, j)].append(2)
#                 if j in [14, 17] and i in [18, 19, 20, 21]:
#                     point_map_south[(i, j)].append(0)
#                 # if j in [4, 10] and i == 19:
#                 #     point_map_south[(i, j)].append(1)
#                 #     point_map_south[(i, j)].append(2)
#
#             if east:
#                 point_map_east[(i, j)] = []
#                 if i in [3, 9, 15, 19]:
#                     if j in range(1, 22):
#                         point_map_east[(i, j)].append(0)
#                     if j in [14, 17]:
#                         point_map_east[(i, j)].append(1)
#                         point_map_east[(i, j)].append(2)
#                     if j in [4, 10]:
#                         point_map_east[(i, j)].append(1)
#             if west:
#                 point_map_west[(i, j)] = []
#                 if i in [3, 9, 15, 19]:
#                     if j in range(1, 22):
#                         point_map_west[(i, j)].append(0)
#                     if j in [14, 17]:
#                         point_map_west[(i, j)].append(1)
#                         point_map_west[(i, j)].append(2)
#                     if j in [4, 10]:
#                         point_map_west[(i, j)].append(2)
#
#     return point_map_north, point_map_south, point_map_east, point_map_west


if __name__ == '__main__':

    get_next_step_choices_car_smooth_turn()
    pass