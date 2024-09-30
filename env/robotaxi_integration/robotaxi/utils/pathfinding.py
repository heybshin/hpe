import math

from robotaxi_integration.robotaxi.utils.next_step_choice import get_next_step_choices_car_smooth_turn, get_next_step_choices_pedestrian
from robotaxi_integration.robotaxi.gameplay.entities import SnakeDirection
import queue
from collections import namedtuple

LANE_AREA = [((4, 1), (5, 4)),
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


class Point(namedtuple('PointTuple', ['x', 'y'])):
    """ Represents a 2D point with named axes. """

    def __add__(self, other):
        """ Add two points coordinate-wise. """
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """ Subtract two points coordinate-wise. """
        return Point(self.x - other.x, self.y - other.y)


class NewSnakeDirection(object):
    """ Defines all possible directions the snake can take, as well as the corresponding offsets. """

    NORTH = Point(-1, 0)
    EAST = Point(0, 1)
    SOUTH = Point(1, 0)
    WEST = Point(0, -1)


ALL_SNAKE_DIRECTIONS = [
    NewSnakeDirection.NORTH,
    NewSnakeDirection.EAST,
    NewSnakeDirection.SOUTH,
    NewSnakeDirection.WEST,
]


def is_in_lane(point):
    for lane in LANE_AREA:
        if (lane[0][0] <= point[0] <= lane[1][0] and
                lane[0][1] <= point[1] <= lane[1][1]):
            return True
    return False


def pf_dijkstra_v(map, start=None, end=None):
    if not start or not end:
        raise ValueError("Start or destination point is not given.")

    frontier = queue.PriorityQueue()
    frontier.put(start, 0)
    came_from = dict()
    cost = dict()
    actions = dict()
    came_from[start] = None
    cost[start] = 0

    directions = ALL_SNAKE_DIRECTIONS

    while not frontier.empty():
        current = frontier.get()

        if current == end:
            break

        for action in map[current[1]][current[0]]:
            new_cost = cost[current] + 1 + math.ceil(action / 2) * 5

            direction_curr = current[1]
            direction_idx = directions.index(direction_curr)
            position_curr = Point(current[0][0], current[0][1])

            if action == 0:
                direction_next = direction_curr
                position_next = position_curr + direction_next
            if action == 1:
                direction_next = directions[direction_idx - 1]
                position_next = position_curr + direction_curr + direction_next
            if action == 2:
                direction_next = directions[(direction_idx + 1) % len(directions)]
                position_next = position_curr + direction_curr + direction_next
            if is_in_lane(position_curr): direction_next = direction_curr

            next = ((position_next.x, position_next.y), direction_next)
            if next not in cost or new_cost < cost[next]:
                cost[next] = new_cost
                frontier.put(next, new_cost)
                came_from[next] = current
                actions[next] = action

            # print(action)

    if end not in came_from:
        return None
    else:
        solution = [end]
        act_list = []
        while solution[-1] != start:
            act_list.append(actions[solution[-1]])
            solution.append(came_from[solution[-1]])
        solution.reverse()
        act_list.reverse()
        return solution, act_list


def pf_dijkstra_p(map, start=None, end=None):
    if not start or not end:
        raise ValueError("Start or destination point is not given.")

    frontier = queue.PriorityQueue()
    frontier.put(start, 0)
    came_from = dict()
    cost = dict()
    actions = dict()
    came_from[start] = None
    cost[start] = 0

    directions = ALL_SNAKE_DIRECTIONS

    while not frontier.empty():
        current = frontier.get()

        if current == end:
            break

        for action in map[current[1]][current[0]]:
            new_cost = cost[current] + 1

            direction_curr = current[1]
            direction_idx = directions.index(direction_curr)
            position_curr = Point(current[0][0], current[0][1])

            if action == 0:
                direction_next = direction_curr
            if action == 1:
                direction_next = directions[direction_idx - 1]
            if action == 2:
                direction_next = directions[(direction_idx + 1) % len(directions)]

            position_next = position_curr + direction_next

            next = ((position_next.x, position_next.y), direction_next)
            if next not in cost or new_cost < cost[next]:
                cost[next] = new_cost
                frontier.put(next, new_cost)
                came_from[next] = current
                actions[next] = action

            # print(action)

    if end not in came_from:
        return None
    else:
        solution = [end]
        act_list = []
        while solution[-1] != start:
            act_list.append(actions[solution[-1]])
            solution.append(came_from[solution[-1]])
        # act_list.append(actions[solution[-1]])
        # solution.append(came_from[solution[-1]])

        solution.reverse()
        act_list.reverse()
        return solution, act_list


def pathfinding(start, end, agent=None):

    start = direction_remapping(start)
    end = direction_remapping(end)

    if agent == 'vehicle':
        map_n, map_s, map_e, map_w = get_next_step_choices_car_smooth_turn()
    elif agent == 'pedestrian':
        map_n, map_s, map_e, map_w = get_next_step_choices_pedestrian()
    else:
        raise ValueError("Agent type is not given.")

    map = {NewSnakeDirection.NORTH: map_n,
           NewSnakeDirection.EAST: map_e,
           NewSnakeDirection.SOUTH: map_s,
           NewSnakeDirection.WEST: map_w,
           }
    if agent == 'vehicle':
        return pf_dijkstra_v(map, start, end)
    elif agent == 'pedestrian':
        return pf_dijkstra_p(map, start, end)


def direction_remapping(data):

    point = data[0]
    direction = data[1]

    if direction == SnakeDirection.NORTH:
        return (point, NewSnakeDirection.NORTH)
    if direction == SnakeDirection.EAST:
        return (point, NewSnakeDirection.EAST)
    if direction == SnakeDirection.SOUTH:
        return (point, NewSnakeDirection.SOUTH)
    if direction == SnakeDirection.WEST:
        return (point, NewSnakeDirection.WEST)


if __name__ == "__main__":

    start_pos = ((8, 1), SnakeDirection.EAST)
    # des_pos = ((12, 15), SnakeDirection.NORTH)
    # des_pos = ((16, 11), SnakeDirection.WEST)
    des_pos = ((7, 13), SnakeDirection.EAST)
    # des_pos = ((6, 11), SnakeDirection.WEST)

    # start_pos = ((17, 7), SnakeDirection.EAST)
    # des_pos = ((14, 5), SnakeDirection.SOUTH)
    # des_pos = ((20, 15), SnakeDirection.SOUTH)
    # des_pos = ((2, 15), SnakeDirection.NORTH)
    # des_pos = ((2, 16), SnakeDirection.NORTH)

    solution = pathfinding(start_pos, des_pos, agent='vehicle')

    pass
