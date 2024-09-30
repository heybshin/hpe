from dualsense import DualSense



if __name__ == "__main__":

    dualsense = DualSense()
    while True:
        # dualsense.general_rumble(0.4, (0.01, 0), 'both')
        #
        # dualsense.general_rumble(0.4, (0.1, 0), 'both')
        # dualsense.general_rumble(0.4, (1, 0), 'both')
        # dualsense.general_rumble(0.4, (0, 0.01), 'both')
        # dualsense.general_rumble(0.4, (0, 0.1), 'both')
        # dualsense.general_rumble(0.4, (0, 1), 'both')

        # dualsense.general_rumble(0.2, [1, 1], 'both')
        # dualsense.general_rumble(0.4, [1, 1], 'both')
        # dualsense.general_rumble(0.6, [1, 1], 'both')
        #



        # dualsense.general_rumble(0.4, [0.05, 0.05], 'both')
        # dualsense.general_rumble(0.4, [0.1, 0.1], 'both')
        # dualsense.general_rumble(0.4, [0.2, 0.2], 'both')
        # dualsense.general_rumble(0.4, [0.5, 0.5], 'both')
        # dualsense.general_rumble(0.4, [1, 1], 'both')

        # dualsense.general_rumble(0.4, [0.05, 0.05], 'left')
        dualsense.general_rumble(0.2, [0.1, 0.1], 'left')
        dualsense.general_rumble(0.2, [0.2, 0.2], 'left')
        dualsense.general_rumble(0.2, [0.5, 0.5], 'left')
        dualsense.general_rumble(0.2, [0.8, 0.8], 'left')
        dualsense.general_rumble(0.2, [1, 1], 'left')

        # dualsense.general_rumble(0.4, [0.05, 0.05], 'right')
        dualsense.general_rumble(0.2, [0.1, 0.1], 'right')
        dualsense.general_rumble(0.2, [0.2, 0.2], 'right')
        dualsense.general_rumble(0.2, [0.5, 0.5], 'right')
        dualsense.general_rumble(0.2, [0.8, 0.8], 'right')
        dualsense.general_rumble(0.2, [1, 1], 'right')

        # dualsense.general_rumble(0.4, [0.01, 0.01], 'left')
        # dualsense.general_rumble(0.4, [0.01, 0.01], 'right')
