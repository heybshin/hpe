import os
from PIL import ImageGrab



def create_log_folder(base_dir='logs'):
    # Ensure the base directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # List all directories in the base directory
    existing_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # Filter out folders that match the pattern "expX"
    exp_folders = [f for f in existing_folders if f.startswith('exp') and f[3:].isdigit()]

    # Determine the highest folder number
    if exp_folders:
        highest_num = max(int(f[3:]) for f in exp_folders)
    else:
        highest_num = 0

    # Create the new folder with the next number in sequence
    new_folder_name = f"exp{highest_num + 1}"
    new_folder_path = os.path.join(base_dir, new_folder_name)
    os.makedirs(new_folder_path)

    print(f"Created new log folder: {new_folder_path}")
    return new_folder_path


def get_screen(args, id):
    # Capture the screen
    screen = ImageGrab.grab()
    # save the image naming using index
    # resize the image to 720p
    screen = screen.resize((1280, 720))
    os.makedirs(os.path.join(args.baselog_dir, 'images'), exist_ok=True)
    screen.save('{}/images/{}.png'.format(args.baselog_dir, id))

    return screen


def screen_capture(counter):
    # Capture the screen
    screen = ImageGrab.grab()
    # save the image naming using index
    # resize the image to 720p
    screen = screen.resize((1280, 720))
    screen.save(f'./images/{counter}.png')
