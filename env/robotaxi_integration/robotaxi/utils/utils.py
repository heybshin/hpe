from PIL import Image
import pygame
import os


def get_screenshot(screen, scale_factor=2):

    frame_image = pygame.image.tostring(screen, 'RGB')
    width, height = screen.get_size()
    pil_image = Image.frombytes('RGB', (width, height), frame_image)

    new_width = width // scale_factor
    new_height = height // scale_factor
    return pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def add_frame_to_buffer(buffer, screen, scale_factor=2):

    frame_image = pygame.image.tostring(screen, 'RGB')
    width, height = screen.get_size()
    pil_image = Image.frombytes('RGB', (width, height), frame_image)

    new_width = width // scale_factor
    new_height = height // scale_factor
    pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    buffer.append(pil_image)


def save_buffered_frames(image_buffer, output_dir='screenshots', mode=1, start_frame_num=0, trial=0):
    # os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    filepath = f'{output_dir}/mode_{mode}/trial_{trial}'
    os.makedirs(filepath, exist_ok=True)  # Ensure the directory exists
    for i, frame in enumerate(image_buffer):
        # Generate a filename for each frame
        frame_num = start_frame_num + i
        filename = f'{filepath}/totalframe_{frame_num:05d}_frame_{i:03d}.png'
        # Save the frame
        frame.save(filename)
        print(f'Saved {filename}')

    # Optionally clear the buffer after saving
    image_buffer.clear()