import pygame
import cv2
from PIL import Image

if __name__ == '__main__':

    # for i in [10]:
    # for i in range(5, 90, 5):
    for i in range(-175, -90, 5):

        path = './icon/turn/pickup/{}.png'.format(i)

        # Load your original image
        original_image = Image.open(path)

        # Define the padding size (e.g., 50 pixels on each side)
        padding = 100

        # Create a new image with a transparent background and larger size
        new_width = original_image.width + 2 * padding
        new_height = original_image.height + 2 * padding

        # Create an image with transparent background (RGBA)
        new_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))

        # Paste the original image into the center of the new image
        new_image.paste(original_image, (padding, padding))

        # Save the padded image
        path = './icon/turn/pickup/{}.png'.format(i)

        new_image.save(path)
