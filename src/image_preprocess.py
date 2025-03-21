from random import random

import cv2
import os
import glob
from typing import Literal, List
import math
import numpy as np
import random

###### TODO
# 1. split resizing with keeping proportions into two-step process:
    # - resize image, keep proportions
    # - (do something else) - for instance fft it, add noise
    # - add boundaries at end


class ImageProcess:
    def __init__(self, path):
        self.path = path
        self.image = cv2.imread(self.path)
        self.original_image = self.image.copy()

    def __resize_keep_proportions(self, new_shape: List[int], interpolation):

        x, y = new_shape[0], new_shape[1]
        x_old, y_old = self.image.shape[1], self.image.shape[0]
        old_aspect = x_old / y_old
        new_aspect = x / y

        if old_aspect < new_aspect:
            resize_factor = y / y_old
            new_x = math.floor(x_old * resize_factor)
            new_size = (new_x, y)
            borders = (x - new_x) // 2
            border_list = [0, 0, borders, borders + (x - new_x) % 2]
        else:

            resize_factor = x / x_old
            new_y = math.floor(y_old * resize_factor)
            new_size = (x, new_y)
            borders = (y - new_y) // 2
            border_list = [borders, borders + (y - new_y) % 2, 0, 0]


        self.image = cv2.resize(self.image, new_size, interpolation=interpolation)
        self.image = cv2.copyMakeBorder(self.image, *border_list, cv2.BORDER_CONSTANT)


    def __resize_stretch(self, size: List[int], interpolation):
        self.image = cv2.resize(self.image, size, interpolation=interpolation)


    def show_image(self, img = None):
        if img is None:
            img = self.image
        cv2.imshow('Press any key to destroy window', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def resize_image(self, size: List[int], resizing_option: Literal["keep_proportions", "stretch"], interpolation: Literal["nearest", "linear", "cubic", "lanczos"]):
        interp_opts = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        interpolation = interp_opts[interpolation]
        if resizing_option == "keep_proportions":
            self.__resize_keep_proportions(size, interpolation)
        elif resizing_option == "stretch":
            self.__resize_stretch(size, interpolation)


    def __fft(self, channel):
        channel = channel / channel.max()
        fft_channel = np.fft.fft2(channel)
        fft_shift = np.fft.fftshift(fft_channel)
        mag_spectrum = np.log1p(np.abs(fft_channel))
        return fft_shift, mag_spectrum

    def __cutoff(self,fft_shift, channel, f):
        y, x = channel.shape
        c_y, c_x = y//2, x//2

        mask = np.zeros((y, x), np.uint8)
        new_y, new_x = np.ogrid[:y, :x]
        mask_area = (new_x - c_x)**2 + (new_y - c_y)**2 <= f ** 2
        mask[mask_area] = 1
        filtered_fft = fft_shift * mask
        ifft_shift = np.fft.ifftshift(filtered_fft)
        channel_filtered = np.abs(np.fft.ifft2(ifft_shift))
        return channel_filtered


    def dist_lowpass(self, max_f:float):
        channels = []
        for channel in cv2.split(self.image):
            shift, spectrum = self.__fft(channel)
            lowpassed = self.__cutoff(shift, spectrum, max_f)
            channels.append(lowpassed)


        image = cv2.merge(channels)
        self.image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


    def dist_noise_gaussian(self, mean:float, std:float):
        noise = np.random.normal(mean, std, self.image.shape).astype(np.int16)
        noisy_img = self.image.astype(np.int16) + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        self.image = noisy_img
        pass

    def dist_blackholes(self, min_radius:float, max_radius:float, hole_amount:int, circle_color:tuple = (0, 0, 0)):
        img_size = self.image.shape[0:2]
        for hole in range(hole_amount):
            selected_radius = random.randint(int(min_radius), int(max_radius))
            selected_center_y = random.randint(0, img_size[0])
            selected_center_x = random.randint(0, img_size[1])
            self.image = cv2.circle(self.image, (selected_center_x, selected_center_y), selected_radius, circle_color, -1)


    def save_image(self, path):
        cv2.imwrite(path, self.image)



if __name__ == "__main__":
    path = '/home/maciejka/Desktop/school/S8/labwork-project/db/dataset/11.jpg'
    img = ImageProcess(path)
    img.resize_image([256, 256], 'keep_proportions', interpolation="cubic")
    img.dist_blackholes(10, 20, 3)
    img.save_image('output.jpg')
