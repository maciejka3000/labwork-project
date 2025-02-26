

import cv2
import os
import glob

class ImageProcess:
    def __init__(self, path):
        self.path = path
        self.image = cv2.imread(self.path)




if __name__ == "__main__":
    path = '/home/maciejka/Desktop/school/S8/labwork-project/db/dataset/9.jpg'
    image_process = ImageProcess(path)
