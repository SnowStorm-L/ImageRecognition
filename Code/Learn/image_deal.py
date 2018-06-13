#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 PM8:14
# @Author  : L
# @Email   : L862608263@163.com
# @File    : image_deal.py
# @Software: PyCharm

from PIL import Image as Pil_image
from PIL import ImageDraw as Pil_image_draw
import os


class ImageDeal:

    def open_image(self, image_path):
        image = Pil_image.open(image_path)
        return image

    def image_grayscale(self, image):
        return image.convert('L')

    def image_binarization(self, image):
        return image.convert('1')

    # 降噪
    # 根据一个点A的RGB值，与周围的8个点的RBG值比较，设定一个值N（0 <N <8），当A的RGB值与周围8个点的RGB相等数小于N时，此点为噪点
    # threshold 图像二值化阀值
    # N: Integer 降噪率 0 <N <8
    # deep 降噪次数

    def image_clear_noise(self, image, threshold, N, deep):
        draw = Pil_image_draw.Draw(image)
        for i in range(0, deep):
            for x in range(1, image.size[0] - 1):
                for y in range(1, image.size[1] - 1):
                    color = self.__get_pixel(image, x, y, threshold, N)
                    if color is None:
                        draw.point((x, y), color)
        return draw

    # 二值判断,如果确认是噪声,用改点的上面一个点的灰度进行替换
    # 该函数也可以改成RGB判断的,具体看需求如何

    def __get_pixel(self, image, x, y, threshold, N):
        L = image.getpixel((x, y))
        if L > threshold:
            L = True
        else:
            L = False
        near_dots = 0
        if L == (image.getpixel((x - 1, y - 1)) > threshold):
            near_dots += 1
        if L == (image.getpixel((x - 1, y)) > threshold):
            near_dots += 1
        if L == (image.getpixel((x - 1, y + 1)) > threshold):
            near_dots += 1
        if L == (image.getpixel((x, y - 1)) > threshold):
            near_dots += 1
        if L == (image.getpixel((x, y + 1)) > threshold):
            near_dots += 1
        if L == (image.getpixel((x + 1, y - 1)) > threshold):
            near_dots += 1
        if L == (image.getpixel((x + 1, y)) > threshold):
            near_dots += 1
        if L == (image.getpixel((x + 1, y + 1)) > threshold):
            near_dots += 1

        if near_dots < N:
            return image.getpixel((x, y - 1))
        else:
            return None


if __name__ == "__main__":
    os.chdir("/Users/l/Desktop/Test")

    image_deal = ImageDeal()
    origin_image = image_deal.open_image('test.png')
    # origin_image.show()

    binarization_image = image_deal.image_binarization(origin_image)
    binarization_image.show()

    grayscale_image = image_deal.image_grayscale(origin_image)
    grayscale_image.show()

    # image_deal.image_clear_noise(grayscale_image, 50, 4, 4)
    # grayscale_image.show()

