#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 PM5:34
# @Author  : L
# @Email   : L862608263@163.com
# @File    : demo.py
# @Software: PyCharm

# 多维数组处理库
import numpy

# 图片库
from PIL import Image as Pil_image

# 机器学习的库
from sklearn import svm

import os


class ImageRecognition:

    def open_image(self, image_add):
        """打开图片  并且灰度化convert('L')"""
        image = Pil_image.open(image_add).convert('L')
        return image

    def image_binarization(self, image):
        """图片二值化"""
        image_np = numpy.array(image)
        rows, cols = image_np.shape
        for i in range(rows):
            for j in range(cols):
                if image_np[i, j] <= 128:
                    image_np[i, j] = 0
                else:
                    image_np[i, j] = 1
        return image_np

    def image_noise_deal(self, image_np):
        """噪点处理"""
        rows, cols = image_np.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                num = 0
                if image_np[i - 1, j]:
                    num += 1
                if image_np[i + 1, j]:
                    num += 1
                if image_np[i, j - 1]:
                    num += 1
                if image_np[i, j + 1]:
                    num += 1
                if num >= 3:
                    image_np[i, j] = 1
        return image_np

    def split_image(self, start_num, end_num, image_np):
        """分割验证码的方法"""
        image_np_var_deal = self.image_np_var[start_num: end_num]
        var_deal, every_length, image_np_deal_list = [], [], []
        image_np_deal = image_np[:, start_num:end_num]
        number = 0
        for i in range(len(image_np_var_deal)):
            if image_np_var_deal[i] == 0:
                if image_np_var_deal[i - 1] != 0:
                    var_deal.append(image_np_var_deal[number + 1:i])
                    image_np_deal_list.append(image_np_deal[:, number + 1:i])
                    every_length.append(i - number - 1)
                    number = i
                else:
                    number = i
        return image_np_deal_list, every_length

    def fusion_image(self, g):
        """合成图片"""
        new_g = [self._split_checkcode(i) for i in g]
        return numpy.vstack(new_g)

    def verify_length(self, image_np_deal_list, every_length):
        g = image_np_deal_list
        d = every_length
        if len(d) < 4:
            leave_out = 4 - len(d)
            if leave_out == 1:
                max_index = d.index(max(d))
                one, two = numpy.array_split(g[max_index], 2, axis=1)
                g.pop(max_index)
                g.insert(max_index, two)
                g.insert(max_index, one)
            if leave_out == 3:
                result = numpy.array_split(g[0], 4, axis=1)
                g.pop()
                g.extend(result)
            if leave_out == 2:
                if 0.4 < d[0] / d[1] < 2.5:
                    one, two = numpy.array_split(g[0], 2, axis=1)
                    three, four = numpy.array_split(g[1], 2, axis=1)
                    g = [one, two, three, four]
                else:
                    max_index = d.index(max(d))
                    one, two, three = numpy.array_split(g[max_index], 3, axis=1)
                    g.pop(max_index)
                    g.insert(max_index, three)
                    g.insert(max_index, two)
                    g.insert(max_index, one)
        elif len(d) > 4:
            for i, j in enumerate(d):
                if j == 1:
                    g.pop(i)
        elif len(d) == 4:
            for i, j in enumerate(d):
                if j == 1:
                    g.pop(i)
                    d.pop(i)
                    return self.verify_length(g, d)
        return g

    def _split_checkcode(self, one):
        for i in range(18 - one.shape[1]):
            if i % 2:
                one = numpy.hstack((numpy.array([1] * 27)[:, numpy.newaxis], one))
            else:
                one = numpy.hstack((one, numpy.array([1] * 27)[:, numpy.newaxis]))
        one = one.ravel()[numpy.newaxis, :]
        return one

    def get_start_index_deal(self, image_np):
        self.image_np_var = image_np.var(axis=0)
        for i in range(1, len(self.image_np_var) - 1):
            if self.image_np_var[i] != self.image_np_var[i - 1]:
                return i - 1

    def get_end_index_deal(self, _):
        for i in range(len(self.image_np_var) - 1, 0, -1):
            if self.image_np_var[i] != self.image_np_var[i - 1]:
                return i + 1


if __name__ == "__main__":

    os.chdir(r'/Users/l/Desktop/ImageRecognition/Resource/check_code')

    # 给个处理好的图片的矩阵存在x
    x = []
    for i in range(180):
        image_recognize = ImageRecognition()

        image_name = str(i) + '.gif'

        image = image_recognize.open_image(image_name)

        image_np = image_recognize.image_binarization(image)

        image_np = image_recognize.image_noise_deal(image_np)

        start_num = image_recognize.get_start_index_deal(image_np)

        end_num = image_recognize.get_end_index_deal(image_np)

        image_np_deal_list, every_length = image_recognize.split_image(start_num, end_num, image_np)

        g = image_recognize.verify_length(image_np_deal_list, every_length)

        x.append(image_recognize.fusion_image(g))

    x = numpy.vstack([x[i] for i in range(180)])

    # 打开验证码的答案

    class_ = list('0123456789abcdefghijklmnopqrstuvwxyz')

    y_class = {i: class_[i] for i in range(36)}

    y_class2 = {class_[i]: i for i in range(36)}

    str1 = ''

    with open('check_code.txt', 'r') as f:
        for i in f.readlines():
            str1 += i

    y = str1.replace('\n', '')

    new_yy = [y_class2[i] for i in y]

    # 通过上面的图确定的系数

    def model_training(a, new_yy, i):
        svc = svm.SVC(gamma=0.001, C=100)
        svc.fit(a[:i * 4], numpy.array(new_yy[:i * 4]))
        print('识别字符准确率 ', svc.score(a[i * 4:], numpy.array(new_yy[i * 4:])))
        return svc.predict(a[i * 4:]), numpy.array(new_yy[i * 4:])

    y_predict, y_true = model_training(x, new_yy, 140)
    y_predict_class = [y_class[i] for i in y_predict]
    y_true_class = [y_class[i] for i in y_true]
    result_char = ''.join(y_predict_class)
    true_char = ''.join(y_true_class)

    count = 0
    for i in range(0, 160, 4):
        print('识别结果:  %s 实际结果: %s' % (result_char[i:i + 4], true_char[i:i + 4]))
        if result_char[i:i + 4] == true_char[i:i + 4]:
            count += 1
    print('识别数量', count)
    print('识别准确率', count / 40)
