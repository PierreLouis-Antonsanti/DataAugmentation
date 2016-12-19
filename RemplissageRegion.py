#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://www.google.ca/url?sa=t&rct=j&q=&esrc=s&source=web&cd=4&cad=rja&uact=8&sqi=2&ved=0ahUKEwjC4u3m1_bQAhVM6WMKHRl2Ca8QFgg8MAM&url=http%3A%2F%2Fwww.f-legrand.fr%2Fscidoc%2Fdocmml%2Fimage%2Fextraction%2Fregions%2Fregions.html&usg=AFQjCNES6H_-aADzesTfvap9WGWsbsh7ig&sig2=0TbxaP9FxrujlKUMtD3zDw&bvm=bv.141536425,d.cGc

import numpy as np
import matplotlib.pyplot as plt

def fill(data, xsize, ysize, x_start, y_start):

    stack = [(x_start, y_start)]

    while stack:

        # plt.imshow(data)
        # plt.show()

        x, y, stack = stack[0][0], stack[0][1], stack[1:]

        if data[x, y] == 0:
            data[x, y] = 1
            if x > 0:
                if data[x-1, y] == 0:
                    stack.append((x - 1, y))
            if x < (xsize - 1):
                if data[x + 1, y] == 0:
                    stack.append((x + 1, y))
            if y > 0:
                if data[x, y - 1] == 0:
                    stack.append((x, y - 1))
            if y < (ysize - 1):
                if data[x, y + 1] == 0:
                    stack.append((x, y + 1))


    return data