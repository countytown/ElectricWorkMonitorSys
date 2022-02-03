import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import cv2

def draw(filename, result):
    img = Image.open(filename)
    w, h = img.size
    draw = ImageDraw.Draw(img)
    result = np.array(result)
    x = result[0][0]
    y = result[0][1]
    angle = result[0][2]
    height = result[0][3]
    width = result[0][4]
    # [1007.0, 399.0, 38.0, 108.0, 77.0]
    x = 500
    y = 200
    width = 108
    height = 38
    angle = -10
    anglePi = -(angle) * math.pi / 180.0
    # anglePi = 1.57
    cosA = math.cos(anglePi)
    sinA = math.sin(anglePi)

    x1 = x - 0.5 * width
    y1 = y - 0.5 * height

    x0 = x + 0.5 * width
    y0 = y1

    x2 = x1
    y2 = y + 0.5 * height

    x3 = x0
    y3 = y2

    x0n = (x0 - x) * cosA - (y0 - y) * sinA + x
    y0n = (x0 - x) * sinA + (y0 - y) * cosA + y

    x1n = (x1 - x) * cosA - (y1 - y) * sinA + x
    y1n = (x1 - x) * sinA + (y1 - y) * cosA + y

    x2n = (x2 - x) * cosA - (y2 - y) * sinA + x
    y2n = (x2 - x) * sinA + (y2 - y) * cosA + y

    x3n = (x3 - x) * cosA - (y3 - y) * sinA + x
    y3n = (x3 - x) * sinA + (y3 - y) * cosA + y

    draw.line([(x0n, y0n), (x1n, y1n)], fill=(0, 0, 255))
    draw.line([(x1n, y1n), (x2n, y2n)], fill=(255, 0, 0))
    draw.line([(x2n, y2n), (x3n, y3n)], fill=(0, 0, 255))
    draw.line([(x0n, y0n), (x3n, y3n)], fill=(255, 0, 0))

    plt.imshow(img)
    plt.show()
    four_points = [(x0n,y0n),(x1n,y1n),(x2n,y2n),(x3n,y3n)]
    print(four_points)

draw('aaa.png', [[100,100,90,50,50]])