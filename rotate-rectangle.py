import math
from graphics import *


class RotateRectangle:
    def __init__(self, width, height, rotation, centrePoint):
        self._width = width
        self._height = height
        self._rotation = rotation
        self._xCentre = centrePoint.getX()
        self._yCentre = centrePoint.getY()
        self._radius = (((width / 2) ** 2) + ((height / 2) ** 2)) ** 0.5
        self._phi = math.atan(width / height)

    def getPoints(self):
        angleList = []
        pointList = []

        angle1 = math.pi / 2 + self._phi - self._rotation
        angle2 = math.pi / 2 - self._phi - self._rotation
        angle3 = 3 * math.pi / 2 + self._phi - self._rotation
        angle4 = 3 * math.pi / 2 - self._phi - self._rotation
        angleList.append(angle1)
        angleList.append(angle2)
        angleList.append(angle3)
        angleList.append(angle4)

        for i in range(4):
            point = Point(self._radius * math.cos(angleList[i]) + self._xCentre,
                          self._radius * (-1) * math.sin(angleList[i]) + self._yCentre)
            pointList.append(point)

        return pointList

    def draw(self, win):
        self._pointList = self.getPoints()
        for i in range(4):
            line = Line(self._pointList[i], self._pointList[(i + 1) % 4])
            line.draw(win)