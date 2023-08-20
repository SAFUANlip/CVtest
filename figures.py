import cv2
import math
import numpy as np
from abc import ABC, abstractmethod

class Figure(ABC):
    @abstractmethod
    def draw(self):
        pass


class Circle(Figure):
    def __init__(self, radius, center):
        self.radius = radius
        self.center = center

    def draw(self, img, color):
        draw = cv2.circle(img, self.center, self.radius, color, -1)
        return draw


class Polygon(Figure):
    def __init__(self):
        self.pts

    def draw(self, img, color):
        draw = cv2.fillPoly(img, [self.pts], color)
        return draw

    def rotate(self, angl):
        new_pts = np.array([self._rot(x, y, angl, self.xc, self.yc) for x, y in self.pts])
        self.pts = new_pts

    def _rot(self, x, y, angl, xc, yc):
        x -= xc
        y -= yc
        _x = x * math.cos(angl) - y * math.sin(angl)
        _y = x * math.sin(angl) + y * math.cos(angl)
        return [int(_x) + xc, int(_y) + yc]


class Rhombus(Polygon):
    def __init__(self, x, y, theta=45, length=10):
        self.theta = theta
        self.length = length
        a = length * math.cos(math.radians(theta))
        b = length * math.sin(math.radians(theta))

        self.xc = int(x + length * math.cos(math.radians(theta/2)) * math.cos(math.radians(theta/2)))
        self.yc = int(y + length * math.cos(math.radians(theta/2)) * math.sin(math.radians(theta/2)))

        self.pts = np.array([
            [int(x), int(y)],
            [int(x + length), int(y)],
            [int(x + length + a), int(y + b)],
            [int(x + a), int(y + b)]])


class Triangle(Polygon):
    def __init__(self, pts):
        self.pts = np.array(pts)
        self.xc = int((pts[0][0] + pts[1][0] + pts[2][0]) / 3)
        self.yc = int((pts[0][1] + pts[1][1] + pts[2][1]) / 3)


class Hexagon(Polygon):
    def __init__(self, radius, center):
        self.xc, self.yc = center
        self.radius = radius
        self.pts = np.array([[self.xc, self.yc-self.radius],
                      [self.xc+int(radius*math.cos(math.pi/6)), self.yc-radius//2],
                      [self.xc+int(radius*math.cos(math.pi/6)), self.yc+radius//2],
                      [self.xc, self.yc+radius],
                      [self.xc-int(radius*math.cos(math.pi/6)), self.yc+radius//2],
                      [self.xc-int(radius*math.cos(math.pi/6)), self.yc-radius//2]])




