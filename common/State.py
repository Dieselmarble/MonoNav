import math
class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = 0.0
        self.vy = 0.0
        self.w = 0.0

    def getLinearSpeed(self):
        return math.sqrt(self.vx * self.vx + self.vy * self.vy)

    def __str__(self):
        str = ("X: %f \
               Y: %f \
               Yaw: %f" %(self.x, self.y, self.yaw))
        return str