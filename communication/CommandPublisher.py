from __future__ import print_function
import time
import rospy
from geometry_msgs.msg import Twist

class Broadcaster:
    def __init__(self):
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.speed = rospy.get_param("~speed", 1.0)
        self.turn = rospy.get_param("~turn", 1.0)
        self.x = 0
        self.y = 0
        self.z = 0
        self.th = 0

    def vels(self, speed, turn):
        return "currently:\tspeed %s\tturn %s " % (speed,turn)

    def setTurnFactor(self, turn):
        self.turn = turn

    def setSpeedFactor(self, speed):
        self.speed = speed

    def setLinearSpeedXYZ(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def setTh(self, th):
        self.th = th

    def getXYZ(self):
        return self.x, self.y, self.z

    def getTh(self):
        return self.th

    def getSpeed(self):
        return self.speed

    def getTurn(self):
        return self.turn

    def publish(self):
        try:
            #print(self.vels(self.speed, self.turn))
            x,y,z = self.getXYZ()
            th = self.getTh()
            twist = Twist()
            speed = self.getSpeed()
            turn = self.getTurn()
            twist.linear.x = x * speed
            twist.linear.y = y * speed
            twist.linear.z = z * speed
            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = th*turn
            self.pub.publish(twist)

        except Exception as e:
            print(e)
            twist = Twist()
            twist.linear.x = 0
            twist.linear.y = 0
            twist.linear.z = 0
            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = 0
            self.pub.publish(twist)

if __name__ == '__main__':
    broadcaster = Broadcaster()
    broadcaster.setSpeedFactor(0.25)
    broadcaster.setTurnFactor(0.0)
    while(1):
        broadcaster.setLinearSpeedXYZ(1, 1, 0)
        broadcaster.publish()
        time.sleep(0.01)