import nav_msgs.msg
import rospy
from common.State import State
from tf.transformations import euler_from_quaternion

class WheelOdomReceiver():
    def __init__(self):
        self.receiveWheel = rospy.Subscriber("/odom", nav_msgs.msg.Odometry, self.getWheelOdomCallback)
        self.stateWheel = State()

    def getStateWheel(self):
        return self.stateWheel

    def combinationalFiltering(self):
        self.stateWheel = self.getWheelOdomCallback()

    def getWheelOdomCallback(self, odom_msg):
        #print("time", odom_msg.header.stamp)
        #print(odom_msg.pose.pose.position) #  the x,y,z pose and quaternion orientation
        stateWheel = State()
        stateWheel.x = odom_msg.pose.pose.position.x
        stateWheel.y = odom_msg.pose.pose.position.y
        quaternionWheel = (odom_msg.pose.pose.orientation.x,
                      odom_msg.pose.pose.orientation.y,
                      odom_msg.pose.pose.orientation.z,
                      odom_msg.pose.pose.orientation.w)
        stateWheel.yaw = euler_from_quaternion(quaternionWheel)[2]
        stateWheel.vx = odom_msg.twist.twist.linear.x
        stateWheel.vy = odom_msg.twist.twist.linear.y
        stateWheel.vz = odom_msg.twist.twist.linear.z
        stateWheel.w = odom_msg.twist.twist.angular.z
        self.stateWheel = stateWheel