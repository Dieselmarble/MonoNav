import rospy
import numpy as np
import tf2_ros
from common.State import State
from tf.transformations import euler_from_quaternion
from rospy_tutorials.msg import Floats

'''
Track the visual position of robot refer to word axis
'''
class VisualOdomReceiver():
    def __init__(self):
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.translation = np.asarray([0, 0, 0])
        self.rotation = np.asarray([0, 0, 0, 1])
        self.robotCor = np.zeros(2)
        rospy.Subscriber("robot_cor", Floats, self.callback)

    def callback(self, data):
        self.robotCor = (data.data[0], data.data[1])
        self.stateVisual = State()
        self.stateVisual.x = int(self.robotCor[0])
        self.stateVisual.y = int(self.robotCor[1])

    def listen(self):
        try:
            trans = self.tfBuffer.lookup_transform('world', 'tracker', rospy.Time())
            x = trans.transform.rotation.x
            y = trans.transform.rotation.y
            z = trans.transform.rotation.z
            w = trans.transform.rotation.w
            rotation = np.asarray([x, y, z, w])
            self.rotation = np.asarray(rotation)
            self.translation = np.asarray([trans.transform.translation.x,
                                          trans.transform.translation.y, trans.transform.translation.z])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            # print(e)
            pass

    def getTranslation(self):
        return self.translation

    def getRotation(self):
        return self.rotation

    def getVisualOdomState(self):
        return self.stateVisual

    def getVisualOdomStateByTF(self):
        stateVisual = State()
        self.listen()
        orientation = self.getRotation()
        position = self.getTranslation()
        stateVisual.x = position[0]
        stateVisual.y = position[1]
        stateVisual.z = position[2]
        stateVisual.yaw = euler_from_quaternion(orientation)[2]
        return stateVisual

def callback(data):
    robotCor = (data.data[0], data.data[1])

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("robot_cor", Floats, callback)
    rospy.spin()