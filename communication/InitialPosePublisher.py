import rospy
import tf2_ros
import tf
import numpy as np
import geometry_msgs.msg
from common import GlobalVar

class InitialPosePublisher():
    def __init__(self):
        self.pub = rospy.Publisher('/initialpose', geometry_msgs.msg.PoseWithCovarianceStamped, queue_size=10)

    def publish(self, **kwargs):
        if kwargs is not None:
            position = kwargs.get('position')
            orientation = kwargs.get('orientation')
            frameId = kwargs.get("frameId")
        # get position and orientation
        p = geometry_msgs.msg.PoseWithCovarianceStamped()
        p.header.frame_id = frameId
        p.header.stamp = rospy.Time.now()
        p.pose.pose.position.x = position[0]
        p.pose.pose.position.y = position[1]
        p.pose.pose.position.z = position[2]
        p.pose.pose.orientation.x = orientation[0]
        p.pose.pose.orientation.y = orientation[1]
        p.pose.pose.orientation.z = orientation[2]
        p.pose.pose.orientation.w = orientation[3]
        # publish pose
        self.pub.publish(p)

    def getInitialPosition(self):
        duration = 0.0
        time_begin = rospy.Time.now()
        publisher = InitialPosePublisher()
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        while duration < 10.0:
            time_end = rospy.Time.now()
            duration = (time_end - time_begin).to_sec()
            try:
                trans = tfBuffer.lookup_transform('world', 'tracker', rospy.Time())
                x = trans.transform.rotation.x
                y = trans.transform.rotation.y
                z = trans.transform.rotation.z
                w = trans.transform.rotation.w
                rotation = np.asarray([x, y, z, w])
                self.rotation = np.asarray(rotation)
                self.translation = np.asarray([trans.transform.translation.x,
                                              trans.transform.translation.y, trans.transform.translation.z])

                publisher.publish(position=self.translation, orientation=self.rotation, frameId="world")
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                print(e)


if __name__ == '__main__':
    GlobalVar._init()
    rospy.init_node('TF_Publisher', disable_signals=True)
    publisher = InitialPosePublisher()
    publisher.getInitialPosition()