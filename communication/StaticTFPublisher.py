import rospy
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import tf
import geometry_msgs.msg
from common import GlobalVar

GlobalVar._init()

class StaticTFPublisher():
    def __init__(self):
        rospy.init_node('static_tf2_broadcaster')
        self.broadcaster = StaticTransformBroadcaster()

    def publish(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            static_transformStamped = geometry_msgs.msg.TransformStamped()
            static_transformStamped.header.stamp = rospy.Time.now()
            static_transformStamped.header.frame_id = "/tracker"
            static_transformStamped.child_frame_id = "/odom"
            static_transformStamped.transform.translation.x = float(0.)
            static_transformStamped.transform.translation.y = float(0)
            static_transformStamped.transform.translation.z = float(0)
            quat = tf.transformations.quaternion_from_euler(0.,0.,0.)
            static_transformStamped.transform.rotation.x = quat[0]
            static_transformStamped.transform.rotation.y = quat[1]
            static_transformStamped.transform.rotation.z = quat[2]
            static_transformStamped.transform.rotation.w = quat[3]
            self.broadcaster.sendTransform(static_transformStamped)
            rate.sleep()
if __name__ == '__main__':
    pub = StaticTFPublisher()
    pub.publish()