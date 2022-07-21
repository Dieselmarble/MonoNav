from __future__ import print_function
import rospy
import tf2_ros
import tf
import geometry_msgs.msg

class TFPublisher:
    def __init__(self):
        self.br = tf2_ros.TransformBroadcaster()
        self.X = 0.
        self.Y = 0.
        self.Z = 0.
        self.R = 0. # roll
        self.P = 0. # pitch
        self.Y = 0. # yaw
        self.frame_id = None
        self.child_frame_id = None

    def publish(self, **kwargs):
        try:
            if kwargs is not None:
                 translation = kwargs.get('t')
                 euler = kwargs.get('r')
                 frame_id = kwargs.get('frame_id')
                 child_frame_id = kwargs.get('child_frame_id')

            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = frame_id
            t.child_frame_id = child_frame_id
            t.transform.translation.x = translation[0]
            t.transform.translation.y = translation[1]
            t.transform.translation.z = translation[2]

            q = tf.transformations.quaternion_from_euler(euler[0], euler[1], euler[2])
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]

            self.br.sendTransform(t)

        except Exception as e:
            print(e)

if __name__ == '__main__':
    pass