import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from cv2 import aruco
import rospy
from common import GlobalVar
from common.Transformation import Transformation
from communication.TFPublisher import TFPublisher
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats

class Localization:
    def __init__(self, *args):
        GlobalVar._init()
        rospy.init_node('TF_Publisher', disable_signals=True)
        self.pub = rospy.Publisher('robot_cor', numpy_msg(Floats), queue_size=10)
        self.rate = rospy.Rate(10)
        # camera parameters
        self.dist = GlobalVar.getValue('dist')
        self.mtx = GlobalVar.getValue('mtx')
        # other parameters
        self.SHOW_ANIMATION = GlobalVar.getValue('showAnimation')
        self.GENERATE_MARKER = False
        # font for displaying text (below)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.W = GlobalVar.getValue('Width')
        self.H = GlobalVar.getValue('Height')

    def generateQR(self):
        if self.GENERATE_MARKER:
            background = np.ones((400,400))*255
            # id 0 for tracker, id 1 for world
            marker = aruco.drawMarker(aruco.Dictionary_get(aruco.DICT_6X6_1000), 1, 100)
            marker = cv2.resize(marker,(360,360))
            background[20:380,20:380]=marker
            cv2.imwrite('../camera/marker.png', background)

    def arucoAnimation(self, frame, rvec, tvec, ids, corners):
        # 在画面上 标注auruco标签的各轴
        for i in range(rvec.shape[0]):
            aruco.drawAxis(frame, self.mtx, self.dist, rvec[i, :, :], tvec[i, :, :], 0.03)
            aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.putText(frame, "Id: " + str(ids), (0, 64), self.font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
        ###### 距离估计 #####
        distance = tvec[0][0][2]  # 单位是米
        # 显示距离
        cv2.putText(frame, 'distance:' + str(round(distance, 4)) + str('m'), (0, 110), self.font, 0.75, (0, 255, 0), 2,
                    cv2.LINE_AA)
        ###### 角度估计 #####
        # 考虑Z轴（蓝色）的角度
        # 本来正确的计算方式如下，但是由于蜜汁相机标定的问题，实测偏航角度能最大达到104°所以现在×90/104这个系数作为最终角度
        deg = rvec[0][0][2] / math.pi * 180
        # deg=rvec[0][0][2]/math.pi*180*90/104
        # 旋转矩阵到欧拉角
        R = np.zeros((3, 3), dtype=np.float64)
        cv2.Rodrigues(rvec[0], R)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:  # 偏航，俯仰，滚动
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        # 偏航，俯仰，滚动换成角度
        rx = x * 180.0 / math.pi
        ry = y * 180.0 / math.pi
        rz = z * 180.0 / math.pi
        cv2.putText(frame, 'deg_z:' + str(ry)[:4] + str('deg'), (0, 140), self.font, 0.75, (0, 255, 0), 2,
                    cv2.LINE_AA)
        # print("偏航，俯仰，滚动",rx,ry,rz)

    def trackQRCode(self):
        publisher = TFPublisher()
        video = cv2.VideoCapture(1)
        if not video.isOpened():
            raise Exception("Could not open video device")
        video.set(cv2.CAP_PROP_FRAME_WIDTH, self.W)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)

        while not rospy.is_shutdown():
            _, frame = video.read()
            corners, ids, _ = aruco.detectMarkers(frame, aruco.Dictionary_get(aruco.DICT_6X6_1000))
            # record route
            if ids is not None:
                # mass center of the ROI box
                # 获取aruco返回的rvec旋转矩阵、tvec位移矩阵
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.09, self.mtx, self.dist)
                (rvec - tvec).any()  # get rid of that nasty numpy value array error
                self.arucoAnimation(frame, rvec, tvec, ids, corners)
                # enumerate captured ids
                for id_ in range(len(ids)):
                    # marker to camera
                    rotationVector = rvec[id_][0]
                    translationVector = tvec[id_][0]
                    # TF publisher
                    rotationMatrix, _ = cv2.Rodrigues(rotationVector)
                    #translationVector = np.dot(-rotationMatrix.T, translationVector)
                    eulers = Transformation.rotationMatrixToEulerAngles(rotationMatrix)
                    # determine the correct frame
                    id = ids[id_][0]
                    childFrameId = ''
                    if id == 0:
                        childFrameId = '/tracker'
                        massCenter = np.sum(corners[id_][0], axis=0) / 4
                        cx = int(massCenter[0])
                        cy = int(massCenter[1])
                        point = np.array([cx, cy], dtype=np.float32)
                        self.pub.publish(point)
                        cv2.circle(frame, (cx, cy) , radius=60, color=(0,255,255), thickness=4)
                    elif id == 1:
                        childFrameId = '/world'
                    publisher.publish(t=translationVector, r=eulers, frame_id='/camera', child_frame_id=childFrameId)
            else:
                # DRAW "NO IDS"
                cv2.putText(frame, "No Ids", (0, 64), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.namedWindow('Tracking')
            cv2.resizeWindow('Tracking', self.W, self.H)
            cv2.imshow("Tracking", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord(' '):  # 按空格键保存
                filename = str(time.time())[:10] + ".png"
                cv2.imwrite(filename, frame)
        self.rate.sleep()
        cv2.destroyAllWindows()
        return

if __name__ == '__main__':
    tracker = Localization()
    tracker.trackQRCode()

