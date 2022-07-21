from common import GlobalVar
import numpy as np
import math
import cv2

GlobalVar._init()
H = GlobalVar.getValue('Height')

class Transformation(object):
    def __init__(self):
        pass

    @staticmethod
    def visualOdom2Image(ax, ay):
        dts = GlobalVar.getValue('world') # world aruco的位置
        ori = GlobalVar.getValue('origin') # 世界坐标系origin在图片中的位置
        scale = GlobalVar.getValue('scale')
        return (ori[0]+int(ax*scale), abs(ori[1]+int(ay*scale)-H))


    @staticmethod
    def convertCoordinates(cx, cy):
        ccx = [x for x in cx]
        ccy = [abs(x - H) for x in cy]
        return ccx, ccy

    '''
        from resolution axis to camera axis
    '''
    @staticmethod
    def resolutionAxis2CameraAxis(ax, ay, mtx):
        pixels = np.stack((ax, ay, np.ones(len(ax))))
        coordinates = np.dot(np.linalg.inv(mtx), pixels)
        rx = coordinates[0,:]
        ry = coordinates[1,:]
        return rx, ry

    @staticmethod
    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    @staticmethod
    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(R):
        assert (Transformation.isRotationMatrix(R))
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    @staticmethod
    def inversePerspective(rvec, tvec):
        R, _ = cv2.Rodrigues(rvec)
        R = np.matrix(R).T
        invTvec = np.dot(R, np.matrix(-tvec))
        invRvec, _ = cv2.Rodrigues(R)
        return invRvec, invTvec

    '''
        tracker2WorldRvec, tracker2WorldTvec = Transformation.relativePosition(trackerRvec, trackerTvec, \
                                                              worldRvec, worldTvec)
        R, _ = cv2.Rodrigues(tracker2WorldRvec)
    '''
    @staticmethod
    def relativePosition(rvec1, tvec1, rvec2, tvec2):
        rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
        rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))
        # inverse the second marker
        invRvec, invTvec = Transformation.inversePerspective(rvec2, tvec2)
        info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
        composedRvec, composedTvec = info[0], info[1]
        composedRvec = composedRvec.reshape((3, 1))
        composedTvec = composedTvec.reshape((3, 1))
        return composedRvec, composedTvec