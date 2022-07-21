import os
import numpy as np

def _init():
    os.system("C:/opt/ros/noetic/catkin_ws/devel/setup.bat")
    os.system("C:/opt/ros/noetic/x64/setup.bat")
    os.environ['ROS_MASTER_URI'] = "http://192.168.8.131:11311/"
    os.environ['ROS_IP'] = "192.168.8.129"
    #os.environ['ROS_MASTER_URI'] = "http://127.0.0.1:11311/"
    #os.environ['ROS_IP'] = "127.0.0.1"
    # dist matrix
    #dist = np.array(([[-0.58650416, 0.59103816, -0.00443272, 0.00357844, -0.27203275]]))
    dist = np.array([[0.1563097, -0.50631796, -0.00239552, -0.00120906, 0.4788988]])
    # camera intrinsics
    mtx = np.array([[889.69615837, 0. , 647.5929987],
                    [0.,889.09279472, 351.73822108],
                    [0., 0.,  1.]])

    perspective = np.load('d:/MonoNav/mono_depth/results/perspective.npy').astype(float)
    # global variables
    global _globalDict
    _globalDict = {'Width': 1280,
                   'Height': 720,
                   'targetSpeed': 3.0/3.6,  #km/h -> m/s
                   'linearLimit': 0.15,
                   'angularLimit': 0.00,
                   'errorThreshold': 0.023,
                   'goalThreshold': 0.050,
                   'showAnimation': True,
                   'commFrequency': 5,
                   'distMatrix': dist,
                   'mtx': mtx,
                   'scale': 600,
                   'world':  np.float32([[640, 500], [640, 530], [670, 500], [670, 530]]),
                   'origin': (655,515), # world origin, don't need second perspective
                   'perspective': perspective
    }

def setValue(key, value):
    _globalDict[key] = value

def getValue(key, defValue=None):
    try:
        return _globalDict[key]
    except KeyError:
        return defValue