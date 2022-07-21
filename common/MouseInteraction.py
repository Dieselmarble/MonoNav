import time
from common import GlobalVar
import cv2
import rospy
from common.State import State
from rospy_tutorials.msg import Floats
from common.Transformation import Transformation
from communication.WheelOdomReceiver import WheelOdomReceiver
from communication.VisualOdomReceiver import VisualOdomReceiver

H = GlobalVar.getValue('Height')
W = GlobalVar.getValue('Width')
scale = GlobalVar.getValue('scale')
Hr = GlobalVar.getValue('perspective')

class MouseInteraction:
    def __init__(self, img):
        self.img = img
        self.trajectory = []
        self.pointSelected = None
        # 当鼠标按下时为True
        # self.wheelOdomReceiver = WheelOdomReceiver()
        # self.visualOdomReceiver = VisualOdomReceiver()

    def getTrajectory(self):
        return self.trajectory

    def getPointSelected(self):
        return self.pointSelected

    def displayRoute(self):
        # coordinates on non-wrapped image
        #stateWheel = self.wheelOdomReceiver.getStateWheel()
        #center = (int(state.x*scale), int(abs(state.y*scale - H)))
        # visual state from visual odometry
        data = rospy.wait_for_message("robot_cor", Floats)
        # transform center to image axis
        #center = Transformation.visualOdom2Image(stateVisual.x, stateVisual.y)
        center = (int(data.data[0]), int(data.data[1]))
        # draw marker of robot and warp perspective
        self.img = cv2.drawMarker(self.img, center, color=(0,255,0), thickness=2)
        # convert to BEV
        Hr = GlobalVar.getValue('perspective')
        self.img = cv2.warpPerspective(self.img, Hr, (W, H))
        # world origin don't need second perspective
        ori = GlobalVar.getValue('origin') # 世界坐标系origin在图片中的位置
        self.img = cv2.drawMarker(self.img, ori, color=(255,255,0), thickness=2)
        # show map under BEV
        print("Please draw the route on screen")
        cv2.namedWindow('Occupancy Map')
        cv2.setMouseCallback('Occupancy Map', self.drawRoute)
        # for now, only append one route with two points
        while (1):
            cv2.resizeWindow('Occupancy Map', W, H)
            cv2.imshow('Occupancy Map', self.img)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
        cv2.destroyAllWindows()

    def displayCircle(self):
        print("Please select one point on the floor")
        cv2.namedWindow('RGB Map')
        cv2.setMouseCallback('RGB Map', self.drawCircle)
        # for now, only append one route with two points
        while (1 and len(self.trajectory)<1):
            cv2.resizeWindow('RGB Map', W, H)
            cv2.imshow('RGB Map', self.img)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
        cv2.destroyAllWindows()

    def drawCircle(self, event, x, y, flags, param):
        global ix, iy, drawing
        # 当按下左键时返回起始位置坐标
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            xy = "%d,%d" % (x, y)
            self.pointSelected = (ix,iy)
            # print coordinates
            self.trajectory.append((x, y))
            cv2.circle(self.img, (x, y), 3, (0, 255, 255), -1)
            cv2.putText(self.img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (255, 255, 255), thickness=1)

    # 创建回调函数
    def drawRoute(self, event, x, y, flags, param):
        global ix, iy, drawing
        # 当按下左键时返回起始位置坐标
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            # print coordinates
            self.trajectory.append((x,y))
            cv2.circle(self.img, (x, y), 3, (0, 255, 255), -1)

        # 当左键按下并移动时绘制图形，event可以查看移动，flag查看是否按下
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            if drawing == True:
                self.trajectory.append((x, y))
                # 绘制圆圈，小圆点连在一起就成了线，3代表笔画的粗细
                cv2.circle(self.img, (x, y), 3, (0, 255, 255), -1)

        # 当鼠标松开时停止绘图
        elif event == cv2.EVENT_LBUTTONUP:
            drawing == False
            self.trajectory.append((x, y))
            return