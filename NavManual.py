from control.LQRController import *
from planning.CubicSplinePlanner import *
from common.MouseInteraction import MouseInteraction
from common.OccupancyMap import OccupancyMap
import cv2
import rospy
from common.Transformation import Transformation

H = GlobalVar.getValue('Height')
W = GlobalVar.getValue('Width')
scale = GlobalVar.getValue('scale')

class NavManual():
    def __init__(self, *args):
        rospy.init_node('MonoNav', disable_signals=True)
        GlobalVar._init()

    def run(self):
        # trajectory
        seg_data, floorID = self.loadData()
        ax, ay = self.selfDefinedRoute(seg_data, floorID)
        # from image to maps
        #ax, ay = Transformation.resolutionAxis2CameraAxis(ax, ay, mtx=GlobalVar.getValue('mtx'))
        ax = [x/scale for x in ax]
        ay = [y/scale for y in ay]
        # interpolation
        cx, cy, cyaw, ck, s, goal = self.trajectoryInterpolation(ax, ay, ds=0.1)
        # speed curve take account of the direction
        sp = calc_speed_profile(cx, cy, cyaw, GlobalVar.getValue('targetSpeed'))
        stateList = stateTrajectories(cx, cy, cyaw)
        closed_loop_prediction(stateList, goal, cx, cy)

    def loadData(self):
        # dataset
        rgb_filename = 'mono_depth/results/capture.png'
        depth_filename = 'mono_depth/results/depth.npy'
        seg_filename = 'mono_depth/results/segmentation.npy'
        # data loading
        rgb_data = cv2.imread(rgb_filename)
        seg_data = np.load(seg_filename).astype(float)
        depth_data = np.load(depth_filename)
        # resize to larger size
        w, h = H, W
        rgb_data = cv2.resize(rgb_data, (h, w))
        seg_data = cv2.resize(seg_data, (h, w))
        floor = MouseInteraction(rgb_data)
        floor.displayCircle()
        floorPoint = floor.getPointSelected()
        fx = floorPoint[0]
        fy = floorPoint[1]
        floorID = seg_data[fy, fx]
        rospy.sleep(0.5)
        return seg_data, floorID

    def selfDefinedRoute(self, seg_data, floorID):
        # create occupancy map
        occu = OccupancyMap(seg_data, floorID)
        occuMap = occu.convert2Occu() # in gray scale
        # show the available zone and draw a route on map
        occuMapColor = cv2.cvtColor(occuMap*255, cv2.COLOR_GRAY2RGB) # in color
        # define route
        route = MouseInteraction(occuMapColor)
        route.displayRoute()
        trajectory = route.getTrajectory()
        # sample the trajectory point with a sampling period of 5
        trajectory = [trajectory[i*5] for i in range(int(len(trajectory)/5))]
        ax = [float(i[0]) for i in trajectory]
        ay = [float(i[1]) for i in trajectory]
        # using source as origin point (0,0)
        ax, ay = Transformation.convertCoordinates(ax, ay)
        return ax, ay

    def trajectoryInterpolation(self, ax, ay, ds):
        cx, cy, cyaw, ck, s = calc_spline_course(ax, ay, ds=ds)
        goal = [ax[-1], ay[-1]]
        # plot the trajectory
        plt.subplots(1)
        plt.plot(ax, ay, "xb", label="input")
        plt.plot(cx, cy, "-r", label="spline")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()
        plt.show()
        return cx, cy, cyaw, ck, s, goal

if __name__ == '__main__':
    Nav = NavManual()
    Nav.run()