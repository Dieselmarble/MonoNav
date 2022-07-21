import cv2
import scipy.linalg as la
import matplotlib.pyplot as plt
import math
import numpy as np
from communication.CommandPublisher import Broadcaster
from communication.WheelOdomReceiver import WheelOdomReceiver
from communication.VisualOdomReceiver import VisualOdomReceiver
from control.LQRSolver import LQR
from common import GlobalVar
from common.State import State
from rospy_tutorials.msg import Floats
import rospy
from control.EKF import EKF

try:
    from planning import CubicSplinePlanner
except:
    raise

GlobalVar._init()
linearLimit = GlobalVar.getValue('linearLimit')
angularLimit = GlobalVar.getValue('angularLimit')
errorThreshold = GlobalVar.getValue('errorThreshold')
goalThreshold = GlobalVar.getValue('goalThreshold')
showAnimation = GlobalVar.getValue('showAnimation')
commFrequency = GlobalVar.getValue('commFrequency')
scale = GlobalVar.getValue('scale')

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def solve_DARE(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    X = Q
    maxiter = 50
    eps = 0.01
    for i in range(maxiter):
        Xn = A.T @ X @ A - A.T @ X @ B @ \
            la.inv(R + B.T @ X @ B) @ B.T @ X @ A + Q
        if (abs(Xn - X)).max() < eps:
            break
        X = Xn

    return Xn


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_DARE(A, B, Q, R)

    # compute the LQR gain
    K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    eigVals, eigVecs = la.eig(A - B @ K)

    return K, X, eigVals

def calcNearestIndex(state, trajectory, removedStates):
    # distance from x to every trajectory point
    dx = []
    dy = []
    # all points on trajectory
    for i in range(len(trajectory)):
        if i in removedStates:
            dx.append(100000)
            dy.append(100000)
        else:
            dx.append(state.x - trajectory[i].x)
            dy.append(state.y - trajectory[i].y)
    # direct distance
    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
    # min distance^2
    mind = min(d)
    # index of min distance
    ind = d.index(mind)
    # min distance
    mind = math.sqrt(mind)
    # xy distance to nearest point
    dxl = trajectory[ind].x - state.x
    dyl = trajectory[ind].y - state.y
    # trajectory yaw - point angle
    angle = pi_2_pi(trajectory[ind].yaw - math.atan2(dyl, dxl))
    # turn around
    if angle < 0:
        mind *= -1
    for i in range(0,ind):
        removedStates.add(i)
    if len(removedStates) == len(trajectory):
        ind = len(trajectory)-1
    return ind, mind

def getStateError(state, target_state):
    # state_error = np.asarray([
    #     state.x-target_state.x,
    #     state.y-target_state.y,
    #     state.yaw-target_state.yaw])

    state_error = np.asarray([
        state.x-target_state.x,
        state.y-target_state.y,])

    state_error_magnitude = np.linalg.norm(state_error)
    #print(f'State Error Magnitude = {state_error_magnitude}')
    return state_error_magnitude

def getVisualOdometry():
    data = rospy.wait_for_message("robot_cor", Floats)
    stateVisual = State()
    stateVisual.x = data.data[0]
    stateVisual.y = data.data[1]
    pts = np.array([stateVisual.x, stateVisual.y])
    pts = pts.reshape(1, -1, 2).astype(np.float32)
    Hr = GlobalVar.getValue('perspective')
    dst = cv2.perspectiveTransform(pts, Hr)
    stateVisual.x = dst[0][0][0]/scale
    H = GlobalVar.getValue('Height')
    stateVisual.y = abs(H - dst[0][0][1])/scale
    return stateVisual

def updateState(stateWheelThis, stateWheelInitial, stateVisualInitial):
    xMoved = stateWheelThis.x - stateWheelInitial.x
    yMoved = stateWheelThis.y - stateWheelInitial.y
    stateCurrent = State()
    print("stateVisualInitial" + str(stateVisualInitial))
    stateCurrent.x = stateVisualInitial.x + xMoved
    stateCurrent.y += stateVisualInitial.y + yMoved
    stateCurrent.vx = stateWheelThis.vx
    stateCurrent.yaw = stateWheelThis.yaw
    return stateCurrent

def closed_loop_prediction(trajectory, goal, cx, cy):
    # initialize controllers and communication
    broadcaster = Broadcaster()
    broadcaster.setSpeedFactor(0.10)
    broadcaster.setTurnFactor(0.0)
    # odom receiver
    wheelOdomReceiver = WheelOdomReceiver()
    # LQR controller
    controller = LQR()
    # kalman filter
    ekf = EKF()

    # visual mode
    A = np.eye(3)
    Q = np.array([[0.637, 0, 0],
                  [0, 0.637, 0],
                  [0, 0, 0.637]])
    R = np.array([[0.0136, 0, 0],
                  [0, 0.0136, 0],
                  [0, 0, 0.0136]])

    '''
    # odometry mode
    Q = np.array([[0.837, 0, 0],
                  [0, 0.837, 0],
                  [0, 0, 0.837]])
    R = np.array([[0.0136, 0, 0],
                  [0, 0.0136, 0],
                  [0, 0, 0.0136]])
    '''

    r = rospy.Rate(commFrequency)  # 10hz
    dt = 1/commFrequency
    # initial position
    stateVisual = getVisualOdometry()
    stateVisualInitial = stateVisual
    stateWheelInitial = wheelOdomReceiver.getStateWheel()
    print(stateVisualInitial)

    # walked states
    removedStates = set()
    # get the closest point by pure pursuit
    targetInd, _ = calcNearestIndex(stateVisual, trajectory, removedStates)
    target_state = State()
    # motion trajectory
    while not rospy.is_shutdown():
        '''
         Odometry mode
        '''
        #stateWheelThis = wheelOdomReceiver.getStateWheel()
        #state = updateState(stateWheelThis, stateWheelInitial, stateVisualInitial)

        '''
         Pure visual mode
        '''
        state = getVisualOdometry()
        print(state)

        # stateEst = ekf.combinedFilter(ekf, stateWheel, stateVisual)
        # find the nearest state
        targetInd, _ = calcNearestIndex(state, trajectory, removedStates)
        target_state.x = trajectory[targetInd].x
        target_state.y = trajectory[targetInd].y
        target_state.yaw = trajectory[targetInd].yaw
        # LQR control
        state_error = getStateError(state, target_state)
        B = controller.getB(dt)
        # LQR returns the optimal control input
        optimal_control_input = controller.lqr(state,
                                    target_state,
                                    Q, R, A, B)
        # broadcast the command through communication
        xSpeed = speedFilter(optimal_control_input[0], linearLimit)
        ySpeed = speedFilter(optimal_control_input[1], linearLimit)
        wSpeed = speedFilter(optimal_control_input[2], angularLimit)
        broadcaster.setLinearSpeedXYZ(xSpeed, ySpeed, 0),
        broadcaster.setTh(wSpeed)
        broadcaster.setSpeedFactor(1.0)
        broadcaster.setTurnFactor(1.0)
        broadcaster.publish()
        print(f'control Input = {optimal_control_input}')
        # switch state
        if state_error < errorThreshold:
            removedStates.add(targetInd)
            #target_ind+=1
            #print("\nSwitch State!")
        # stop condition
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        if math.hypot(dx, dy) <= goalThreshold:
            print("Goal")
            break
        # next round control
        r.sleep()
        # plot trajectory
        if targetInd % 1 == 0 and showAnimation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(state.x, state.y, "ob", label="trajectory")
            plt.plot(target_state.x, target_state.y, "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("speed[km/h]:" + str(round(state.getLinearSpeed() * 3.6, 2)) + " "
                      + "yaw[degrees]:" + str(round(state.yaw/math.pi*360, 2)) + " "
                      + "target index:" + str(targetInd))
            plt.pause(0.001)

def speedFilter(speedInput, speedLimit):
    speedInput = speedInput
    speedOutput = speedInput
    if speedInput > speedLimit:
        speedOutput = speedLimit
    elif speedInput < -speedLimit:
        speedOutput = -speedLimit
    #print(str(speedInput) + '/=======/' + str(speedOutput))
    return speedOutput

def calc_speed_profile(cx, cy, cyaw, target_speed):
    speed_profile = [target_speed] * len(cx)
    direction = 1.0
    # Set stop point
    for i in range(len(cx) - 1):
        dyaw = abs(cyaw[i + 1] - cyaw[i])
        switch = math.pi / 4.0 <= dyaw < math.pi / 2.0
        if switch:
            direction *= -1
        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed
        if switch:
            speed_profile[i] = 0.0
    speed_profile[-1] = 0.0
    return speed_profile

def stateTrajectories(cx, cy, cyaw):
    stateList = []
    for x, y, yaw in zip(cx, cy, cyaw):
        state = State()
        state.x = x
        state.y = y
        state.yaw = 0
        stateList.append(state)
    return stateList

def main():
    GlobalVar._init()
    print("LQR steering control tracking start!!")
    ax = [0.0, 0.5, 0.89, 5.24]
    ay = [0.0, 0.6, 0.36, 5.36]
    #ax, ay = Transformation.resolutionAxis2ImageAxis(ax, ay, GlobalVar.getValue('mtx'))
    goal = [ax[-1], ay[-1]]
    cx, cy, cyaw, ck, s = CubicSplinePlanner.calc_spline_course(
       ax, ay, ds=0.02)
    target_speed = 3.0 / 3.6  # simulation parameter km/h -> m/s
    sp = calc_speed_profile(cx, cy, cyaw, target_speed)
    stateList = stateTrajectories(cx, cy, cyaw)
    closed_loop_prediction(stateList, goal, cx, cy)

if __name__ == '__main__':
    main()