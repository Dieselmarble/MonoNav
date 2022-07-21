import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from common.State import State

class EKF():
    def __init__(self):
        self.ud = np.zeros(3)
        self.z = np.zeros(2)
        # Predicted posteriori estimate covariance
        self.PEst = np.eye(5)
        # Predicted (a priori) estimate covariance
        self.xEst = np.zeros(5)
        # Covariance for EKF simulation
        self.Q = np.diag([
            0.1,  # variance of location on x-axis
            0.1,  # variance of location on y-axis
            1.0,  # variance of x velocity
            1.0,  # variance of y velocity
            np.deg2rad(1.0),  # variance of yaw angle
        ]) ** 2  # predict state covariance
        self.R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance
        self.DT = 0.1  # time tick [s]

    def combinedFilter(self, stateWheel, stateVisual):
        self.setSensorReading(stateWheel)
        self.setInputValue(stateVisual)
        xEst = self.filtering()  # X = [x, y, v_x, v_y, yaw]
        stateEst = State()
        stateEst.x = xEst[0]  # State = [x, y, yaw, vx, vy, w]
        stateEst.y = xEst[1]
        stateEst.vx = xEst[2]
        stateEst.vy = xEst[3]
        stateEst.yaw = xEst[4]
        return stateEst

    """
    Get sensor readings from the visual odometry
    """
    def getSensorReading(self):
        return self.z

    """
    Get input vector u from the LQR controller
    """
    def getInputValue(self):
        return self.ud

    def setInputValue(self, optimalInput):
        self.ud = optimalInput

    def setSensorReading(self, state):
        z = np.zeros(2)
        z[0] = state.x
        z[1] = state.y

    """
    State vector
    X = [x, y, v_x, v_y, yaw]
    motion model
    x_{t+1} = x_t+v_x*dt
    y_{t+1} = y_t+v_y*dt
    v_x{t+1} = v_x{t}
    v_y{t+1} = v_y{t}
    yaw_{t+1} = yaw_t+omega*dt
    Y = Fx + Bu
    """
    def motion_model(self, x, u):
        F = np.array([[1.0, 0., 0., 0., 0.],
                      [0., 1.0, 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 1.0]])
        # Input [v_x v_y w]
        B = np.array([[self.DT, 0., 0.],
                      [self.DT, 0., 0.],
                      [1.0, 0., 0.],
                      [1.0, 0., 0],
                      [0., self.DT, 0.]])
        x = F @ x + B @ u
        return x

    """
    only observe the position vector
    """
    def observation_model(x):
        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ])
        z = H @ x
        return z

    """
    Jacobian of Motion Model
    motion model
    x_{t+1} = x_t+v_x*dt
    y_{t+1} = y_t+v_y*dt
    v_x{t+1} = v_x{t}
    v_y{t+1} = v_y{t}
    yaw_{t+1} = yaw_t+omega*dt
    so
    dx/dt = v_x
    dy/dt = v_y
    dx/dy = v_x/v_y
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    def jacob_f(self):
        jF = np.array([
            [1.0, 0.0, self.DT, 0.0, 0],
            [0.0, 1.0, 0,0, self.DT, 0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]])
        return jF

    """
    Jacobian of Observation Model
    """
    def jacob_h(self):
        jH = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ])
        return jH

    """
    Fk, the state-transition model;
    Hk, the observation model;
    Qk, the covariance of the process noise;
    Rk, the covariance of the observation noise;
    zk, observation (or measurement) of the true state xk
    """
    def ekf_estimation(self, xEst, PEst, z, u):
        #  Predict
        xPred = self.motion_model(xEst, u)
        jF = self.jacob_f()
        PPred = jF @ PEst @ jF.T + self.Q
        #  Update
        jH = self.jacob_h()
        zPred = self.observation_model(xPred)
        y = z - zPred # residual
        S = jH @ PPred @ jH.T + self.R # covariance
        K = PPred @ jH.T @ np.linalg.inv(S) # kalman gain
        # update x and p
        xEst = xPred + K @ y
        PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
        return xEst, PEst

    def filtering(self, xEst):
        z = self.getSensorReading()
        ud = self.getInputValue()
        xEst, PEst = self.ekf_estimation(xEst, self.PEst, z, ud)
        self.PEst = PEst
        self.xEst = xEst
        return xEst