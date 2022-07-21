import numpy as np


class LQR:
    def __init__(self):

        ######################## DEFINE CONSTANTS #####################################
        # Supress scientific notation when printing NumPy arrays
        np.set_printoptions(precision=3, suppress=True)
        # Optional Variables
        self.max_linear_velocity = 3.0  # meters per second
        self.max_angular_velocity = 1.5708  # radians per second

    def getB(self, deltat):
        """
        Calculates and returns the B matrix
        3x2 matix ---> number of states x number of control inputs

        Expresses how the state of the system [x,y,yaw] changes
        from t-1 to t due to the control commands (i.e. control inputs).

        :param yaw: The yaw angle (rotation angle around the z axis) in radians
        :param deltat: The change in time from timestep t-1 to t in seconds

        :return: B matrix ---> 3x2 NumPy array
        """
        B = np.array([[1.0 * deltat, 0, 0],
                      [0, 1.0 * deltat, 0],
                      [0, 0, 1.0 * deltat]])
        return B


    def state_space_model(self, A, state_t_minus_1, B, control_input_t_minus_1):
        """
        Calculates the state at time t given the state at time t-1 and
        the control inputs applied at time t-1

        :param: A   The A state transition matrix
            3x3 NumPy Array
        :param: state_t_minus_1     The state at time t-1
            3x1 NumPy Array given the state is [x,y,yaw angle] --->
            [meters, meters, radians]
        :param: B   The B state transition matrix
            3x2 NumPy Array
        :param: control_input_t_minus_1     Optimal control inputs at time t-1
            2x1 NumPy Array given the control input vector is
            [linear velocity of the car, angular velocity of the car]
            [meters per second, radians per second]

        :return: State estimate at time t
            3x1 NumPy Array given the state is [x,y,yaw angle] --->
            [meters, meters, radians]
        """
        # These next 6 lines of code which place limits on the angular and linear
        # velocities of the robot car can be removed if you desire.
        control_input_t_minus_1[0] = np.clip(control_input_t_minus_1[0],
                                             -self.max_linear_velocity,
                                             self.max_linear_velocity)
        control_input_t_minus_1[1] = np.clip(control_input_t_minus_1[1],
                                             -self.max_angular_velocity,
                                             self.max_angular_velocity)
        state_estimate_t = (A @ state_t_minus_1) + (B @ control_input_t_minus_1)

        return state_estimate_t


    def lqr(self, actual_state_x, desired_state_xf, Q, R, A, B):
        """
        Discrete-time linear quadratic regulator for a nonlinear system.

        Compute the optimal control inputs given a nonlinear system, cost matrices,
        current state, and a final state.

        Compute the control variables that minimize the cumulative cost.

        Solve for P using the dynamic programming method.

        :param actual_state_x: The current state of the system
            3x1 NumPy Array given the state is [x,y,yaw angle] --->
            [meters, meters, radians]
        :param desired_state_xf: The desired state of the system
            3x1 NumPy Array given the state is [x,y,yaw angle] --->
            [meters, meters, radians]
        :param Q: The state cost matrix
            3x3 NumPy Array
        :param R: The input cost matrix
            2x2 NumPy Array
        :param dt: The size of the timestep in seconds -> float

        :return: u_star: Optimal action u for the current state
            2x1 NumPy Array given the control input vector is
            [linear velocity of the car, angular velocity of the car]
            [meters per second, radians per second]
        """
        # We want the system to stabilize at desired_state_xf.
        x_error = np.asarray([
            actual_state_x.x - desired_state_xf.x,
            actual_state_x.y - desired_state_xf.y,
            actual_state_x.yaw - desired_state_xf.yaw])


        # Solutions to discrete LQR problems are obtained using the dynamic
        # programming method.
        # The optimal solution is obtained recursively, starting at the last
        # timestep and working backwards.
        # You can play with this number
        N = 50

        # Create a list of N + 1 elements
        P = [None] * (N + 1)

        Qf = Q

        # LQR via Dynamic Programming
        P[N] = Qf

        # For i = N, ..., 1
        for i in range(N, 0, -1):
            # Discrete-time Algebraic Riccati equation to calculate the optimal
            # state cost matrix
            P[i - 1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
                R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)

            # Create a list of N elements
        K = [None] * N
        u = [None] * N

        # For i = 0, ..., N - 1
        for i in range(N):
            # Calculate the optimal feedback gain K
            K[i] = -np.linalg.pinv(R + B.T @ P[i + 1] @ B) @ B.T @ P[i + 1] @ A

            u[i] = K[i] @ x_error

        # Optimal control input is u_star
        u_star = u[N - 1]

        return u_star