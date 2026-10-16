import numpy as np


class DiffDriveSim:
    """差速驱动机器人离散时间运动学模型"""

    @staticmethod
    def step(x, u, dt):
        """
        x: [x, y, theta]
        u: [v, omega]
        dt: 采样周期
        """
        x_, y_, theta = x
        v, omega = u

        x_next = x_ + v * np.cos(theta) * dt
        y_next = y_ + v * np.sin(theta) * dt
        theta_next = theta + omega * dt

        # 角度归一到 [-pi, pi]
        theta_next = (theta_next + np.pi) % (2 * np.pi) - np.pi
        return np.array([x_next, y_next, theta_next])
