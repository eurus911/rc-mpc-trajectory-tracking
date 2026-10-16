import numpy as np


class TrajectoryGenerator:
    """生成直线 / 圆弧 / S 型参考轨迹"""

    @staticmethod
    def generate_line(total_time, dt, v_ref=0.3):
        T_steps = int(total_time / dt)
        t = np.arange(T_steps + 1) * dt
        x = v_ref * t
        y = np.zeros_like(t)
        theta = np.zeros_like(t)
        return np.stack([x, y, theta], axis=1)

    @staticmethod
    def generate_circle(total_time, dt, v_ref=0.3, R=1.0, direction=1):
        """
        R: 半径
        direction: 1 逆时针, -1 顺时针
        """
        T_steps = int(total_time / dt)
        t = np.arange(T_steps + 1) * dt

        omega_c = direction * v_ref / R
        x = R * np.cos(omega_c * t)
        y = R * np.sin(omega_c * t)
        theta = omega_c * t + direction * np.pi / 2.0
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        return np.stack([x, y, theta], axis=1)

    @staticmethod
    def generate_s_curve(total_time, dt, v_ref=0.3, A=0.5, freq=0.1):
        """
        x 方向匀速前进，y = A sin(2π f t) 的 S 型轨迹
        """
        T_steps = int(total_time / dt)
        t = np.arange(T_steps + 1) * dt

        x = v_ref * t
        y = A * np.sin(2 * np.pi * freq * t)

        dy_dt = A * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)
        dx_dt = np.full_like(t, v_ref)
        theta = np.arctan2(dy_dt, dx_dt)
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        return np.stack([x, y, theta], axis=1)

    @staticmethod
    def generate(traj_type, total_time, dt, v_ref=0.3,
                 rng=None):
        """
        统一接口：
            traj_type: 'line' / 'circle' / 's_curve'
        对 circle / s_curve 做一点随机化，让数据多样一些
        """
        if rng is None:
            rng = np.random.default_rng()

        if traj_type == "line":
            return TrajectoryGenerator.generate_line(total_time, dt, v_ref)
        elif traj_type == "circle":
            R = rng.uniform(0.8, 1.5)
            direction = 1 if rng.random() < 0.5 else -1
            return TrajectoryGenerator.generate_circle(total_time, dt, v_ref, R, direction)
        elif traj_type == "s_curve":
            A = rng.uniform(0.3, 0.8)
            freq = rng.uniform(0.05, 0.15)
            return TrajectoryGenerator.generate_s_curve(total_time, dt, v_ref, A, freq)
        else:
            raise ValueError(f"Unknown trajectory type: {traj_type}")
