import numpy as np

from robot import DiffDriveSim
from trajectory import TrajectoryGenerator


class BaselineController:
    """
    基线控制器：几何追踪 + 简单增益，类似纯追踪
    """

    @staticmethod
    def get_action(x, x_ref, v_ref,
                   k_theta=2.0, k_y=1.0,
                   v_limits=(0.0, 0.6),
                   omega_limits=(-1.5, 1.5)):
        """
        输入:
            x: 当前状态 [x, y, theta]
            x_ref: 目标状态 [x, y, theta]
            v_ref: 期望前进速度
        输出:
            u = [v, omega]
        """
        # 世界坐标误差
        dx = x_ref[0] - x[0]
        dy = x_ref[1] - x[1]
        theta = x[2]

        # 转到车体坐标系
        ex_body = np.cos(-theta) * dx - np.sin(-theta) * dy
        ey_body = np.sin(-theta) * dx + np.cos(-theta) * dy

        # 航向误差
        theta_ref = x_ref[2]
        heading_error = theta_ref - theta
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        # 控制律：v 使用参考速度并考虑航向误差，omega 用航向 + 横向误差
        v = v_ref * np.cos(heading_error)
        omega = k_theta * heading_error + k_y * ey_body

        # 饱和
        v = np.clip(v, *v_limits)
        omega = np.clip(omega, *omega_limits)
        return np.array([v, omega])


def collect_dataset_from_baseline(dt=0.05,
                                  traj_types=("line", "circle", "s_curve"),
                                  num_trials_per_type=4,
                                  T_per_trial=150,
                                  v_ref_range=(0.2, 0.5),
                                  seed=0):
    """
    用基线控制器在直线 / 圆弧 / S 曲线上跑，收集 (x_t,u_t,x_{t+1}) 数据，用于 ESN 训练
    """
    rng = np.random.default_rng(seed)
    X_list, U_list, Y_list = [], [], []

    for traj_type in traj_types:
        for _ in range(num_trials_per_type):
            v_ref = rng.uniform(*v_ref_range)
            total_time = T_per_trial * dt

            # 生成参考轨迹（内部带轻微随机化）
            x_ref_seq = TrajectoryGenerator.generate(
                traj_type, total_time, dt, v_ref, rng=rng
            )

            # 初始状态为参考起点附近的小扰动
            x0 = x_ref_seq[0].copy()
            x0[0] += rng.normal(scale=0.05)
            x0[1] += rng.normal(scale=0.05)
            x0[2] += rng.normal(scale=0.05)
            x = x0

            for k in range(T_per_trial):
                x_ref_k = x_ref_seq[k + 1]  # 看下一时刻的参考点
                u = BaselineController.get_action(
                    x, x_ref_k, v_ref,
                    k_theta=2.0, k_y=1.0,
                    v_limits=(0.0, 0.6),
                    omega_limits=(-1.5, 1.5)
                )
                x_next = DiffDriveSim.step(x, u, dt)

                X_list.append(x.copy())
                U_list.append(u.copy())
                Y_list.append(x_next.copy())

                x = x_next

    X = np.stack(X_list)
    U = np.stack(U_list)
    Y = np.stack(Y_list)  # (T, 3): [x', y', θ']
    
    # 构造4维输出 Y4 = [x', y', sin(θ'), cos(θ')]
    Y4 = np.zeros((Y.shape[0], 4))
    Y4[:, 0] = Y[:, 0]  # x'
    Y4[:, 1] = Y[:, 1]  # y'
    Y4[:, 2] = np.sin(Y[:, 2])  # sin(θ')
    Y4[:, 3] = np.cos(Y[:, 2])  # cos(θ')
    
    return X, U, Y, Y4
