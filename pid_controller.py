"""
PID控制器实现
任务书要求：PID或纯追踪基线控制
"""

import numpy as np
from core_utils import normalize_angle


class PIDController:
    """
    PID控制器用于差速机器人轨迹跟踪
    位置控制使用PID，角度控制使用PID
    """
    
    def __init__(self,
                 kp_x=2.0, ki_x=0.0, kd_x=0.5,
                 kp_y=2.0, ki_y=0.0, kd_y=0.5,
                 kp_theta=3.0, ki_theta=0.0, kd_theta=0.5,
                 v_limits=(0.0, 0.6),
                 omega_limits=(-1.5, 1.5)):
        """
        参数:
            kp_x, ki_x, kd_x: x方向PID参数
            kp_y, ki_y, kd_y: y方向PID参数
            kp_theta, ki_theta, kd_theta: 角度PID参数
        """
        self.kp_x, self.ki_x, self.kd_x = kp_x, ki_x, kd_x
        self.kp_y, self.ki_y, self.kd_y = kp_y, ki_y, kd_y
        self.kp_theta, self.ki_theta, self.kd_theta = kp_theta, ki_theta, kd_theta
        self.v_limits = v_limits
        self.omega_limits = omega_limits
        
        # 积分项和上次误差（用于每个控制器实例）
        self.reset()
    
    def reset(self):
        """重置积分项和上次误差"""
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.integral_theta = 0.0
        self.last_error_x = 0.0
        self.last_error_y = 0.0
        self.last_error_theta = 0.0
    
    @staticmethod
    def get_action(x, x_ref, v_ref,
                   kp_x=2.0, ki_x=0.0, kd_x=0.5,
                   kp_y=2.0, ki_y=0.0, kd_y=0.5,
                   kp_theta=3.0, ki_theta=0.0, kd_theta=0.5,
                   v_limits=(0.0, 0.6),
                   omega_limits=(-1.5, 1.5),
                   dt=0.05,
                   integral_state=None,
                   last_error_state=None):
        """
        静态方法：计算PID控制输出
        
        参数:
            x: 当前状态 [x, y, theta]
            x_ref: 目标状态 [x, y, theta]
            v_ref: 期望前进速度（用于速度前馈）
            integral_state: 积分项状态 [integral_x, integral_y, integral_theta]
            last_error_state: 上次误差 [last_x, last_y, last_theta]
        
        返回:
            u: [v, omega]
        """
        if integral_state is None:
            integral_state = np.zeros(3)
        if last_error_state is None:
            last_error_state = np.zeros(3)
        
        # 计算误差（世界坐标系）
        e_x = x_ref[0] - x[0]
        e_y = x_ref[1] - x[1]
        e_theta = normalize_angle(x_ref[2] - x[2])
        
        # 转换到车体坐标系
        theta = x[2]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        e_x_body = cos_theta * e_x + sin_theta * e_y
        e_y_body = -sin_theta * e_x + cos_theta * e_y
        e_theta_body = e_theta
        
        # PID控制
        # x方向（前进方向）
        integral_x = integral_state[0] + e_x_body * dt
        derivative_x = (e_x_body - last_error_state[0]) / dt if dt > 0 else 0.0
        u_x = kp_x * e_x_body + ki_x * integral_x + kd_x * derivative_x
        
        # y方向（横向）
        integral_y = integral_state[1] + e_y_body * dt
        derivative_y = (e_y_body - last_error_state[1]) / dt if dt > 0 else 0.0
        u_y = kp_y * e_y_body + ki_y * integral_y + kd_y * derivative_y
        
        # 角度
        integral_theta = integral_state[2] + e_theta_body * dt
        derivative_theta = (e_theta_body - last_error_state[2]) / dt if dt > 0 else 0.0
        u_theta = kp_theta * e_theta_body + ki_theta * integral_theta + kd_theta * derivative_theta
        
        # 速度合成：前进速度 = 前馈 + PID修正，角速度 = 角度PID
        v = v_ref + u_x
        omega = u_theta + 0.5 * u_y  # 横向误差也影响角速度
        
        # 饱和
        v = np.clip(v, *v_limits)
        omega = np.clip(omega, *omega_limits)
        
        # 更新状态（用于下次调用）
        new_integral_state = np.array([integral_x, integral_y, integral_theta])
        new_last_error_state = np.array([e_x_body, e_y_body, e_theta_body])
        
        return np.array([v, omega]), new_integral_state, new_last_error_state

