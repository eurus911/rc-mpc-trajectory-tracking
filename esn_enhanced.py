"""
增强版ESN：支持增量预测、残差模式、单位圆归一化、多维ridge alpha等
发布版修复了角度解码，并支持独立轨迹的储备池重置。
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, Union, List
from robot import DiffDriveSim
from core_utils import normalize_angle, DataTools


class ESNEnhanced:
    """
    增强版储备池计算网络
    - 支持增量预测模式（局部坐标系）
    - 支持残差模式（residual learning，提升多步稳定性）
    - 支持sin/cos单位圆归一化
    - 支持多维ridge alpha
    - 支持特征增强（不增加reservoir）
    - 支持数据增强（训练时）
    """
    
    def __init__(self,
                 n_inputs: int,
                 n_reservoir: int = 500,
                 n_outputs: int = 4,
                 spectral_radius: float = 0.8,
                 sparsity: float = 0.15,
                 leak_rate: float = 0.5,
                 ridge_alpha: Optional[Union[float, np.ndarray]] = None,
                 use_incremental: bool = True,
                 use_residual: bool = True,
                 normalize_unit_circle: bool = True,
                 use_feature_augmentation: bool = True,
                 dt: float = 0.05,
                 v_limits: Tuple[float, float] = (0.0, 0.6),
                 omega_limits: Tuple[float, float] = (-1.5, 1.5),
                 seed: int = 42) -> None:
        if n_inputs != 5 or n_outputs != 4 or not use_incremental:
            raise ValueError("publication edition supports 5 inputs and 4 incremental outputs")
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.use_incremental = use_incremental
        self.use_residual = use_residual and use_incremental  # 残差模式仅在增量模式下有效
        self.normalize_unit_circle = normalize_unit_circle
        self.use_feature_augmentation = use_feature_augmentation
        self.dt = dt
        self.v_limits = v_limits
        self.omega_limits = omega_limits
        
        # ridge_alpha处理
        if ridge_alpha is None:
            ridge_alpha = 5e-4
        if np.isscalar(ridge_alpha):
            self.ridge_alpha = np.ones(n_outputs) * ridge_alpha
        else:
            self.ridge_alpha = np.array(ridge_alpha)
        assert len(self.ridge_alpha) == n_outputs
        
        self.rng = np.random.default_rng(seed)
        self._init_weights()
        self.is_fitted = False
        
        # 存储增量统计量（用于语义一致性）
        self.delta_mean = None
        self.delta_std = None
        
    def _init_weights(self):
        """初始化权重"""
        # 输入权重: reservoir_size x (1 + n_inputs)
        self.W_in = self.rng.uniform(-1, 1,
                                     size=(self.n_reservoir, 1 + self.n_inputs))
        
        # reservoir递归权重
        W = self.rng.uniform(-1, 1,
                             size=(self.n_reservoir, self.n_reservoir))
        mask = self.rng.random((self.n_reservoir, self.n_reservoir)) < self.sparsity
        W *= mask
        
        # 调整谱半径
        eigvals = np.linalg.eigvals(W)
        rho_W = np.max(np.abs(eigvals))
        if rho_W == 0:
            rho_W = 1.0
        self.W = W * (self.spectral_radius / rho_W)
        
        self.W_out = None  # 线性读出层权重
        
    def _reservoir_step(self, r, inp):
        """Reservoir状态更新"""
        z = np.concatenate(([1.0], inp))  # 加bias
        pre_act = self.W_in @ z + self.W @ r
        r_new = (1 - self.leak_rate) * r + self.leak_rate * np.tanh(pre_act)
        return r_new

    def _compat_get_weights(self):
        """
        兼容性方法：获取权重矩阵
        返回: W_in, W_reservoir (兼容旧接口)
        """
        return self.W_in, self.W
    
    def _normalize_unit_circle_angle(self, sin_cos):
        """
        将sin/cos角度对归一化到单位圆（只对角度部分）
        sin_cos: (..., 2) 或 (..., n, 2)，最后两维是[sin, cos]
        返回: 归一化后的sin_cos
        """
        if not self.normalize_unit_circle:
            return sin_cos
        
        shape = sin_cos.shape
        if len(shape) == 1:
            # 单个向量，假设最后两维是sin/cos
            sin_val, cos_val = sin_cos[-2], sin_cos[-1]
            norm = np.sqrt(sin_val**2 + cos_val**2) + 1e-8
            sin_cos = sin_cos.copy()
            sin_cos[-2] = sin_val / norm
            sin_cos[-1] = cos_val / norm
        else:
            # 批量处理
            sin_cos = sin_cos.copy()
            sin_vals = sin_cos[..., -2]
            cos_vals = sin_cos[..., -1]
            norm = np.sqrt(sin_vals**2 + cos_vals**2) + 1e-8
            sin_cos[..., -2] = sin_vals / norm
            sin_cos[..., -1] = cos_vals / norm
        
        return sin_cos
    
    def _compute_incremental_target(self, X, Y, U=None):
        """
        计算增量目标（局部坐标系）或残差目标
        X: (T, 3) 当前状态 [x, y, θ]
        Y: (T, 3) 下一状态 [x', y', θ']
        U: (T, 2) 控制输入 [v, omega]（残差模式需要）
        返回: (T, 4) [Δx_body, Δy_body, sin(Δθ), cos(Δθ)] 或残差
        """
        T = X.shape[0]
        Y_delta = np.zeros((T, 4))
        
        for t in range(T):
            if self.use_residual and U is not None:
                # 残差模式：计算名义预测和真实值的差异
                x_kin_next = DiffDriveSim.step(X[t], U[t], self.dt)
                
                # 计算真实增量
                dx_global = Y[t, 0] - X[t, 0]
                dy_global = Y[t, 1] - X[t, 1]
                dtheta_real = Y[t, 2] - X[t, 2]
                
                # 计算名义增量
                dx_kin_global = x_kin_next[0] - X[t, 0]
                dy_kin_global = x_kin_next[1] - X[t, 1]
                dtheta_kin = x_kin_next[2] - X[t, 2]
                
                # 残差（在机体系下）
                theta = X[t, 2]
                cos_th = np.cos(theta)
                sin_th = np.sin(theta)
                
                # 真实增量（机体系）
                dx_body_real = cos_th * dx_global + sin_th * dy_global
                dy_body_real = -sin_th * dx_global + cos_th * dy_global
                
                # 名义增量（机体系）
                dx_body_kin = cos_th * dx_kin_global + sin_th * dy_kin_global
                dy_body_kin = -sin_th * dx_kin_global + cos_th * dy_kin_global
                
                # 残差 = 真实 - 名义
                dx_body_residual = dx_body_real - dx_body_kin
                dy_body_residual = dy_body_real - dy_body_kin
                dtheta_residual = normalize_angle(dtheta_real - dtheta_kin)
                
                Y_delta[t, 0] = dx_body_residual
                Y_delta[t, 1] = dy_body_residual
                Y_delta[t, 2] = np.sin(dtheta_residual)
                Y_delta[t, 3] = np.cos(dtheta_residual)
            else:
                # 增量模式：直接计算增量
                dx_global = Y[t, 0] - X[t, 0]
                dy_global = Y[t, 1] - X[t, 1]
                dtheta = Y[t, 2] - X[t, 2]
                
                # 旋转到机器人坐标系
                theta = X[t, 2]
                cos_th = np.cos(theta)
                sin_th = np.sin(theta)
                dx_body = cos_th * dx_global + sin_th * dy_global
                dy_body = -sin_th * dx_global + cos_th * dy_global
                
                # 角度增量（wrap到[-pi, pi]）
                dtheta = normalize_angle(dtheta)
                
                Y_delta[t, 0] = dx_body
                Y_delta[t, 1] = dy_body
                Y_delta[t, 2] = np.sin(dtheta)
                Y_delta[t, 3] = np.cos(dtheta)
        
        return Y_delta
    
    def _clamp_delta_physical(self, delta_real, u_real):
        """
        物理约束：clamp增量到物理上界
        delta_real: (4,) [dx_body, dy_body, sin(dtheta), cos(dtheta)] 真实值
        u_real: (2,) [v, omega] 控制输入
        返回: 约束后的delta_real
        """
        dt = self.dt
        
        # 物理上界：最大位移和角度增量
        max_dx = self.v_limits[1] * dt  # 最大线速度 * dt
        max_dy = self.v_limits[1] * dt
        max_dtheta = abs(self.omega_limits[1]) * dt  # 最大角速度 * dt
        
        # clamp位置增量
        delta_real[0] = np.clip(delta_real[0], -max_dx, max_dx)
        delta_real[1] = np.clip(delta_real[1], -max_dy, max_dy)
        
        # clamp角度增量
        dtheta = np.arctan2(delta_real[2], delta_real[3])
        dtheta = np.clip(dtheta, -max_dtheta, max_dtheta)
        delta_real[2] = np.sin(dtheta)
        delta_real[3] = np.cos(dtheta)
        
        return delta_real
    
    def fit(self, X, U, Y_global, washout=100, noise_std=0.01, stats=None, sequence_starts=None):
        """
        训练ESN读出层（支持多维ridge alpha和特征增强）
        X: (T, n_x) 当前状态（归一化）
        U: (T, n_u) 当前控制（归一化）
        Y_global: (T, 3) 下一状态（全局坐标，未归一化）
        washout: washout步数
        noise_std: 数据增强噪声标准差（训练时对X,U加噪声）
        stats: 归一化统计量（用于计算增量目标，必须包含x_mean/x_std, u_mean/u_std）
        """
        T = X.shape[0]
        assert T == U.shape[0] == Y_global.shape[0]
        assert Y_global.shape[1] == 3, "Y_global must be (T, 3) global coordinates"
        n_x = X.shape[1]
        n_u = U.shape[1]
        assert n_x + n_u == self.n_inputs
        
        # 反归一化X和U以计算增量
        if stats is not None:
            X_real = X * stats["x_std"] + stats["x_mean"]
            U_real = U * stats["u_std"] + stats["u_mean"]
        else:
            X_real = X
            U_real = U
        
        # 转换为增量目标或全局目标
        if self.use_incremental:
            # 计算增量目标（局部坐标系）或残差目标
            Y_delta = self._compute_incremental_target(X_real, Y_global, U_real)
            Y_target = Y_delta  # (T, 4): [Δx_body, Δy_body, sin(Δθ), cos(Δθ)] 或残差
            
            # 计算增量/残差的统计量（关键：用于归一化训练目标）
            self.delta_mean = np.mean(Y_target, axis=0)
            self.delta_std = np.std(Y_target, axis=0) + 1e-8
            
            # 归一化增量目标（关键：使得phi@W_out输出就是归一化量）
            Y_target_norm = (Y_target - self.delta_mean) / self.delta_std
        r = np.zeros(self.n_reservoir)
        Rs, Ts, kept = [], [], []
        sequence_starts = set([0] if sequence_starts is None else sequence_starts)
        age = 0

        for t in range(T):
            if t in sequence_starts:
                r[:] = 0.0
                age = 0
            # 数据增强：添加小噪声
            if noise_std > 0:
                X_noisy = X[t] + self.rng.normal(0, noise_std, size=X[t].shape)
                U_noisy = U[t] + self.rng.normal(0, noise_std, size=U[t].shape)
            else:
                X_noisy = X[t]
                U_noisy = U[t]
            
            inp = np.concatenate([X_noisy, U_noisy])
            r = self._reservoir_step(r, inp)
            
            if age >= washout:
                Rs.append(r.copy())
                Ts.append(Y_target_norm[t])
                kept.append(t)
            age += 1

        if not kept:
            raise ValueError("washout leaves no training samples")
        Rs = np.stack(Rs)  # (T-washout, n_reservoir)
        Ts = np.stack(Ts)  # (T-washout, n_outputs) - 已经是归一化的
        
        # 特征增强：拼接输入特征到readout
        if self.use_feature_augmentation:
            Phi = np.hstack([np.ones((Rs.shape[0], 1)), Rs, U[kept]])
        else:
            Phi = np.hstack([np.ones((Rs.shape[0], 1)), Rs])
        
        # 多维ridge回归：每个输出维度使用不同的alpha
        n_features = Phi.shape[1]
        self.W_out = np.zeros((n_features, self.n_outputs))
        
        for j in range(self.n_outputs):
            alpha = self.ridge_alpha[j]
            A = Phi.T @ Phi + alpha * np.eye(n_features)
            b = Phi.T @ Ts[:, j]  # Ts已经是归一化的
            self.W_out[:, j] = np.linalg.solve(A, b)
        
        self.is_fitted = True
        self.r0_trained = r.copy()
        
        # Unit-circle geometry applies to physical sin/cos, after decoding.
        self.unit_circle_error_train = None

        return self
    
    def _decode_increment(self, prediction):
        """Decode the standardized readout before enforcing angle geometry."""
        return self._normalize_unit_circle_angle(
            prediction * self.delta_std + self.delta_mean
        )

    def predict_sequence(self, X, U, stats=None, sequence_starts=None):
        """Predict body-frame increments; stats selects physical output units.

        Inputs are standardized. With stats=None the output remains standardized.
        In residual mode the increments are residuals relative to DiffDriveSim.
        """
        if not self.is_fitted:
            raise ValueError("fit the model before prediction")
        r = np.zeros(self.n_reservoir)
        predictions = []
        starts = set([0] if sequence_starts is None else sequence_starts)
        for t in range(len(X)):
            if t in starts:
                r[:] = 0.0
            r = self._reservoir_step(r, np.concatenate([X[t], U[t]]))
            phi = np.concatenate([[1.0], r, U[t]]) if self.use_feature_augmentation else np.r_[1.0, r]
            pred = phi @ self.W_out
            if self.use_incremental:
                physical = self._decode_increment(pred)
                pred = physical if stats is not None else (physical - self.delta_mean) / self.delta_std
            predictions.append(pred)
        return np.asarray(predictions)

    def rollout(self, x0, U_seq, r0=None, stats=None):
        """
        MPC使用的rollout（保持接口不变）
        x0: (n_x,) 归一化的当前状态 [x, y, θ]
        U_seq: (N, n_u) 归一化的控制序列
        r0: reservoir初始状态
        stats: 归一化统计量（必需，必须包含x_mean/x_std, y_mean/y_std, u_mean/u_std, delta_mean/delta_std）
        
        返回: (Y_pred_global, r_final)
            Y_pred_global: (N+1, 4) 归一化的全局输出 [x, y, sin(θ), cos(θ)]
            与mpc.py期望的格式一致
        """
        assert self.is_fitted
        assert stats is not None, "stats required for rollout"
        
        n_x = x0.shape[0]
        n_u = U_seq.shape[1]
        assert n_x + n_u == self.n_inputs
        
        N = U_seq.shape[0]
        r = np.zeros(self.n_reservoir) if r0 is None else r0.copy()
        
        # 反归一化控制序列（用于物理约束和残差模式）
        U_seq_real = DataTools.denormalize(U_seq, stats["u_mean"], stats["u_std"])
        
        # 获取增量统计量（从stats或self）
        if self.use_incremental:
            if "delta_mean" in stats and "delta_std" in stats:
                delta_mean = stats["delta_mean"]
                delta_std = stats["delta_std"]
            elif self.delta_mean is not None and self.delta_std is not None:
                delta_mean = self.delta_mean
                delta_std = self.delta_std
            else:
                raise ValueError("delta_mean/delta_std not found in stats or self. Please ensure fit() was called correctly.")
        
        if self.use_incremental:
            # 增量模式：预测增量，然后积分（关键：自回归更新）
            Y_pred_global = np.zeros((N + 1, 4))
            x_mean, x_std = stats["x_mean"], stats["x_std"]
            y_mean, y_std = stats["y_mean"], stats["y_std"]
            
            # 反归一化初始状态
            x0_real = x0 * x_std + x_mean
            x_curr = x0_real[0]
            y_curr = x0_real[1]
            theta_curr = x0_real[2]
            
            # 初始状态的归一化输出（使用y_mean/y_std，与mpc.py期望一致）
            # x和y使用y_mean/y_std归一化（全局输出格式）
            Y_pred_global[0, 0] = (x_curr - y_mean[0]) / y_std[0]
            Y_pred_global[0, 1] = (y_curr - y_mean[1]) / y_std[1]
            sin_theta0 = np.sin(theta_curr)
            cos_theta0 = np.cos(theta_curr)
            if self.normalize_unit_circle:
                norm0 = np.sqrt(sin_theta0**2 + cos_theta0**2) + 1e-8
                sin_theta0 /= norm0
                cos_theta0 /= norm0
            Y_pred_global[0, 2] = (sin_theta0 - y_mean[2]) / y_std[2]
            Y_pred_global[0, 3] = (cos_theta0 - y_mean[3]) / y_std[3]
            
            # 自回归：每步预测后立即更新x_curr_norm（关键修复）
            x_curr_norm = x0.copy()  # 初始归一化状态
            
            for k in range(N):
                # 1. 预测增量（归一化）- 使用当前更新后的x_curr_norm
                inp = np.concatenate([x_curr_norm, U_seq[k]])
                r = self._reservoir_step(r, inp)
                
                if self.use_feature_augmentation:
                    phi = np.concatenate([[1.0], r, U_seq[k]])
                else:
                    phi = np.concatenate([[1.0], r])
                
                delta_pred_norm = phi @ self.W_out  # 归一化增量
                # Decode first: standardized sin/cos do not lie on a unit circle.
                delta_real = self._normalize_unit_circle_angle(
                    delta_pred_norm * delta_std + delta_mean
                )
                
                dx_body = delta_real[0]
                dy_body = delta_real[1]
                dtheta = np.arctan2(delta_real[2], delta_real[3])
                
                # 3. 残差模式：加上名义预测
                if self.use_residual:
                    u_real = U_seq_real[k]
                    x_kin_next = DiffDriveSim.step(
                        np.array([x_curr, y_curr, theta_curr]), 
                        u_real, 
                        self.dt
                    )
                    
                    # 名义增量（机体系）
                    cos_th = np.cos(theta_curr)
                    sin_th = np.sin(theta_curr)
                    dx_kin_global = x_kin_next[0] - x_curr
                    dy_kin_global = x_kin_next[1] - y_curr
                    dx_body_kin = cos_th * dx_kin_global + sin_th * dy_kin_global
                    dy_body_kin = -sin_th * dx_kin_global + cos_th * dy_kin_global
                    dtheta_kin = normalize_angle(x_kin_next[2] - theta_curr)
                    
                    # 真实增量 = 名义增量 + 残差
                    dx_body += dx_body_kin
                    dy_body += dy_body_kin
                    dtheta += dtheta_kin
                
                # 4. 物理约束clamp
                delta_real_clamped = self._clamp_delta_physical(
                    np.array([dx_body, dy_body, np.sin(dtheta), np.cos(dtheta)]),
                    U_seq_real[k]
                )
                dx_body = delta_real_clamped[0]
                dy_body = delta_real_clamped[1]
                dtheta = np.arctan2(delta_real_clamped[2], delta_real_clamped[3])
                
                # 5. 旋转到全局坐标系并积分（关键：立即更新状态）
                cos_th = np.cos(theta_curr)
                sin_th = np.sin(theta_curr)
                dx_global = cos_th * dx_body - sin_th * dy_body
                dy_global = sin_th * dx_body + cos_th * dy_body
                
                x_curr += dx_global
                y_curr += dy_global
                theta_curr += dtheta
                theta_curr = normalize_angle(theta_curr)
                
                # 6. 计算sin/cos并归一化到单位圆（只对角度部分）
                sin_theta = np.sin(theta_curr)
                cos_theta = np.cos(theta_curr)
                if self.normalize_unit_circle:
                    norm = np.sqrt(sin_theta**2 + cos_theta**2) + 1e-8
                    sin_theta /= norm
                    cos_theta /= norm
                
                # 7. 归一化全局输出（使用y_mean/y_std）并写入Y_pred_global
                # x和y使用y_mean/y_std归一化（全局输出格式，与mpc.py期望一致）
                Y_pred_global[k+1, 0] = (x_curr - y_mean[0]) / y_std[0]
                Y_pred_global[k+1, 1] = (y_curr - y_mean[1]) / y_std[1]
                Y_pred_global[k+1, 2] = (sin_theta - y_mean[2]) / y_std[2]
                Y_pred_global[k+1, 3] = (cos_theta - y_mean[3]) / y_std[3]
                
                # 8. 更新x_curr_norm用于下一步（关键：自回归，必须立即更新）
                x_real_3d = np.array([x_curr, y_curr, theta_curr])
                x_curr_norm = (x_real_3d - x_mean) / x_std
        return Y_pred_global, r

