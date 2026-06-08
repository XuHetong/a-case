#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一弹塑性损伤本构模型 —— 三维应变驱动数值验证
======================================================

基于连续损伤力学 (CDM) 与不可逆热力学框架，对单个材料点进行
应变驱动的数值模拟。严格保留所有张量运算，不做一维简化。

理论框架：
- 自由能包含二阶 (四阶刚度) 和三阶 (六阶刚度) 弹性项
- Murakami 对称化损伤效应张量 M(D)
- Hill (von Mises) 各向异性屈服准则
- 关联塑性流动法则
- 由塑性变形驱动的传统损伤 D (二阶张量) 和早期非线性损伤 κ (标量)

Voigt 约定：采用保范 (norm-preserving) 约定
  ε_v = [ε₁₁, ε₂₂, ε₃₃, √2·ε₂₃, √2·ε₁₃, √2·ε₁₂]
  σ_v = [σ₁₁, σ₂₂, σ₃₃, √2·σ₂₃, √2·σ₁₃, √2·σ₁₂]
  C_v[α,β] = C_{ijkl}（无额外因子）
"""

import numpy as np
from numpy.linalg import inv, eigvalsh, eigh, norm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互后端，适合无 GUI 环境

# ---- 配置中文字体 ----
# Windows 11 下优先使用微软雅黑，备选黑体、宋体
_chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
_font_found = None
for _f in _chinese_fonts:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        _font_found = _f
        break
    except Exception:
        continue

if _font_found:
    plt.rcParams['font.sans-serif'] = [_font_found] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示为方块
    print(f"[字体] 已加载中文字体: {_font_found}")
else:
    print("[字体] 未找到中文字体，将回退到英文标签")

import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 辅助函数
# ============================================================================

def mat_to_voigt(m):
    """
    将 3×3 对称矩阵转换为 6 分量保范 Voigt 向量。

    Parameters
    ----------
    m : ndarray, shape (3, 3)

    Returns
    -------
    v : ndarray, shape (6,)
        Voigt 向量: [11, 22, 33, √2·23, √2·13, √2·12]
    """
    v = np.zeros(6)
    v[0] = m[0, 0]
    v[1] = m[1, 1]
    v[2] = m[2, 2]
    v[3] = np.sqrt(2) * m[1, 2]
    v[4] = np.sqrt(2) * m[0, 2]
    v[5] = np.sqrt(2) * m[0, 1]
    return v


def voigt_to_mat(v):
    """
    将 6 分量保范 Voigt 向量转换为 3×3 对称矩阵。

    Parameters
    ----------
    v : ndarray, shape (6,)

    Returns
    -------
    m : ndarray, shape (3, 3)
    """
    m = np.zeros((3, 3))
    m[0, 0] = v[0]
    m[1, 1] = v[1]
    m[2, 2] = v[2]
    m[0, 1] = m[1, 0] = v[5] / np.sqrt(2)
    m[0, 2] = m[2, 0] = v[4] / np.sqrt(2)
    m[1, 2] = m[2, 1] = v[3] / np.sqrt(2)
    return m


def positive_projection(mat):
    """
    计算对称矩阵的正定投影 ⟨A⟩_+：
    对矩阵进行特征分解，将负特征值置零，重构矩阵。

    Parameters
    ----------
    mat : ndarray, shape (3, 3)

    Returns
    -------
    mat_pos : ndarray, shape (3, 3)
    """
    eigvals, eigvecs = eigh(mat)
    eigvals_pos = np.maximum(eigvals, 0.0)
    mat_pos = eigvecs @ np.diag(eigvals_pos) @ eigvecs.T
    return mat_pos


def smooth_step(x, beta):
    """
    平滑阶跃函数（Sigmoid 近似 Heaviside）。

    H(x) ≈ 1 / (1 + exp(-β·x))

    Parameters
    ----------
    x : float or ndarray
    beta : float
        锐度参数。

    Returns
    -------
    float or ndarray
    """
    bx = np.clip(beta * x, -100.0, 100.0)
    return 1.0 / (1.0 + np.exp(-bx))


# ============================================================================
# Voigt 指标映射
# ============================================================================

_VOIGT_UNMAP = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


# ============================================================================
# 主类：三维弹塑性损伤本构模型
# ============================================================================

class DamagePlasticityModel3D:
    """
    统一弹塑性损伤本构模型（三维，保范 Voigt 约定）。

    状态变量
    --------
    eps      : (3,3) 总应变张量
    eps_e    : (3,3) 弹性应变张量
    eps_p    : (3,3) 塑性应变张量
    D        : (3,3) 传统各向异性损伤张量（对称二阶）
    kappa    : float 早期非线性损伤标量，∈ [0, 1)
    p        : float 累积等效塑性应变，≥ 0

    材料常数
    --------
    弹性    : E (Young), nu (Poisson)
    三阶    : l_murn, m_murn, n_murn (Murnaghan 常数)
    塑性    : sigma_y (初始屈服应力), K (线性硬化模量)
    损伤-D  : S, s, Y_th_D
    损伤-κ  : S_kappa, s_kappa, Y_th_kappa
    数值    : beta (Sigmoid 锐度)
    """

    def __init__(self, params):
        """
        Parameters
        ----------
        params : dict
        """
        # ---- 弹性常数 ----
        self.E = params['E']
        self.nu = params['nu']
        self.lame = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.mu_shear = self.E / (2.0 * (1.0 + self.nu))

        # ---- Murnaghan 三阶弹性常数 ----
        self.l_murn = params['l_murn']
        self.m_murn = params['m_murn']
        self.n_murn = params['n_murn']

        # ---- 塑性参数 ----
        self.sigma_y = params['sigma_y']
        self.K = params['K']

        # ---- 传统损伤参数 ----
        self.S = params['S']
        self.s = params['s']
        self.Y_th_D = params['Y_th_D']

        # ---- 早期损伤参数 ----
        self.S_kappa = params['S_kappa']
        self.s_kappa = params['s_kappa']
        self.Y_th_kappa = params['Y_th_kappa']

        # ---- 数值参数 ----
        self.beta = params['beta']
        self.tol = params.get('tol', 1e-10)
        self.max_iter = params.get('max_iter', 50)
        self.fd_h = params.get('fd_h', 1e-8)
        self.D_max = params.get('D_max', 0.99)

        # ---- 构建常张量 ----
        self._build_constant_tensors()

        # ---- 初始化状态 ----
        self.reset()

    def reset(self):
        """将所有状态变量重置为初始值。"""
        self.eps = np.zeros((3, 3))
        self.eps_e = np.zeros((3, 3))
        self.eps_p = np.zeros((3, 3))
        self.D = np.zeros((3, 3))
        self.kappa = 0.0
        self.p = 0.0
        self._yield_flag = False
        self._damage_D_flag = False
        self._damage_kappa_flag = False

    # ------------------------------------------------------------------
    def _build_constant_tensors(self):
        """
        构建不随损伤演化的常四阶张量（Voigt 6×6 矩阵形式）。

        使用保范 Voigt 约定：
          应力型向量：σ_v = [σ₁₁, σ₂₂, σ₃₃, √2σ₂₃, √2σ₁₃, √2σ₁₂]
          应变型向量：ε_v = [ε₁₁, ε₂₂, ε₃₃, √2ε₂₃, √2ε₁₃, √2ε₁₂]
          四阶张量 C_v[α,β] = C_{i(α)j(α)k(β)l(β)}，无额外因子。
        """
        lam = self.lame
        mu = self.mu_shear

        # ---- C₀⁽²⁾：无损各向同性四阶刚度张量 ----
        # C_ijkl = λ δ_ij δ_kl + μ (δ_ik δ_jl + δ_il δ_jk)
        self.C0_2 = np.zeros((6, 6))
        self.C0_2[0:3, 0:3] = lam
        for i in range(3):
            self.C0_2[i, i] += 2.0 * mu
        self.C0_2[3, 3] = 2.0 * mu
        self.C0_2[4, 4] = 2.0 * mu
        self.C0_2[5, 5] = 2.0 * mu

        # ---- H：Hill 屈服张量（此处取各向同性 von Mises） ----
        # H = (3/2) I_dev
        # 保范 Voigt 形式：
        self.H = np.zeros((6, 6))
        self.H[0:3, 0:3] = -0.5
        for i in range(3):
            self.H[i, i] = 1.0
        self.H[3, 3] = 1.5
        self.H[4, 4] = 1.5
        self.H[5, 5] = 1.5

        # ---- H⁺（Moore-Penrose 伪逆） ----
        # H = (3/2) I_dev 在 Voigt 空间中秩为 5（体积分量为零）
        # H⁺ = (2/3) I_dev, 仅在偏量空间有意义
        self.H_pinv = np.linalg.pinv(self.H)

        # ---- I_s（对称化恒等四阶张量） ----
        self.I_s = np.eye(6)

    # ------------------------------------------------------------------
    # Murakami 损伤效应张量
    # ------------------------------------------------------------------

    def _build_murakami_M(self, D_mat):
        """
        构建 Murakami 对称化四阶损伤效应张量 M(D)（Voigt 6×6 矩阵）。

        定义（指标形式）：
            M_ijkl = 1/4 [ (δ-D)⁻¹_ik δ_jl  +  δ_ik (δ-D)⁻¹_jl
                         + (δ-D)⁻¹_il δ_jk  +  δ_il (δ-D)⁻¹_jk ]

        具有对称性：M_ijkl = M_jikl = M_ijlk = M_klij。

        Parameters
        ----------
        D_mat : ndarray, shape (3, 3)

        Returns
        -------
        M_v : ndarray, shape (6, 6)
        """
        delta = np.eye(3)
        # 防止 (δ - D) 奇异
        D_eigvals, D_eigvecs = eigh(D_mat)
        D_eigvals = np.clip(D_eigvals, -np.inf, self.D_max)
        D_mat_safe = D_eigvecs @ np.diag(D_eigvals) @ D_eigvecs.T

        delta_minus_D = delta - D_mat_safe
        delta_minus_D_inv = inv(delta_minus_D)

        M_v = np.zeros((6, 6))
        for a, (i, j) in enumerate(_VOIGT_UNMAP):
            for b, (k, l) in enumerate(_VOIGT_UNMAP):
                val = 0.25 * (
                    delta_minus_D_inv[i, k] * delta[j, l] +
                    delta[i, k] * delta_minus_D_inv[j, l] +
                    delta_minus_D_inv[i, l] * delta[j, k] +
                    delta[i, l] * delta_minus_D_inv[j, k]
                )
                M_v[a, b] = val

        return M_v

    # ------------------------------------------------------------------
    # 受损四阶刚度张量
    # ------------------------------------------------------------------

    def _compute_C_tilde_2(self, D_mat):
        """
        C̃⁽²⁾(D) = M⁻¹(D) : C₀⁽²⁾

        Returns
        -------
        C_tilde_2_v : ndarray, shape (6, 6)
        M_inv_v : ndarray, shape (6, 6)
        M_v : ndarray, shape (6, 6)
        """
        M_v = self._build_murakami_M(D_mat)
        M_inv_v = inv(M_v)
        C_tilde_2_v = M_inv_v @ self.C0_2
        return C_tilde_2_v, M_inv_v, M_v

    # ------------------------------------------------------------------
    # 三阶弹性项（Murnaghan 应变能）
    # ------------------------------------------------------------------

    def _compute_third_order_stress(self, eps_e_mat):
        """
        计算无损三阶应力 σ³ = ∂ψ³/∂ε。

        Murnaghan 三阶应变能密度：
            ψ³ = (l+2m)/3·I₁³ - 2m·I₁·I₂ + n·I₃

        I₁ = tr(ε), I₂ = (tr²(ε) - tr(ε²))/2, I₃ = det(ε)。

        应力贡献（利用 Cayley-Hamilton）：
            σ³ = [l·I₁² - 2m·I₂]·I + 2m·I₁·ε + n·cof(ε)
            cof(ε) = ε² - I₁·ε + I₂·I

        Parameters
        ----------
        eps_e_mat : ndarray, shape (3, 3)

        Returns
        -------
        sigma3_mat : ndarray, shape (3, 3)
        """
        I1 = np.trace(eps_e_mat)
        eps2 = eps_e_mat @ eps_e_mat
        I2 = 0.5 * (I1**2 - np.trace(eps2))

        # 余子式矩阵 (cofactor)
        cof = eps2 - I1 * eps_e_mat + I2 * np.eye(3)

        sigma3_mat = (
            (self.l_murn * I1**2 - 2.0 * self.m_murn * I2) * np.eye(3) +
            2.0 * self.m_murn * I1 * eps_e_mat +
            self.n_murn * cof
        )
        return sigma3_mat

    def _compute_third_order_energy(self, eps_e_mat):
        """
        计算无损三阶应变能密度：
            ψ³ = (l+2m)/3·I₁³ - 2m·I₁·I₂ + n·I₃

        Parameters
        ----------
        eps_e_mat : ndarray, shape (3, 3)

        Returns
        -------
        psi3 : float
        """
        I1 = np.trace(eps_e_mat)
        eps2 = eps_e_mat @ eps_e_mat
        I2 = 0.5 * (I1**2 - np.trace(eps2))
        I3 = np.linalg.det(eps_e_mat)
        psi3 = ((self.l_murn + 2.0 * self.m_murn) / 3.0 * I1**3
                - 2.0 * self.m_murn * I1 * I2
                + self.n_murn * I3)
        return psi3

    # ------------------------------------------------------------------
    # 应力计算
    # ------------------------------------------------------------------

    def compute_stress(self, eps_e_mat, D_mat, kappa, return_details=False):
        """
        计算应力张量。

        σ = C̃⁽²⁾(D) : εᵉ + (1-κ) · σ³(εᵉ)

        Parameters
        ----------
        eps_e_mat : ndarray, shape (3, 3)
        D_mat : ndarray, shape (3, 3)
        kappa : float
        return_details : bool

        Returns
        -------
        sigma_mat : ndarray, shape (3, 3)
        sigma_v : ndarray, shape (6,)
        details : dict (if return_details)
        """
        # 线性部分
        C_tilde_2_v, M_inv_v, M_v = self._compute_C_tilde_2(D_mat)
        eps_e_v = mat_to_voigt(eps_e_mat)
        sigma_lin_v = C_tilde_2_v @ eps_e_v
        sigma_lin_mat = voigt_to_mat(sigma_lin_v)

        # 三阶部分
        sigma3_mat = self._compute_third_order_stress(eps_e_mat)
        sigma_total_mat = sigma_lin_mat + (1.0 - kappa) * sigma3_mat

        sigma_v = mat_to_voigt(sigma_total_mat)

        if return_details:
            return sigma_total_mat, sigma_v, {
                'C_tilde_2_v': C_tilde_2_v,
                'M_v': M_v,
                'M_inv_v': M_inv_v,
                'sigma_lin_mat': sigma_lin_mat,
                'sigma_lin_v': sigma_lin_v,
                'sigma3_mat': sigma3_mat,
            }
        return sigma_total_mat, sigma_v

    # ------------------------------------------------------------------
    # 损伤驱动力
    # ------------------------------------------------------------------

    def _compute_Y_D(self, eps_e_v, D_mat, C_tilde_2_v=None):
        """
        计算传统损伤驱动力 Y_D = -∂ψ/∂D。

        对 D 的 6 个独立 Voigt 分量，通过中心有限差分计算梯度。

        Parameters
        ----------
        eps_e_v : ndarray, shape (6,)
        D_mat : ndarray, shape (3, 3)
        C_tilde_2_v : ndarray, optional

        Returns
        -------
        Y_D_mat : ndarray, shape (3, 3)
        Y_D_v : ndarray, shape (6,)
        """
        h = self.fd_h

        if C_tilde_2_v is None:
            C_tilde_2_v, _, _ = self._compute_C_tilde_2(D_mat)

        Y_D_v = np.zeros(6)

        for alpha in range(6):
            i, j = _VOIGT_UNMAP[alpha]

            # 正向扰动
            Dp = D_mat.copy()
            Dp[i, j] += h
            Dp[j, i] = Dp[i, j]
            Ctp, _, _ = self._compute_C_tilde_2(Dp)
            psi_p = 0.5 * eps_e_v @ Ctp @ eps_e_v

            # 负向扰动
            Dm = D_mat.copy()
            Dm[i, j] -= h
            Dm[j, i] = Dm[i, j]
            Ctm, _, _ = self._compute_C_tilde_2(Dm)
            psi_m = 0.5 * eps_e_v @ Ctm @ eps_e_v

            Y_D_v[alpha] = -(psi_p - psi_m) / (2.0 * h)

        Y_D_mat = voigt_to_mat(Y_D_v)
        return Y_D_mat, Y_D_v

    def _compute_Y_kappa(self, eps_e_mat):
        """
        计算早期损伤驱动力（标量）：
            Y_κ = max(0, ψ³(εᵉ))

        Parameters
        ----------
        eps_e_mat : ndarray, shape (3, 3)

        Returns
        -------
        Y_kappa : float
        """
        psi3 = self._compute_third_order_energy(eps_e_mat)
        return max(0.0, psi3)

    # ------------------------------------------------------------------
    # 屈服函数与流动方向
    # ------------------------------------------------------------------

    def _compute_yield_and_flow(self, sigma_v, D_mat, M_v, p):
        """
        屈服函数与流动方向。

        Φ(σ, D, R) = ‖σ̃‖_H - σ_y - K·p ≤ 0
        σ̃ = M(D) : σ

        流动方向（关联流动）：
            n_v = H · σ̃_v / ‖σ̃_v‖_H

        Returns
        -------
        Phi : float
        sigma_tilde_v : ndarray, shape (6,)
        n_v : ndarray, shape (6,)
        norm_H : float
        """
        sigma_tilde_v = M_v @ sigma_v

        norm_H_sq = sigma_tilde_v @ self.H @ sigma_tilde_v
        norm_H = np.sqrt(max(norm_H_sq, 0.0))

        R = self.K * p
        Phi = norm_H - self.sigma_y - R

        if norm_H > 1e-16:
            n_v = self.H @ sigma_tilde_v / norm_H
        else:
            n_v = np.zeros(6)

        return Phi, sigma_tilde_v, n_v, norm_H

    # ------------------------------------------------------------------
    # 返回映射（Newton-Raphson 塑性修正）
    # ------------------------------------------------------------------

    def _return_mapping(self, eps_e_trial_v, sigma_trial_v, sigma_tilde_trial_v,
                        Phi_trial, n_trial_v, norm_H_trial,
                        D_mat, M_v, C_tilde_2_v, M_inv_v, kappa):
        """
        通过 Newton-Raphson 求解塑性乘子 Δλ，使 Φ = 0。

        塑性流动：
            εᵉ = εᵉ_trial - Δλ · n(σ̃)
            σ  = C̃⁽²⁾ : εᵉ + (1-κ)·σ³(εᵉ)

        径向返回近似（冻结 n）：
            σ̃(Δλ) ≈ σ̃_trial - Δλ · Z
            Z = M : C̃⁽²⁾ : n_trial

        残差：r(Δλ) = ‖σ̃(Δλ)‖_H - σ_y - K·(p + Δλ) = 0

        Returns
        -------
        delta_lambda : float
        converged : bool
        n_iter : int
        """
        p_n = self.p

        # 有效刚度算子 Z = M : C̃⁽²⁾ : n_trial
        Z_v = M_v @ C_tilde_2_v @ n_trial_v

        # 初始猜测 Δλ
        denom = n_trial_v @ Z_v + self.K
        if denom < 1e-16:
            return 0.0, False, 0
        delta_lambda = max(0.0, Phi_trial / denom)

        for it in range(self.max_iter):
            sigma_tilde_v = sigma_tilde_trial_v - delta_lambda * Z_v

            norm_H_sq = sigma_tilde_v @ self.H @ sigma_tilde_v
            norm_H = np.sqrt(max(norm_H_sq, 1e-30))

            # 残差
            r = norm_H - self.sigma_y - self.K * (p_n + delta_lambda)

            if abs(r) < self.tol * (self.sigma_y + self.K * p_n + 1.0):
                return delta_lambda, True, it + 1

            # 导数 dr/dΔλ = -(Zᵀ·H·σ̃)/‖σ̃‖_H - K
            d_norm_H = -(Z_v @ self.H @ sigma_tilde_v) / norm_H
            dr = d_norm_H - self.K

            if abs(dr) < 1e-30:
                break

            delta_lambda_new = delta_lambda - r / dr

            if delta_lambda_new < 0.0:
                delta_lambda_new = 0.5 * delta_lambda

            if abs(delta_lambda_new - delta_lambda) < self.tol:
                delta_lambda = delta_lambda_new
                return delta_lambda, True, it + 1

            delta_lambda = delta_lambda_new

        return max(0.0, delta_lambda), False, self.max_iter

    # ------------------------------------------------------------------
    # 损伤演化
    # ------------------------------------------------------------------

    def _update_damage_D(self, Y_D_mat, dp):
        """
        显式更新传统损伤 D。

        演化方程：
            Y_eq_D = max_i ⟨λ_i(Y_D)⟩_+
            P = ⟨Y_D⟩_+ / ‖⟨Y_D⟩_+‖_F
            dD = dp · (Y_eq_D / S)^s · P · H(Y_eq_D - Y_th_D)

        Parameters
        ----------
        Y_D_mat : ndarray, shape (3, 3)
        dp : float

        Returns
        -------
        dD_mat : ndarray, shape (3, 3)
        Y_eq_D : float
        """
        if dp <= 0.0:
            return np.zeros((3, 3)), 0.0

        Y_D_pos = positive_projection(Y_D_mat)

        eigvals_pos = np.maximum(eigvalsh(Y_D_mat), 0.0)
        Y_eq_D = np.max(eigvals_pos) if len(eigvals_pos) > 0 else 0.0

        H_val = smooth_step(Y_eq_D - self.Y_th_D, self.beta)

        if Y_eq_D <= 0.0 or H_val < 1e-10:
            return np.zeros((3, 3)), Y_eq_D

        norm_Y_pos = norm(Y_D_pos, 'fro')
        if norm_Y_pos > 1e-16:
            P_mat = Y_D_pos / norm_Y_pos
        else:
            P_mat = np.zeros((3, 3))

        dD_mat = dp * (Y_eq_D / self.S)**self.s * P_mat * H_val
        return dD_mat, Y_eq_D

    def _update_damage_kappa(self, Y_kappa, dp):
        """
        显式更新早期损伤标量 κ。

        演化方程：
            dκ = dp · (Y_κ / S_κ)^s_κ · H(Y_κ - Y_th_κ)

        Parameters
        ----------
        Y_kappa : float
        dp : float

        Returns
        -------
        dkappa : float
        """
        if dp <= 0.0:
            return 0.0

        H_val = smooth_step(Y_kappa - self.Y_th_kappa, self.beta)

        if Y_kappa <= 0.0 or H_val < 1e-10:
            return 0.0

        dkappa = dp * (Y_kappa / self.S_kappa)**self.s_kappa * H_val
        return dkappa

    # ------------------------------------------------------------------
    # 主步进函数：给定应变增量，更新状态
    # ------------------------------------------------------------------

    def step(self, delta_eps_mat):
        """
        应变驱动的单步推进。

        1. 弹性预测
        2. 屈服检测
        3. 若屈服，塑性修正（返回映射）
        4. 显式更新损伤变量 D 和 κ

        Parameters
        ----------
        delta_eps_mat : ndarray, shape (3, 3)

        Returns
        -------
        sigma_mat : ndarray, shape (3, 3)
        info : dict
        """
        # ---- 0. 保存上一步状态 ----
        eps_e_old_mat = self.eps_e.copy()
        D_old_mat = self.D.copy()
        kappa_old = self.kappa
        p_old = self.p

        # ---- 1. 弹性预测 ----
        eps_e_trial_mat = eps_e_old_mat + delta_eps_mat
        eps_e_trial_v = mat_to_voigt(eps_e_trial_mat)

        sigma_trial_mat, sigma_trial_v, details = self.compute_stress(
            eps_e_trial_mat, D_old_mat, kappa_old, return_details=True
        )
        M_v = details['M_v']
        C_tilde_2_v = details['C_tilde_2_v']
        M_inv_v = details['M_inv_v']

        # ---- 2. 屈服检测 ----
        Phi_trial, sigma_tilde_trial_v, n_trial_v, norm_H_trial = \
            self._compute_yield_and_flow(sigma_trial_v, D_old_mat, M_v, p_old)

        self._yield_flag = Phi_trial > self.tol

        # ---- 3. 塑性修正 ----
        if self._yield_flag:
            delta_lambda, converged, n_iters = self._return_mapping(
                eps_e_trial_v, sigma_trial_v, sigma_tilde_trial_v,
                Phi_trial, n_trial_v, norm_H_trial,
                D_old_mat, M_v, C_tilde_2_v, M_inv_v, kappa_old
            )

            if delta_lambda > 0.0:
                eps_e_v = eps_e_trial_v - delta_lambda * n_trial_v
                eps_e_mat = voigt_to_mat(eps_e_v)
                delta_eps_p_mat = voigt_to_mat(delta_lambda * n_trial_v)

                self.eps_e = eps_e_mat
                self.eps_p = self.eps_p + delta_eps_p_mat
                self.p = p_old + delta_lambda
            else:
                eps_e_mat = eps_e_trial_mat
                self.eps_e = eps_e_mat
                delta_lambda = 0.0
                converged = True
                n_iters = 0
                self._yield_flag = False
        else:
            eps_e_mat = eps_e_trial_mat
            self.eps_e = eps_e_mat
            delta_lambda = 0.0
            converged = True
            n_iters = 0

        # ---- 4. 重算应力 ----
        sigma_mat, sigma_v = self.compute_stress(eps_e_mat, D_old_mat, kappa_old)

        # ---- 5. 显式更新损伤 ----
        eps_e_v = mat_to_voigt(self.eps_e)
        dp = delta_lambda

        # 传统损伤 D
        Y_D_mat, Y_D_v = self._compute_Y_D(eps_e_v, D_old_mat, C_tilde_2_v)
        dD_mat, Y_eq_D = self._update_damage_D(Y_D_mat, dp)
        D_new_mat = D_old_mat + dD_mat
        D_new_mat = 0.5 * (D_new_mat + D_new_mat.T)
        D_eigvals, D_eigvecs = eigh(D_new_mat)
        D_eigvals_clipped = np.clip(D_eigvals, 0.0, self.D_max)
        D_new_mat = D_eigvecs @ np.diag(D_eigvals_clipped) @ D_eigvecs.T
        self.D = D_new_mat
        self._damage_D_flag = Y_eq_D > self.Y_th_D + self.tol

        # 早期损伤 κ
        Y_kappa = self._compute_Y_kappa(eps_e_mat)
        dkappa = self._update_damage_kappa(Y_kappa, dp)
        self.kappa = min(kappa_old + dkappa, self.D_max)
        self._damage_kappa_flag = Y_kappa > self.Y_th_kappa + self.tol

        # ---- 6. 更新总应变 ----
        self.eps = self.eps + delta_eps_mat

        # ---- 整理返回信息 ----
        info = {
            'yield': self._yield_flag,
            'delta_lambda': delta_lambda,
            'converged': converged,
            'n_iters': n_iters,
            'Phi_trial': Phi_trial,
            'sigma_v': sigma_v,
            'D_diag': np.array([self.D[0, 0], self.D[1, 1], self.D[2, 2]]),
            'D_eigvals': np.sort(eigvalsh(self.D))[::-1],
            'kappa': self.kappa,
            'p': self.p,
            'Y_eq_D': Y_eq_D,
            'Y_kappa': Y_kappa,
            'dD_norm': norm(dD_mat, 'fro'),
            'dkappa': dkappa,
            'damage_D_active': self._damage_D_flag,
            'damage_kappa_active': self._damage_kappa_flag,
        }
        return sigma_mat, info


# ============================================================================
# 主仿真与绘图
# ============================================================================

def run_simulation():
    """
    运行单材料点的应变驱动仿真，并绘制结果。
    """
    print("=" * 70)
    print("  统一弹塑性损伤本构模型 —— 三维数值验证")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 材料参数
    # ------------------------------------------------------------------
    params = {
        'E': 200e3,
        'nu': 0.3,
        'l_murn': -300e3,
        'm_murn': -620e3,
        'n_murn': -720e3,
        'sigma_y': 250.0,
        'K': 500.0,
        'S': 0.5,
        's': 1.0,
        'Y_th_D': 0.1,
        'S_kappa': 0.2,
        's_kappa': 1.5,
        'Y_th_kappa': 0.01,
        'beta': 100.0,
        'tol': 1e-10,
        'max_iter': 50,
        'fd_h': 1e-8,
        'D_max': 0.99,
    }

    print("\n材料参数：")
    print(f"  弹性: E = {params['E']:.1f} MPa, nu = {params['nu']}")
    print(f"  Murnaghan: l = {params['l_murn']:.0f}, m = {params['m_murn']:.0f}, n = {params['n_murn']:.0f} MPa")
    print(f"  塑性: sigma_y = {params['sigma_y']:.1f} MPa, K = {params['K']:.1f} MPa")
    print(f"  传统损伤: S = {params['S']}, s = {params['s']}, Y_th = {params['Y_th_D']}")
    print(f"  早期损伤: S_kappa = {params['S_kappa']}, s_kappa = {params['s_kappa']}, Y_th,kappa = {params['Y_th_kappa']}")

    # ------------------------------------------------------------------
    # 初始化模型
    # ------------------------------------------------------------------
    model = DamagePlasticityModel3D(params)

    # ------------------------------------------------------------------
    # 定义加载路径
    # ------------------------------------------------------------------
    n_steps_loading = 200
    n_steps_hold = 50
    n_steps_unloading = 100
    eps_max = 0.02

    eps_11_loading = np.linspace(0.0, eps_max, n_steps_loading)
    eps_11_hold = np.full(n_steps_hold, eps_max)
    eps_11_unloading = np.linspace(eps_max, 0.0, n_steps_unloading)
    eps_11_all = np.concatenate([eps_11_loading, eps_11_hold, eps_11_unloading])

    n_total = len(eps_11_all)
    print(f"\n加载路径: 共 {n_total} 步 "
          f"(加载 {n_steps_loading} + 保持 {n_steps_hold} + 卸载 {n_steps_unloading})")

    # ------------------------------------------------------------------
    # 存储历史
    # ------------------------------------------------------------------
    history = {
        'eps_11': np.zeros(n_total + 1),
        'sigma_11': np.zeros(n_total + 1),
        'sigma_22': np.zeros(n_total + 1),
        'sigma_33': np.zeros(n_total + 1),
        'sigma_eq': np.zeros(n_total + 1),
        'D_eigval_1': np.zeros(n_total + 1),
        'D_eigval_2': np.zeros(n_total + 1),
        'D_eigval_3': np.zeros(n_total + 1),
        'kappa': np.zeros(n_total + 1),
        'p': np.zeros(n_total + 1),
        'Phi': np.zeros(n_total + 1),
        'Y_eq_D': np.zeros(n_total + 1),
        'Y_kappa': np.zeros(n_total + 1),
        'yield_flag': np.zeros(n_total + 1, dtype=bool),
        'damage_D_flag': np.zeros(n_total + 1, dtype=bool),
        'damage_kappa_flag': np.zeros(n_total + 1, dtype=bool),
    }

    # ------------------------------------------------------------------
    # 逐步仿真
    # ------------------------------------------------------------------
    yield_step = None
    damage_D_onset_step = None
    damage_kappa_onset_step = None

    print("\n推进仿真...")
    for k, eps_11_target in enumerate(eps_11_all):
        delta_eps_mat = np.zeros((3, 3))
        delta_eps_mat[0, 0] = eps_11_target - history['eps_11'][k]

        sigma_mat, info = model.step(delta_eps_mat)

        idx = k + 1
        history['eps_11'][idx] = model.eps[0, 0]
        history['sigma_11'][idx] = sigma_mat[0, 0]
        history['sigma_22'][idx] = sigma_mat[1, 1]
        history['sigma_33'][idx] = sigma_mat[2, 2]
        history['D_eigval_1'][idx] = info['D_eigvals'][0] if len(info['D_eigvals']) > 0 else 0
        history['D_eigval_2'][idx] = info['D_eigvals'][1] if len(info['D_eigvals']) > 1 else 0
        history['D_eigval_3'][idx] = info['D_eigvals'][2] if len(info['D_eigvals']) > 2 else 0
        history['kappa'][idx] = info['kappa']
        history['p'][idx] = info['p']
        history['Phi'][idx] = info['Phi_trial']
        history['Y_eq_D'][idx] = info['Y_eq_D']
        history['Y_kappa'][idx] = info['Y_kappa']
        history['yield_flag'][idx] = info['yield']
        history['damage_D_flag'][idx] = info['damage_D_active']
        history['damage_kappa_flag'][idx] = info['damage_kappa_active']

        s = sigma_mat - np.eye(3) * np.trace(sigma_mat) / 3.0
        history['sigma_eq'][idx] = np.sqrt(1.5 * np.sum(s**2))

        if yield_step is None and info['yield']:
            yield_step = k
        if damage_D_onset_step is None and info['damage_D_active']:
            damage_D_onset_step = k
        if damage_kappa_onset_step is None and info['damage_kappa_active']:
            damage_kappa_onset_step = k

        if (k + 1) % 50 == 0 or k == 0:
            print(f"  步 {k+1:4d}/{n_total}: eps11 = {history['eps_11'][idx]:.6f}, "
                  f"sig11 = {history['sigma_11'][idx]:.2f} MPa, "
                  f"p = {info['p']:.6f}, D1 = {info['D_eigvals'][0]:.4f}, "
                  f"kappa = {info['kappa']:.4f}, "
                  f"yield: {info['yield']}, Phi = {info['Phi_trial']:.4f}")

    print("\n仿真完成。")

    # ------------------------------------------------------------------
    # 输出关键节点
    # ------------------------------------------------------------------
    print("\n关键节点：")
    if yield_step is not None:
        ys = yield_step + 1
        print(f"  首次屈服:       步 {ys}, eps11 = {history['eps_11'][ys]:.6f}, "
              f"sig11 = {history['sigma_11'][ys]:.2f} MPa")
    if damage_D_onset_step is not None:
        ds = damage_D_onset_step + 1
        print(f"  传统损伤起始:   步 {ds}, eps11 = {history['eps_11'][ds]:.6f}, "
              f"D1 = {history['D_eigval_1'][ds]:.4f}")
    if damage_kappa_onset_step is not None:
        ks = damage_kappa_onset_step + 1
        print(f"  早期损伤起始:   步 {ks}, eps11 = {history['eps_11'][ks]:.6f}, "
              f"kappa = {history['kappa'][ks]:.4f}")

    final_idx = n_total
    print(f"\n最终状态: sig11 = {history['sigma_11'][final_idx]:.2f} MPa, "
          f"D 主值 = [{history['D_eigval_1'][final_idx]:.4f}, "
          f"{history['D_eigval_2'][final_idx]:.4f}, "
          f"{history['D_eigval_3'][final_idx]:.4f}], "
          f"kappa = {history['kappa'][final_idx]:.4f}, "
          f"p = {history['p'][final_idx]:.6f}")

    # ------------------------------------------------------------------
    # 绘图
    # ------------------------------------------------------------------
    print("\n生成图表...")
    eps_range = history['eps_11'][:n_total + 1]
    sigma_range = history['sigma_11'][:n_total + 1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('统一弹塑性损伤本构模型 —— 三维数值验证', fontsize=14, fontweight='bold')

    # --- 图 1: σ₁₁ - ε₁₁ 应力-应变曲线 ---
    ax1 = axes[0, 0]
    ax1.plot(eps_range, sigma_range, 'b-', linewidth=1.5, label=r'$\sigma_{11}$ vs $\varepsilon_{11}$')
    if yield_step is not None:
        ys = yield_step + 1
        ax1.plot(history['eps_11'][ys], history['sigma_11'][ys], 'ro',
                 markersize=8, label='屈服点 (ε={:.4f})'.format(history["eps_11"][ys]))
    if damage_D_onset_step is not None:
        ds = damage_D_onset_step + 1
        ax1.plot(history['eps_11'][ds], history['sigma_11'][ds], 's',
                 color='orange', markersize=8,
                 label='D 损伤起始 (ε={:.4f})'.format(history["eps_11"][ds]))
    if damage_kappa_onset_step is not None:
        ks = damage_kappa_onset_step + 1
        ax1.plot(history['eps_11'][ks], history['sigma_11'][ks], '^',
                 color='green', markersize=8,
                 label=r'$\kappa$ 损伤起始 (ε={:.4f})'.format(history["eps_11"][ks]))

    ax1.set_xlabel(r'总应变 $\varepsilon_{11}$', fontsize=12)
    ax1.set_ylabel(r'应力 $\sigma_{11}$ [MPa]', fontsize=12)
    ax1.set_title(r'轴向应力-应变曲线 $\sigma_{11}\!-\!\varepsilon_{11}$', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linewidth=0.5)

    # --- 图 2: D 主值演化 ---
    ax2 = axes[0, 1]
    ax2.plot(eps_range, history['D_eigval_1'][:n_total + 1], 'r-', linewidth=1.5,
             label=r'$D_1$ (最大主值)')
    ax2.plot(eps_range, history['D_eigval_2'][:n_total + 1], 'g-', linewidth=1.5,
             label=r'$D_2$ (中间主值)')
    ax2.plot(eps_range, history['D_eigval_3'][:n_total + 1], 'b-', linewidth=1.5,
             label=r'$D_3$ (最小主值)')
    if damage_D_onset_step is not None:
        ds = damage_D_onset_step + 1
        ax2.axvline(x=history['eps_11'][ds], color='gray', linestyle='--', alpha=0.5,
                    label='D 损伤阈值')
    ax2.set_xlabel(r'总应变 $\varepsilon_{11}$', fontsize=12)
    ax2.set_ylabel(r'损伤主值 $D_i$', fontsize=12)
    ax2.set_title('传统各向异性损伤 D 主值演化', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.02, None)

    # --- 图 3: κ 演化 ---
    ax3 = axes[1, 0]
    ax3.plot(eps_range, history['kappa'][:n_total + 1], 'm-', linewidth=2,
             label=r'$\kappa$ (早期非线性损伤)')
    if damage_kappa_onset_step is not None:
        ks = damage_kappa_onset_step + 1
        ax3.axvline(x=history['eps_11'][ks], color='gray', linestyle='--', alpha=0.5,
                    label=r'$\kappa$ 损伤阈值')
    ax3.set_xlabel(r'总应变 $\varepsilon_{11}$', fontsize=12)
    ax3.set_ylabel(r'$\kappa$', fontsize=12)
    ax3.set_title(r'早期非线性损伤 $\kappa$ 演化', fontsize=12)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.02, None)

    # --- 图 4: 累积塑性应变与损伤驱动力 ---
    ax4 = axes[1, 1]
    color = 'tab:blue'
    ax4.set_xlabel(r'总应变 $\varepsilon_{11}$', fontsize=12)
    ax4.set_ylabel('累积塑性应变 p', fontsize=12, color=color)
    line1, = ax4.plot(eps_range, history['p'][:n_total + 1], color=color,
                      linewidth=1.5, label='p (累积塑性应变)')
    ax4.tick_params(axis='y', labelcolor=color)

    ax4b = ax4.twinx()
    color2 = 'tab:red'
    ax4b.set_ylabel('损伤驱动力', fontsize=12, color=color2)
    line2, = ax4b.plot(eps_range, history['Y_eq_D'][:n_total + 1], '--',
                       color=color2, linewidth=1.2, alpha=0.8,
                       label=r'$Y_{eq,D}$ (传统损伤)')
    line3, = ax4b.plot(eps_range, history['Y_kappa'][:n_total + 1], '-.',
                       color='tab:orange', linewidth=1.2, alpha=0.8,
                       label=r'$Y_\kappa$ (早期损伤)')
    ax4b.tick_params(axis='y', labelcolor=color2)

    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left', fontsize=9)
    ax4.set_title('累积塑性应变与损伤驱动力', fontsize=12)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('d:/课题-ai/damage_plasticity_3d_results.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存至 damage_plasticity_3d_results.png")
    print("\n" + "=" * 70)
    print("验证完成。")
    print("=" * 70)

    return model, history


# ============================================================================
# 入口
# ============================================================================

if __name__ == '__main__':
    model, history = run_simulation()
