# -*- coding: utf-8 -*-
"""
光谱数据预处理的最小工具模块（原样对应简历里"光谱预处理"的说法）。

只依赖 numpy，用 Anaconda 自带环境即可运行，无需额外装包。
"""
import os
import numpy as np

def load_spectrum(path):
    """
    智能读入两列数据：波长/波数 和 强度。
    自动检测第一行是表头还是数据，如果是表头则自动跳过。
    """
    skip_rows = 0  # 默认不跳过
    
    # 1. 自动检测第一行
    if os.path.exists(path):
        try:
            # 尝试用 utf-8 或 gbk 读取第一行（兼容中文表头）
            with open(path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
        except UnicodeError:
            with open(path, 'r', encoding='gbk') as f:
                first_line = f.readline().strip()
                
        if first_line:
            # 兼容逗号分隔或空格/tab分隔
            if ',' in first_line:
                parts = first_line.split(',')
            else:
                parts = first_line.split()
                
            try:
                # 尝试把第一行的第一个元素转成浮点数
                float(parts[0].strip())
                # 如果成功，说明第一行是纯数字（数据），skip_rows 保持为 0
            except ValueError:
                # 如果报错，说明第一行包含字母（表头），需要跳过
                skip_rows = 1

    # 2. 带 skiprows 参数读取
    x, y = np.loadtxt(path, unpack=True, delimiter=',', usecols=(0, 1), skiprows=skip_rows)
    return x, y

def moving_average_smooth(y, window=3):
    """
    滑动平均平滑，用于消除光谱高频毛刺。
    window 为窗口大小（奇数），两端用原始值补边。
    """
    window = int(window)
    if window <= 0:
        raise ValueError("window 必须为正整数")
    if window % 2 == 0:
        window += 1  # 强制奇数，保证对称窗口
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("仅支持一维光谱")
    n = y.size
    if n < window:
        return y.copy()
    half = window // 2
    out = y.copy()
    for i in range(half, n - half):
        out[i] = y[i - half:i + half + 1].mean()
    return out


def min_max_normalize(y, vmin=0.0, vmax=1.0):
    """
    按最大值最小值做线性归一化到 [vmin, vmax]。
    常用于消除强度 / 波长差异，便于不同样本对齐比较。
    """
    y = np.asarray(y, dtype=float)
    lo, hi = float(y.min()), float(y.max())
    if hi - lo < 1e-12:
        return np.full_like(y, vmin)
    t = (y - lo) / (hi - lo)
    return t * (vmax - vmin) + vmin


def find_peak_indices(y, min_prominence=0.05):
    """
    极简寻峰：返回局部极大值位置的索引（用于峰位对齐/特征提取）。
    仅用一阶差分判断局部极大，要求强度相对幅值超过 min_prominence。
    """
    y = np.asarray(y, dtype=float)
    amp = float(np.ptp(y)) if y.size else 0.0
    thr = min_prominence * (amp if amp > 0 else 1.0)
    peaks = []
    for i in range(1, y.size - 1):
        if y[i - 1] < y[i] > y[i + 1] and y[i] - y.min() >= thr:
            peaks.append(i)
    return peaks


def preprocess_pipeline(x, y, window=5):
    """把 平滑 + 基线校正 + 归一化 串成一个流水线，返回 (x, 预处理后的y)。"""
    # 1. 平滑去噪
    y_s = moving_average_smooth(y, window)
    
    # 2. 基线校正（核心新增！放在平滑后，归一化前）
    y_b = baseline_correction(y_s)
    
    # 3. 最值归一化到 [0, 1]
    y_n = min_max_normalize(y_b)
    
    return x, y_n

def area_normalize(y):
    """
    面积归一化：将光谱曲线下面积归一化为 1。
    常用于消除样品浓度差异，便于不同样本对齐比较。
    """
    y = np.asarray(y, dtype=float)

    # 用梯形积分求总面积；取绝对值避免负峰抵消
    area = np.trapezoid(y)

    # 防止全零数组除零
    if abs(area) < 1e-12:
        return y.copy()
    return y / area


def baseline_correction(y, deg=5, tol=1e-3, max_iter=50):
    """
    迭代多项式基线校正。
    原理：反复拟合光谱，对高于基线的部分（特征峰）赋予极小权重，最终拟合出一条平滑的背景基线。
    """
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y))
    
    # 初始化权重，假设所有点都是基线
    w = np.ones_like(y)
    
    for _ in range(max_iter):
        # 加权多项式拟合
        z = np.polyfit(x, y, deg, w=w)
        baseline = np.polyval(z, x)
        
        # 更新权重：高于基线的点（特征峰）权重设为极小值，忽略它们，只拟合低处的背景
        w_new = np.where(y > baseline, tol, 1.0)
        
        # 如果权重收敛，提前退出
        if np.allclose(w, w_new):
            break
        w = w_new
        
    # 返回减去基线后的光谱
    y_corrected = y - baseline
    # 校正后可能出现负值，把它剪裁到 0（可选）
    y_corrected = np.clip(y_corrected, 0, None)
    return y_corrected