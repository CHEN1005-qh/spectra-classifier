# -*- coding: utf-8 -*-
"""
针对 spec_utils.py 的 Pytest 测试用例。

运行方式（在本目录下执行）：
    pytest test_spec_utils.py -v
"""
import os
import numpy as np
import pytest

import spec_utils as su


@pytest.fixture
def synthetic_spectrum():
    """构造一条带噪声的模拟光谱：约 500 个采样点，含一个主峰和基线起伏。"""
    rng = np.random.default_rng(42)
    x = np.arange(500)
    base = 10 + 0.02 * x                      # 缓慢上升的基线
    peak = 20 * np.exp(-((x - 250) ** 2) / (2 * 30 ** 2))  # 高斯主峰
    noise = 0.5 * rng.standard_normal(x.size)              # 随机噪声
    return x, base + peak + noise


def test_moving_average_smooth_shape_preserved(synthetic_spectrum):
    x, y = synthetic_spectrum
    out = su.moving_average_smooth(y, window=5)
    assert out.shape == y.shape
    assert np.isfinite(out).all()


def test_moving_average_smooth_reduces_noise(synthetic_spectrum):
    x, y = synthetic_spectrum
    raw_std = np.diff(y).std()
    smooth_std = np.diff(su.moving_average_smooth(y, window=9)).std()
    # 平滑后相邻差分的波动应明显小于原始噪声（用宽松阈值避免脆弱）
    assert smooth_std < raw_std


def test_moving_average_rejects_bad_window(synthetic_spectrum):
    x, y = synthetic_spectrum
    with pytest.raises(ValueError):
        su.moving_average_smooth(y, window=0)


def test_min_max_normalize_range(synthetic_spectrum):
    x, y = synthetic_spectrum
    out = su.min_max_normalize(y, 0.0, 1.0)
    assert out.min() >= 0.0 - 1e-12
    assert out.max() <= 1.0 + 1e-12
    assert out.max() == pytest.approx(1.0, abs=1e-9)
    assert out.min() == pytest.approx(0.0, abs=1e-9)


def test_min_max_normalize_constant_input():
    # 全零数组不应除零崩溃
    out = su.min_max_normalize(np.zeros(10))
    assert np.allclose(out, 0.0)


def test_find_peaks_detects_known_peak(synthetic_spectrum):
    x, y = synthetic_spectrum
    y_s = su.moving_average_smooth(y, window=5)
    peaks = su.find_peak_indices(y_s, min_prominence=0.5)
    # 主峰中心在 250 附近，峰值位置应落在其附近窗口内
    assert any(abs(idx - 250) <= 5 for idx in peaks[0:20])


def test_preprocess_pipeline_output(synthetic_spectrum):
    x, y = synthetic_spectrum
    px, py = su.preprocess_pipeline(x, y, window=5)
    assert px.tolist() == x.tolist()
    assert py.shape == y.shape
    assert py.min() >= 0.0 - 1e-12
    assert py.max() <= 1.0 + 1e-12


def test_real_spectrum_pipeline():
    """
    真实加载数据文件后跑一遍 preprocess_pipeline，断言归一化到 [0,1]。
    注意：请提前准备真实的数据文件，并修改下方文件名。
    """
    # 1. 指定你的真实数据文件路径（建议放在与本测试文件同目录下）
    data_file = "PET.csv"  # 请替换为真实文件名，例如 "sample_spectrum.txt"

    # 2. 容错处理：如果文件不存在，跳过该测试（避免CI或他人运行时报错）
    if not os.path.exists(data_file):
        pytest.skip(f"未找到真实数据文件 '{data_file}'，跳过此真实加载测试。请准备好文件后重试。")

    # 3. 真实加载数据
    x, y = su.load_spectrum(data_file)

    # 4. 跑预处理流水线
    px, py = su.preprocess_pipeline(x, y)

    # 5. 断言归一化到 [0,1]（按图片要求）
    assert 0.0 <= py.min() and py.max() <= 1.0

def test_area_normalize_total_area_is_one():
    y = np.array([0., 1., 2., 1., 0.])
    yn = su.area_normalize(y)

    assert yn.shape == y.shape
    assert np.isclose(np.trapezoid(yn), 1.0)
    assert not np.isnan(yn).any()

def test_area_normalize_all_zero_no_crash():
    y = np.zeros(5)
    yn = su.area_normalize(y)

    assert np.all(yn == 0)
    assert not np.isnan(yn).any()


def test_baseline_correction_flattens_slope():
    # 构造一条带有斜率基线和高斯峰的光谱
    x = np.linspace(0, 10, 100)
    # 基线：从 2 降到 0.5 (倾斜)
    baseline = 2.0 - 0.15 * x
    # 特征峰：在 x=5 附近加一个高斯峰
    peak = 1.0 * np.exp(-((x - 5)**2) / (2 * 0.5**2))
    y = baseline + peak
    
    # 跑基线校正
    y_corrected = su.baseline_correction(y)
    
    # 断言：原本两端基线差别很大，校正后两端基线应该接近 0
    # 注意：这里只是粗略判断，因为多项式拟合未必完美百分百归零
    assert y_corrected[0] < 0.2
    assert y_corrected[-1] < 0.2
    # 修改这里：允许峰位有 ±2 个点的偏移（因为基线校正会导致倾斜基线上的峰顶产生微小位移）
    assert abs(np.argmax(y_corrected) - np.argmax(y)) <= 2