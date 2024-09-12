import numpy as np
from scipy import fftpack

class OpticalChannel:
    def __init__(self, fiber_length, alpha, beta2, beta3, gamma, D_pmd):
        self.L = fiber_length  # 光纤长度 (km)
        self.alpha = alpha  # 衰减系数 (dB/km)
        self.beta2 = beta2  # 群速度色散 (ps^2/km)
        self.beta3 = beta3  # 三阶色散 (ps^3/km)
        self.gamma = gamma  # 非线性系数 (1/W/km)
        self.D_pmd = D_pmd  # PMD系数 (ps/sqrt(km))

    def ase_noise(self, signal, noise_figure, gain):
        """
        添加放大自发辐射(ASE)噪声
        
        模型: P_ASE = P_signal * NF * (G - 1)
        其中:
        P_ASE: ASE噪声功率
        P_signal: 信号功率
        NF: 噪声系数 (线性单位)
        G: 增益 (线性单位)
        
        噪声被建模为复高斯噪声: n(t) = n_I(t) + j*n_Q(t)
        其中 n_I(t) 和 n_Q(t) 是独立的高斯随机过程,方差为 P_ASE/2
        """
        P_signal = np.mean(np.abs(signal)**2)
        P_ase = P_signal * (10**(noise_figure/10) * (gain - 1))
        noise = np.random.normal(0, np.sqrt(P_ase/2), signal.shape) + \
                1j * np.random.normal(0, np.sqrt(P_ase/2), signal.shape)
        return signal + noise

    def chromatic_dispersion(self, signal, f, z):
        """
        模拟色度色散
        
        传递函数: H(ω,z) = exp(jφ(ω,z))
        相位: φ(ω,z) = -0.5β2ω²z - (1/6)β3ω³z
        其中:
        ω: 角频率
        z: 传输距离
        β2: 群速度色散参数
        β3: 三阶色散参数
        """
        omega = 2 * np.pi * f
        phi = -0.5 * self.beta2 * omega**2 * z - (1/6) * self.beta3 * omega**3 * z
        return signal * np.exp(1j * phi.astype(np.float64)).astype(np.complex128)

    def pmd(self, signal, f, z):
        """
        模拟偏振模色散(PMD)
        
        传递函数: H_PMD(ω) = exp(-jωΔτ/2)
        其中:
        Δτ: 差分群延迟 (DGD)
        DGD模型: Δτ = D_PMD * sqrt(z)
        D_PMD: PMD参数
        z: 传输距离
        """
        omega = 2 * np.pi * f
        delta_tau = self.D_pmd * np.sqrt(z)
        H_pmd = np.exp(-1j * omega * delta_tau / 2).astype(np.complex128)
        return signal * H_pmd

    def kerr_effect(self, signal, z):
        """
        模拟克尔效应 (包括SPM和XPM)
        
        非线性相移: φ_NL = γ|E|²z
        其中:
        γ: 非线性系数
        |E|²: 信号功率
        z: 传输距离
        
        输出信号: E_out = E_in * exp(jφ_NL)
        """
        P = np.abs(signal)**2
        phi_nl = self.gamma * P * z
        return signal * np.exp(1j * phi_nl.astype(np.float64)).astype(np.complex128)

    def fwm(self, signal, z):
        """
        模拟四波混频(FWM)
        
        简化FWM模型: E_FWM = jγz|E|²E
        其中:
        γ: 非线性系数
        |E|²: 信号功率
        E: 信号场强
        z: 传输距离
        
        注: 这是一个简化模型,实际FWM效应更复杂
        """
        P = np.abs(signal)**2
        E_fwm = 1j * self.gamma * z * P * signal
        return signal + E_fwm * 1e-3  # 减小FWM效应的强度

    def propagate(self, signal, f):
        """通过整个光纤信道传播信号"""
        z = 0
        step = 0.1  # km
        while z < self.L:
            signal = self.chromatic_dispersion(signal, f, step)
            signal = self.pmd(signal, f, step)
            signal = self.kerr_effect(signal, step)
            signal = self.fwm(signal, step)
            # 添加功率衰减
            signal *= np.exp(-self.alpha * step / (20 * np.log10(np.e)))
            z += step
        
        # 添加ASE噪声 (假设有一个放大器)
        signal = self.ase_noise(signal, noise_figure=5, gain=20)
        
        return signal

# 辅助函数
def generate_qpsk_signal(num_symbols, samples_per_symbol):
    """生成QPSK信号"""
    symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], num_symbols)
    return np.repeat(symbols, samples_per_symbol)

def add_awgn(signal, snr_db):
    """添加加性高斯白噪声"""
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / (10**(snr_db / 10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    return signal + noise
