import numpy as np
import matplotlib.pyplot as plt
from optical_channel import OpticalChannel, generate_qpsk_signal, add_awgn

# 设置参数
num_symbols = 1000
samples_per_symbol = 32  # 增加采样率
total_samples = num_symbols * samples_per_symbol

# 生成QPSK信号
signal = generate_qpsk_signal(num_symbols, samples_per_symbol)

# 添加AWGN
snr_db = 20
noisy_signal = add_awgn(signal, snr_db)

# 创建光信道
channel = OpticalChannel(
    fiber_length=50,    # 减少光纤长度
    alpha=0.2,          # dB/km
    beta2=-20e-3,       # ps^2/km (减小色散系数)
    beta3=0.1e-3,       # ps^3/km (减小三阶色散系数)
    gamma=1.3e-3,       # 1/W/km (减小非线性系数)
    D_pmd=0.1e-3        # ps/sqrt(km) (减小PMD系数)
)

# 生成频率轴
f = np.linspace(-50e9, 50e9, total_samples)  # 减小频率范围

# 通过信道传播信号
output_signal = channel.propagate(noisy_signal, f)

# 绘制结果
plt.figure(figsize=(15, 10))

plt.subplot(231)
plt.scatter(signal.real, signal.imag, alpha=0.5)
plt.title("Original QPSK Signal")
plt.xlabel("In-phase")
plt.ylabel("Quadrature")

plt.subplot(232)
plt.scatter(noisy_signal.real, noisy_signal.imag, alpha=0.5)
plt.title(f"QPSK Signal with AWGN (SNR = {snr_db} dB)")
plt.xlabel("In-phase")
plt.ylabel("Quadrature")

plt.subplot(233)
plt.scatter(output_signal.real, output_signal.imag, alpha=0.5)
plt.title("Signal after Optical Channel")
plt.xlabel("In-phase")
plt.ylabel("Quadrature")

plt.subplot(234)
plt.plot(f/1e9, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(signal)))**2))
plt.title("Original Signal Spectrum")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Power (dB)")

plt.subplot(235)
plt.plot(f/1e9, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(noisy_signal)))**2))
plt.title("Noisy Signal Spectrum")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Power (dB)")

plt.subplot(236)
plt.plot(f/1e9, 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(output_signal)))**2))
plt.title("Output Signal Spectrum")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Power (dB)")

plt.tight_layout()

# 保存图像
plt.savefig('optical_channel_simulation.png', dpi=300, bbox_inches='tight')

# 显示图像（可选）
plt.show()