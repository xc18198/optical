import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from optical_channel import OpticalChannel

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2)
        )

    def forward(self, x):
        x_real = x.real
        x_imag = x.imag
        x_combined = torch.cat([x_real, x_imag], dim=-1)
        encoded = self.encoder(x_combined)
        encoded_complex = torch.complex(encoded[..., :encoded.shape[-1]//2], encoded[..., encoded.shape[-1]//2:])
        return encoded_complex

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim * 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_real = x.real
        x_imag = x.imag
        x_combined = torch.cat([x_real, x_imag], dim=-1)
        decoded = self.decoder(x_combined)
        decoded_complex = torch.complex(decoded[..., :decoded.shape[-1]//2], decoded[..., decoded.shape[-1]//2:])
        return decoded_complex

class OpticalChannelLayer(nn.Module):
    def __init__(self, channel_params):
        super(OpticalChannelLayer, self).__init__()
        self.channel = OpticalChannel(**channel_params)

    def forward(self, x):
        signal = x.detach().cpu().numpy()
        f = np.linspace(-50e9, 50e9, signal.shape[1])
        channel_output = self.channel.propagate(signal, f)
        return torch.from_numpy(channel_output).to(x.device).cfloat()

class SemanticOpticalCommunication(nn.Module):
    def __init__(self, input_dim, latent_dim, channel_params):
        super(SemanticOpticalCommunication, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.optical_channel = OpticalChannelLayer(channel_params)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        channel_output = self.optical_channel(encoded)
        decoded = self.decoder(channel_output)
        return decoded

def complex_mse_loss(output, target):
    return torch.mean(torch.abs(output - target)**2)

def train(model, data_loader, epochs, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = complex_mse_loss

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in data_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader):.4f}")

def test_autoencoder(model, data, device):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        output = model(data)
        mse = complex_mse_loss(output, data)
    return mse.item()

def test_direct_transmission(optical_channel, data, device):
    with torch.no_grad():
        data = data.to(device)
        channel_output = optical_channel(data)
        mse = complex_mse_loss(channel_output, data)
    return mse.item()

def create_autoencoder_model(input_dim, latent_dim, channel_params, device):
    model = SemanticOpticalCommunication(input_dim, latent_dim, channel_params)
    model.to(device)
    return model

class AutoencoderModel:
    def __init__(self, input_dim=100, latent_dim=20, channel_params=None, device=None):
        if channel_params is None:
            channel_params = {
                'fiber_length': 50,
                'alpha': 0.2,
                'beta2': -20e-3,
                'beta3': 0.1e-3,
                'gamma': 1.3e-3,
                'D_pmd': 0.1e-3
            }
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.model = create_autoencoder_model(input_dim, latent_dim, channel_params, device)
        self.input_dim = input_dim

    def train(self, data_loader, epochs=50):
        train(self.model, data_loader, epochs, self.device)

    def predict(self, input_signal):
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.from_numpy(input_signal).to(self.device).cfloat()
            input_tensor = input_tensor.view(-1, self.input_dim)
            output = self.model(input_tensor)
            return output.cpu().numpy().flatten()

# 使用示例
if __name__ == "__main__":
    # 设置参数
    input_dim = 100
    latent_dim = 20
    channel_params = {
        'fiber_length': 50,
        'alpha': 0.2,
        'beta2': -20e-3,
        'beta3': 0.1e-3,
        'gamma': 1.3e-3,
        'D_pmd': 0.1e-3
    }
    batch_size = 32
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建AutoencoderModel实例
    autoencoder = AutoencoderModel(input_dim, latent_dim, channel_params, device)

    # 生成模拟数据
    data = torch.complex(torch.rand(1000, input_dim), torch.rand(1000, input_dim))
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    # 训练模型
    autoencoder.train(data_loader, epochs)

    # 测试模型
    test_data = torch.complex(torch.rand(100, input_dim), torch.rand(100, input_dim))
    autoencoder_output = autoencoder.predict(test_data.numpy())

    optical_channel = OpticalChannelLayer(channel_params)
    autoencoder_mse = test_autoencoder(autoencoder.model, test_data, device)
    direct_mse = test_direct_transmission(optical_channel, test_data, device)

    print(f"Autoencoder Mean Squared Error: {autoencoder_mse:.4f}")
    print(f"Direct Transmission Mean Squared Error: {direct_mse:.4f}")
    print(f"Performance Improvement: {(direct_mse - autoencoder_mse) / direct_mse * 100:.2f}%")

    # 计算信道容量
    def channel_capacity(snr):
        return np.log2(1 + snr)

    autoencoder_snr = 1 / autoencoder_mse
    direct_snr = 1 / direct_mse
    autoencoder_capacity = channel_capacity(autoencoder_snr)
    direct_capacity = channel_capacity(direct_snr)

    # 绘制结果
    autoencoder.model.eval()
    with torch.no_grad():
        test_data = test_data.to(device)
        autoencoder_output = autoencoder.model(test_data)
        direct_output = optical_channel(test_data)

    # 创建一个大图,包含所有子图
    plt.figure(figsize=(20, 15))

    # 图1: Autoencoder输入vs输出
    plt.subplot(221)
    plt.scatter(test_data[0].real.cpu().numpy(), test_data[0].imag.cpu().numpy(), alpha=0.5, label='Input')
    plt.scatter(autoencoder_output[0].real.cpu().numpy(), autoencoder_output[0].imag.cpu().numpy(), alpha=0.5, label='Output')
    plt.title("Autoencoder: Input vs Output")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.legend()

    # 图2: 直接传输输入vs输出
    plt.subplot(222)
    plt.scatter(test_data[0].real.cpu().numpy(), test_data[0].imag.cpu().numpy(), alpha=0.5, label='Input')
    plt.scatter(direct_output[0].real.cpu().numpy(), direct_output[0].imag.cpu().numpy(), alpha=0.5, label='Output')
    plt.title("Direct Transmission: Input vs Output")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.legend()

    # 图3: 输出幅度分布
    plt.subplot(223)
    autoencoder_mag = np.abs(autoencoder_output[0].cpu().numpy())
    direct_mag = np.abs(direct_output[0].cpu().numpy())
    input_mag = np.abs(test_data[0].cpu().numpy())
    
    plt.hist(input_mag, bins=50, alpha=0.5, label='Input', density=True)
    plt.hist(autoencoder_mag, bins=50, alpha=0.5, label='Autoencoder', density=True)
    plt.hist(direct_mag, bins=50, alpha=0.5, label='Direct', density=True)
    plt.title("Output Magnitude Distribution")
    plt.xlabel("Magnitude")
    plt.ylabel("Normalized Frequency")
    plt.legend()

    # 图4: 性能比较
    plt.subplot(224)
    methods = ['Autoencoder', 'Direct Transmission']
    mse_values = [autoencoder_mse, direct_mse]
    capacity_values = [autoencoder_capacity, direct_capacity]

    x = np.arange(len(methods))
    width = 0.35

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    rects1 = ax1.bar(x - width/2, mse_values, width, label='MSE', alpha=0.7)
    rects2 = ax2.bar(x + width/2, capacity_values, width, label='Channel Capacity', alpha=0.7, color='orange')

    ax1.set_ylabel('Mean Squared Error')
    ax2.set_ylabel('Channel Capacity (bits/s/Hz)')
    ax1.set_title('Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('combined_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 保存单独的图片
    plt.figure(figsize=(10, 8))
    plt.scatter(test_data[0].real.cpu().numpy(), test_data[0].imag.cpu().numpy(), alpha=0.5, label='Input')
    plt.scatter(autoencoder_output[0].real.cpu().numpy(), autoencoder_output[0].imag.cpu().numpy(), alpha=0.5, label='Output')
    plt.title("Autoencoder: Input vs Output")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.legend()
    plt.savefig('autoencoder_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.scatter(test_data[0].real.cpu().numpy(), test_data[0].imag.cpu().numpy(), alpha=0.5, label='Input')
    plt.scatter(direct_output[0].real.cpu().numpy(), direct_output[0].imag.cpu().numpy(), alpha=0.5, label='Output')
    plt.title("Direct Transmission: Input vs Output")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.legend()
    plt.savefig('direct_transmission_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.hist(input_mag, bins=50, alpha=0.5, label='Input', density=True)
    plt.hist(autoencoder_mag, bins=50, alpha=0.5, label='Autoencoder', density=True)
    plt.hist(direct_mag, bins=50, alpha=0.5, label='Direct', density=True)
    plt.title("Output Magnitude Distribution")
    plt.xlabel("Magnitude")
    plt.ylabel("Normalized Frequency")
    plt.legend()
    plt.savefig('magnitude_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 8))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    rects1 = ax1.bar(x - width/2, mse_values, width, label='MSE', alpha=0.7)
    rects2 = ax2.bar(x + width/2, capacity_values, width, label='Channel Capacity', alpha=0.7, color='orange')
    ax1.set_ylabel('Mean Squared Error')
    ax2.set_ylabel('Channel Capacity (bits/s/Hz)')
    ax1.set_title('Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Results have been saved as separate PNG files and a combined PNG file.")