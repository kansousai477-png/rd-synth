import torch
import torch.nn as nn


class Generator(nn.Module):
    """生成器"""

    def __init__(self, latent_dim=100, output_dim=67):
        super(Generator, self).__init__()
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class StudentModel(nn.Module):
    """学生模型"""

    def __init__(self, input_dim=67, num_classes=2, hidden_dims=[64, 32, 8]):
        super(StudentModel, self).__init__()

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            prev_dim = dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class TeacherModel(nn.Module):
    """教师模型"""

    def __init__(self, input_dim=67, num_classes=2, hidden_dims=[64, 32, 8]):
        super(TeacherModel, self).__init__()

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            prev_dim = dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    

class DNN1_CNN1(nn.Module):
    def __init__(self, input_dim=67, num_classes=2):
        super().__init__()
        # 初始全连接层
        self.dnn = nn.Sequential(
            nn.Linear(input_dim, 64),  # 直接映射到64维
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        # 一层CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),  # 卷积层批标准化（参数为通道数）
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 池化层
        )
        # 最后的分类器
        self.classifier = nn.Sequential(
            nn.Linear(32 * 31, 64),  
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.dnn(x)  # 输出: [batch_size, 64]
        x = x.unsqueeze(1)  # 输出: [batch_size, 1, 64]
        x = self.cnn(x)  # 输出: [batch_size, 32, 31]
        x = x.view(x.size(0), -1)  # 输出: [batch_size, 32*31=992]
        return self.classifier(x)  # 输出: [batch_size, num_classes]
    


class DNN(nn.Module):
    def __init__(self, input_dim=67, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),  # 输入层
            nn.ReLU(),
            nn.Linear(64, 32),  # 第一个隐藏层
            nn.ReLU(),
            nn.Linear(32, 8),  # 第二个隐藏层
            nn.ReLU(),
            nn.Linear(8, num_classes),  # 输出层
        )

    def forward(self, x):
        return self.network(x)


class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 3),  # [batch, 32, 66]
            nn.ReLU(),
            nn.MaxPool1d(2),  # [batch, 32, 33]
            nn.Conv1d(32, 64, 3),  # [batch, 64, 31]
            nn.ReLU(),
            nn.MaxPool1d(2)  # [batch, 64, 15]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 15, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度 [batch, 1, 68]
        x = self.features(x)
        return self.classifier(x)


class RNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,  # 每个时间步的特征维度（将每个特征视为时间步）
            hidden_size=16,  # 隐藏状态维度
            num_layers=2,  # 堆叠两层RNN
            batch_first=True  # 输入格式为[batch,序列长度,特征维度]
        )
        self.fc = nn.Sequential(  # 全连接层
            nn.Linear(16, 8),  # 将RNN输出降维
            nn.ReLU(),
            nn.Linear(8, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)  # 形状调整：[batch,68] → [batch,68,1]（添加特征维度）
        output, _ = self.rnn(x)  # RNN前向传播
        last_output = output[:, -1, :]  # 取最后一个时间步的输出
        return self.fc(last_output)  # 通过全连接层分类


class LSTM(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,  # 每个时间步的特征维度
            hidden_size=16,  # 隐藏状态维度
            num_layers=1,  # 堆叠两层LSTM
            batch_first=True,  # 输入格式设置
            bidirectional=False  # 不使用双向LSTM
        )
        self.fc = nn.Sequential(
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)  # 形状调整：[batch,68] → [batch,68,1]
        output, (h_n, c_n) = self.lstm(x)  # LSTM返回输出和隐藏状态
        last_output = output[:, -1, :]  # 取最后一个时间步的输出
        return self.fc(last_output)


class GRU(nn.Module):
    def __init__(self, input_dim=67, num_classes=2):
        super().__init__()
        # GRU核心层，使用适合时序数据的参数设置
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=128,  # 更大的隐藏层维度以捕捉时序特征
            num_layers=1,  # 单层GRU更适合简单时序模式
            batch_first=True
        )
        # 分类头，采用更简洁的结构
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),  # 使用Tanh激活更适合GRU输出特性
            nn.Dropout(0.3),  # 更强的dropout防止过拟合
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 处理2D输入（batch_size, input_dim）为3D时序输入
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 增加序列长度维度
        # GRU处理
        gru_out, _ = self.gru(x)
        # 取最后一个时间步的输出
        final_out = gru_out[:, -1, :]
        # 分类输出
        return self.classifier(final_out)


class Transformer(nn.Module):
    def __init__(self, input_dim=67, num_classes=2):
        super().__init__()
        # 简化嵌入层
        self.embedding = nn.Linear(1, 16)
        # 位置编码保留但不增加复杂度
        self.pos_encoder = nn.Parameter(torch.randn(input_dim, 16))
        # 简化Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=4,  # 注意力头数量
            dim_feedforward=128,  # 前馈网络维度
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)  # 减少层数
        # 简化分类器（单层线性层）
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)  # [batch,68] → [batch,68,1]
        x = self.embedding(x)  # 线性变换到64维 → [batch,68,64]
        x = x + self.pos_encoder  # 添加位置编码（广播机制）
        x = x.permute(1, 0, 2)  # 调整为Transformer需要的格式：[序列长度, batch, 特征维度]
        output = self.transformer(x)  # 通过Transformer编码器
        return self.classifier(output[0])  # 取第一个位置的特征进行分类


class ValidityDiscriminator(nn.Module):
    """生成器真实性验证器"""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(67, 64),  # 输入层
            nn.ReLU(),
            nn.Linear(64, 32),  # 第一个隐藏层
            nn.ReLU(),
            nn.Linear(32, 8),  # 第二个隐藏层
            nn.ReLU(),
            nn.Linear(8, 1),  # 输出层
            nn.Sigmoid()  # 二分类使用Sigmoid激活
        )

    def forward(self, x):
        return self.network(x)
