import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.optim as optim
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import yaml
from models import DNN1_CNN1, StudentModel, Generator, DNN, CNN, RNN, LSTM, Transformer, GRU, TeacherModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
config = yaml.safe_load(open('extraction/config.yaml'))
benign_path = config['benign_path']
malicious_path = config['malicious_path']


def get_student_model(choice, input_dim, num_classes):
    student = None
    if choice == 'CNN':
        student = CNN(num_classes=num_classes).to(device)
    elif choice == 'DNN':
        student = DNN(input_dim=input_dim, num_classes=num_classes).to(device)
    elif choice == 'LSTM':
        student = LSTM(num_classes=num_classes).to(device)
    elif choice == 'RNN':
        student = RNN(num_classes=num_classes).to(device)
    elif choice == 'GRU':
        student = GRU(num_classes=num_classes).to(device)
    elif choice == 'Transformer':
        student = Transformer(input_dim=input_dim, num_classes=num_classes).to(device)
    elif choice == 'teacher':
        student = TeacherModel(num_classes=num_classes).to(device)
    elif choice == 'student':
        student = StudentModel(input_dim=input_dim, num_classes=num_classes).to(device)
    elif choice == 'DNN1_CNN1':
        student = DNN1_CNN1(input_dim=input_dim, num_classes=num_classes).to(device)
    return student


class DFME:
    """数据无关模型提取实现"""

    def __init__(self, teacher, student_choice, latent_dim=100, input_dim=67, num_classes=2,
                 n_G=1, n_S=5, m=1, epsilon=0.001, lr_G=0.0005, lr_S=0.01):
        # 将教师模型移动到指定设备
        self.teacher = teacher.to(device)
        self.teacher.eval()  # 教师模型为黑盒，不可训练

        # 初始化学生模型和生成器，并移动到指定设备
        self.student = get_student_model(student_choice, input_dim, num_classes)
        self.generator = Generator(latent_dim=latent_dim, output_dim=input_dim).to(device)

        # 优化器
        self.optimizer_S = optim.Adam(
            self.student.parameters(), lr=lr_S, weight_decay=1e-4
        )
        self.optimizer_G = optim.Adam(
            self.generator.parameters(), lr=lr_G, betas=(0.5, 0.999)
        )

        # 训练参数
        self.n_G = n_G  # 生成器训练迭代次数
        self.n_S = n_S  # 学生模型训练迭代次数
        self.m = m  # 梯度近似方向数
        self.epsilon = epsilon  # 前向差分步长
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.num_classes = num_classes

        # 损失函数
        self.loss_fn = nn.L1Loss()  # L1损失

        # 记录训练过程
        self.student_acc_history = []
        self.student_loss_history = []
        self.query_count = 0

    def recover_logits(self, probabilities):
        """从概率分布恢复logits"""
        # 添加一个小值防止log(0)
        log_probs = torch.log(probabilities + 1e-10)
        # 计算logits的均值（用于校正）
        mean_logits = torch.mean(log_probs, dim=1, keepdim=True)
        # 恢复近似的logits（减去均值）
        approx_logits = log_probs - mean_logits
        return approx_logits

    def forward_difference(self, z, loss_func):
        """使用前向差分法近似梯度"""
        z = z.detach().requires_grad_(True)
        batch_size = z.size(0)

        # 生成基础样本
        x_base = self.generator(z)

        # 获取基础损失
        with torch.no_grad():
            # 确保教师模型输入在正确设备上
            teacher_probs = torch.softmax(self.teacher(x_base), dim=1)
            teacher_logits = self.recover_logits(teacher_probs)
            student_logits = self.student(x_base)
            base_loss = loss_func(student_logits, teacher_logits)

        # 初始化梯度近似
        grad_approx = torch.zeros_like(z)

        # 在多个随机方向上计算
        for _ in range(self.m):
            # 随机方向 - 确保在正确设备上
            u = torch.randn_like(z, device=device)
            u_norm = u / torch.norm(u, dim=1, keepdim=True)

            # 扰动样本
            z_perturbed = z + self.epsilon * u_norm
            x_perturbed = self.generator(z_perturbed)

            # 获取扰动损失
            with torch.no_grad():
                teacher_probs_p = torch.softmax(self.teacher(x_perturbed), dim=1)
                teacher_logits_p = self.recover_logits(teacher_probs_p)
                student_logits_p = self.student(x_perturbed)
                perturbed_loss = loss_func(student_logits_p, teacher_logits_p)

            # 计算方向导数
            directional_derivative = (perturbed_loss - base_loss) / self.epsilon

            # 更新梯度近似
            grad_approx += directional_derivative.view(-1, 1) * u_norm

        # 平均梯度近似
        grad_approx /= self.m
        self.query_count += 2 * self.m * batch_size  # 每个样本每个方向2次查询

        return grad_approx

    def train_generator_step(self, batch_size=32):
        """训练生成器"""
        self.generator.train()
        self.student.eval()

        # 生成随机噪声 - 确保在正确设备上
        z = torch.randn(batch_size, self.latent_dim, device=device, requires_grad=True)

        # 定义生成器损失函数（最大化学生和教师之间的差异）
        def loss_func(student_logits, teacher_logits):
            return self.loss_fn(student_logits, teacher_logits)

        # 近似梯度
        grad_approx = self.forward_difference(z, loss_func)

        # 更新生成器参数
        self.optimizer_G.zero_grad()
        # 手动设置梯度并更新
        z.backward(grad_approx)
        self.optimizer_G.step()

    def train_student_step(self, batch_size=32):
        """训练学生模型"""
        self.student.train()
        self.generator.eval()

        total_loss = 0.0

        for _ in range(self.n_S):
            # 生成样本 - 确保在正确设备上
            z = torch.randn(batch_size, self.latent_dim, device=device)
            with torch.no_grad():
                x = self.generator(z)

            # 获取教师预测（黑盒）
            with torch.no_grad():
                teacher_probs = torch.softmax(self.teacher(x), dim=1)
                teacher_logits = self.recover_logits(teacher_probs)
                self.query_count += x.size(0)  # 记录查询次数

            # 学生预测
            student_logits = self.student(x)

            # 计算损失
            loss = self.loss_fn(student_logits, teacher_logits)
            total_loss += loss.item()

            # 更新学生模型
            self.optimizer_S.zero_grad()
            loss.backward()
            self.optimizer_S.step()

        return total_loss / self.n_S

    def evaluate_student(self, test_loader):
        """评估学生模型性能"""
        self.student.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in test_loader:
                # 将测试数据移到正确设备
                data = data.to(device)
                labels = labels.to(device)

                outputs = self.student(data)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        self.student_acc_history.append(accuracy)
        return accuracy

    def train(self, epochs, test_loader, query_budget=20000, batch_size=32):
        """训练过程"""
        logger.info(f"开始训练，查询预算为 {query_budget}")
        logger.info(f"Using device: {device}")  # 打印使用的设备

        for epoch in range(epochs):
            # 检查查询预算
            if self.query_count >= query_budget:
                logger.info(f"在epoch {epoch} 中消耗完所有查询预算，停止训练")
                break

            # 训练生成器
            for _ in range(self.n_G):
                self.train_generator_step(batch_size)

            # 训练学生模型
            student_loss = self.train_student_step(batch_size)
            self.student_loss_history.append(student_loss)

            # 评估学生模型
            accuracy = self.evaluate_student(test_loader)

            logger.debug(
                f"Epoch [{epoch + 1}/{epochs}] - Loss: {student_loss:.4f} - Acc: {accuracy:.2f}% - Queries: {self.query_count}/{query_budget}")

            # 动态调整学习率
            if epoch % 10 == 0:
                for param_group in self.optimizer_S.param_groups:
                    param_group['lr'] *= 0.9
                for param_group in self.optimizer_G.param_groups:
                    param_group['lr'] *= 0.9

        logger.success("训练完成！")

    def save_models(self, student_path, generator_path):
        """保存模型"""
        sp = os.path.join(student_path, "student_model.pt")
        gp = os.path.join(generator_path, "generator_model.pt")
        torch.save(self.student.state_dict(), sp)
        torch.save(self.generator.state_dict(), gp)
        logger.success(f"模型已保存至 {sp} {gp}\n")

    def plot_results(self, save_path=None):
        """绘制训练结果"""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.student_loss_history)
        plt.title("Student Model Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("L1 Loss")
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 2)
        plt.plot(self.student_acc_history)
        plt.title("Student Model Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()


class EME:
    """简单模型提取"""

    def __init__(self, teacher, student_choice, latent_dim=100, input_dim=67, num_classes=2,
                 n_S=5, lr_S=0.01):
        self.teacher = teacher.to(device)
        self.teacher.eval()

        self.student = get_student_model(student_choice, input_dim=input_dim, num_classes=num_classes).to(device)

        # 训练参数
        self.n_S = n_S  # 学生模型训练迭代次数
        self.lr_S = lr_S
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.loss_fn = nn.L1Loss()  # L1损失

    def synthesis_dataset(self, synthesis_data_path, query_budget):
        df1 = pd.read_csv(benign_path)
        df2 = pd.read_csv(malicious_path)
        data = pd.concat([df1, df2])

        # 提取特征并转换为PyTorch张量
        X = data.drop('label', axis=1).values
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        feature_columns = data.drop('label', axis=1).columns
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)  # 转换为张量

        # 设置模型为评估模式
        self.teacher.eval()

        # 使用模型进行预测（不计算梯度以提高效率）
        with torch.no_grad():
            outputs = self.teacher(X_tensor)  # 使用forward方法获取原始输出
            probabilities = torch.softmax(outputs, dim=1)  # 应用softmax获取概率
            y_pred = probabilities.argmax(dim=1).cpu().numpy()  # 获取最大概率对应的标签

        # 将特征和预测标签组合成DataFrame
        synthesis_df = pd.DataFrame(X, columns=feature_columns)
        synthesis_df['label'] = y_pred
        
        # 计算抽样比例
        sample_ratio = query_budget / len(synthesis_df)
        synthesis_df = synthesis_df.groupby('label',group_keys=False).apply(
            lambda x: x.sample(frac=sample_ratio))

        # 保存合成数据集为CSV文件
        synthesis_df.to_csv(os.path.join(synthesis_data_path, 'synthesis.csv'), index=False)

    def preprocess(self, synthesis_data_path, test_size=0.3):
        """创建合成数据加载器"""
        data = pd.read_csv(os.path.join(synthesis_data_path, 'synthesis.csv'))

        # 分离特征和目标
        X = data.drop('label', axis=1).values
        y = data['label'].values

        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, stratify=y
        )

        # 转换为PyTorch张量
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True
        )

        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False
        )
        return train_loader, test_loader

    def evaluate_student(self, test_loader):
        """评估学生模型性能"""
        self.student.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in test_loader:
                # 将测试数据移到正确设备
                data = data.to(device)
                labels = labels.to(device)

                outputs = self.student(data)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy


    def train_student(self, train_loader, test_loader, student_path):
        logger.info("开始训练学生模型")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.student.parameters(), lr=self.lr_S, weight_decay=1e-4)

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        best_acc = 0.0

        for epoch in range(self.n_S):
            self.student.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = self.student(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_acc = 100 * correct / total
            train_loss /= len(train_loader)

            # 评估学生模型
            self.student.eval()
            test_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self.student(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            test_acc = 100 * correct / total
            test_loss /= len(test_loader)

            logger.debug(f"Student Epoch [{epoch + 1}/{self.n_S}] - "
                         f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                         f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

            # 更新学习率
            scheduler.step(test_acc)

            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.student.state_dict(), os.path.join(student_path, "easy_student_model.pt"))
        os.remove('data/synthesis.csv')


def pretrain_teacher_model(train_loader, test_loader, input_dim=67, num_classes=2, epochs=3):
    """预训练所有模型"""

    def pretrain(method):
        logger.info(f'开始训练 {method} 模型')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        best_acc = 0.0

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_acc = 100 * correct / total
            train_loss /= len(train_loader)

            # 评估教师模型
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            test_acc = 100 * correct / total
            test_loss /= len(test_loader)

            logger.debug(f"{method} Epoch [{epoch + 1}/{epochs}] - "
                         f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                         f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

            # 更新学习率
            scheduler.step(test_acc)

            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f"extraction/model/{method}.pt")

        logger.success(f"{method} model trained with best accuracy: {best_acc:.2f}%")

    logger.info("预训练模型")
    model = DNN(input_dim=input_dim, num_classes=num_classes).to(device)
    pretrain('DNN')
    model = CNN(num_classes=num_classes).to(device)
    pretrain('CNN')
    model = RNN(num_classes=num_classes).to(device)
    pretrain('RNN')
    model = LSTM(num_classes=num_classes).to(device)
    pretrain('LSTM')
    model = Transformer(input_dim=input_dim, num_classes=num_classes).to(device)
    pretrain('Transformer')
    model = GRU(num_classes=num_classes).to(device)
    pretrain('GRU')
    model = DNN1_CNN1(input_dim=input_dim, num_classes=num_classes).to(device)
    pretrain('DNN1_CNN1')
    model = TeacherModel(num_classes=num_classes).to(device)
    pretrain('teacher_model')