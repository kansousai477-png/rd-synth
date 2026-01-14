import os
from sys import stdout

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from data_loader import load_and_preprocess_data, create_data_loaders
from models import DNN1_CNN1, TeacherModel, CNN, DNN, LSTM, RNN, Transformer, GRU, StudentModel
from train import DFME

# 设置日志
logger.remove(0)
logger.add(stdout, colorize=True, level='DEBUG',
           format='<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> <red>|</red> <level>{level: <8}</level> ''<red>|</red> <level>{message}</level>')

# 设置随机种子
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# np.random.seed(42)

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
config = yaml.safe_load(open('extraction/config.yaml'))

input_dim = config['input_dim']  # 特征维度
num_classes = config['num_classes']  # 类别数
latent_dim = config['latent_dim']  # 生成器输入维度
query_budget = config['query_budget']  # 查询预算
batch_size = config['batch_size']  # 批次大小
student_path = config['student_path']  # 学生模型保存路径
generator_path = config['generator_path']  # 生成器保存路径
use_pretrain = config['use_pretrain']  # 教师模型是否使用预训练模型
teacher_choice = config['teacher_choice']  # 选择教师模型种类
student_choice = config['student_choice']  # 选择学生模型种类
teacher_epochs = config['teacher_epochs']  # 教师模型训练轮数
epochs = config['epochs']  # DFME训练轮数
lr_S = config['lr_S']  # 学生模型学习率
lr_G = config['lr_G']  # 生成器学习率


def get_teacher_model(train_loader, test_loader, input_dim=67, num_classes=2, epochs=50, choice=None,
                      use_pretrain=False):
    """训练教师模型"""
    if choice not in ['CNN', 'DNN', 'LSTM', 'RNN', 'GRU', 'Transformer', 'teacher', 'student','DNN1_CNN1']:
        logger.error('没有相应的预训练模型')
        return None, None
    if use_pretrain:
        logger.info(f'正在载入 {choice} 模型')
        teacher = None
        if choice == 'CNN':
            teacher = CNN(num_classes=num_classes).to(device)
            teacher.load_state_dict(torch.load('extraction/model/CNN.pt'))
        elif choice == 'DNN':
            teacher = DNN(input_dim=input_dim, num_classes=num_classes).to(device)
            teacher.load_state_dict(torch.load('extraction/model/DNN.pt'))
        elif choice == 'LSTM':
            teacher = LSTM(num_classes=num_classes).to(device)
            teacher.load_state_dict(torch.load('extraction/model/LSTM.pt'))
        elif choice == 'RNN':
            teacher = RNN(num_classes=num_classes).to(device)
            teacher.load_state_dict(torch.load('extraction/model/RNN.pt'))
        elif choice == 'GRU':
            teacher = GRU(num_classes=num_classes).to(device)
            teacher.load_state_dict(torch.load('extraction/model/GRU.pt'))
        elif choice == 'Transformer':
            teacher = Transformer(input_dim=input_dim, num_classes=num_classes).to(device)
            teacher.load_state_dict(torch.load('extraction/model/Transformer.pt'))
        elif choice == 'teacher':
            teacher = TeacherModel(num_classes=num_classes).to(device)
            teacher.load_state_dict(torch.load('extraction/model/teacher_model.pt'))
        elif choice == 'student':
            teacher = StudentModel(input_dim=input_dim, num_classes=num_classes).to(device)
            teacher.load_state_dict(torch.load('extraction/model/teacher_model.pt'))
        elif choice == 'DNN1_CNN1':
            teacher = DNN1_CNN1(input_dim=input_dim, num_classes=num_classes).to(device)
            teacher.load_state_dict(torch.load('extraction/model/DNN1_CNN1.pt'))
        teacher.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                # 将测试数据移到正确设备
                data = data.to(device)
                labels = labels.to(device)

                outputs = teacher(data)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        logger.info(f'载入 {choice} 模型成功，准确率为 {acc:.4f}%')
        return teacher, acc
    else:
        logger.info(f"开始训练 {choice} 教师模型")
        teacher = None
        if choice == 'CNN':
            teacher = CNN(num_classes=num_classes).to(device)
        elif choice == 'DNN':
            teacher = DNN(input_dim=input_dim, num_classes=num_classes).to(device)
        elif choice == 'LSTM':
            teacher = LSTM(num_classes=num_classes).to(device)
        elif choice == 'RNN':
            teacher = RNN(num_classes=num_classes).to(device)
        elif choice == 'GRU':
            teacher = GRU(num_classes=num_classes).to(device)
        elif choice == 'Transformer':
            teacher = Transformer(input_dim=input_dim, num_classes=num_classes).to(device)
        elif choice == 'teacher':
            teacher = TeacherModel(num_classes=num_classes).to(device)
        elif choice == 'student':
            teacher = StudentModel(input_dim=input_dim, num_classes=num_classes).to(device)
        elif choice == 'DNN1_CNN1':
            teacher = DNN1_CNN1(input_dim=input_dim, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(teacher.parameters(), lr=0.001, weight_decay=1e-4)

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        best_acc = 0.0

        for epoch in range(epochs):
            teacher.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = teacher(inputs)
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
            teacher.eval()
            test_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = teacher(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            test_acc = 100 * correct / total
            test_loss /= len(test_loader)

            logger.debug(f"Teacher Epoch [{epoch + 1}/{epochs}] - "
                         f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                         f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

            # 更新学习率
            scheduler.step(test_acc)

            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(teacher.state_dict(), "extraction/model/teacher_model.pt")

        logger.success(f"Teacher model trained with best accuracy: {best_acc:.2f}%")
        teacher.load_state_dict(torch.load("extraction/model/teacher_model.pt"))
        return teacher, best_acc


def main():
    if config['is_log']:
        logger.add('extraction/logs/DFME.log', colorize=False, level='DEBUG',
               format='<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> <red>|</red> <level>{level: <8}</level> ''<red>|</red> <level>{message}</level>')
        logger.add('extraction/logs/DFME_info.log', colorize=False, level='INFO',
               format='<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> <red>|</red> <level>{level: <8}</level> ''<red>|</red> <level>{message}</level>')
  
    # 加载和预处理数据
    train_dataset, test_dataset, scaler = load_and_preprocess_data(test_size=0.3)

    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(
        train_dataset, test_dataset, batch_size=batch_size
    )

    # 训练教师模型
    teacher, teacher_acc = get_teacher_model(
        train_loader, test_loader,
        input_dim=input_dim, num_classes=num_classes, epochs=teacher_epochs, use_pretrain=use_pretrain, choice=teacher_choice
    )

    # 初始化DFME训练框架
    logger.info("初始化 DFME 训练框架")
    dfme = DFME(
        teacher=teacher,
        student_choice=student_choice,
        latent_dim=latent_dim,
        input_dim=input_dim,
        num_classes=num_classes,
        m=3,
        epsilon=0.01,
        lr_G=lr_G,
        lr_S=lr_S
    )

    # 执行攻击
    dfme.train(
        epochs=epochs,
        test_loader=test_loader,
        query_budget=query_budget,
        batch_size=batch_size
    )

    # 绘制并保存结果
    # try:
    #     dfme.plot_results(save_path="results.png")
    # except Exception:
    #     logger.error('图片生成失败')

    # 保存模型
    if not os.path.exists(student_path):
        os.makedirs(student_path)
    if not os.path.exists(generator_path):
        os.makedirs(generator_path)
    dfme.save_models(student_path, generator_path)

    # 最终评估
    final_acc = dfme.evaluate_student(test_loader)
    logger.success(f"Final student accuracy: {final_acc:.4f}%")
    logger.success(f"Teacher accuracy: {teacher_acc:.4f}%")
    logger.success(f"Normalized accuracy: {final_acc / teacher_acc:.4f}")
    logger.success(f"Total queries used: {dfme.query_count}")

    # 比较教师和学生模型的性能
    logger.info("Performance Comparison:")
    logger.info(f"Teacher Model Accuracy: {teacher_acc:.4f}%")
    logger.info(f"Student Model Accuracy: {final_acc:.4f}%")
    logger.info(f"Effectiveness of Attack: {final_acc / teacher_acc:.4f}%")


def experiment():
    """实验不同模型组合的提取效果"""
    # 参数设置
    if config['is_log']:
        logger.add('extraction/logs/DFME.log', colorize=False, level='DEBUG',
               format='<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> <red>|</red> <level>{level: <8}</level> ''<red>|</red> <level>{message}</level>')
        logger.add('extraction/logs/DFME_info.log', colorize=False, level='INFO',
               format='<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> <red>|</red> <level>{level: <8}</level> ''<red>|</red> <level>{message}</level>')
    input_dim = 67  # 特征维度
    num_classes = 2  # 类别数
    latent_dim = 50  # 生成器输入维度
    query_budget = 20000  # 查询预算
    batch_size = 32  # 批次大小
    student_path = "model"  # 学生模型保存路径
    generator_path = "model"  # 生成器保存路径
    use_pretrain = True  # 教师模型是否使用预训练模型

    model_list = ['DNN1_CNN1','CNN', 'DNN', 'LSTM', 'RNN', 'GRU', 'Transformer']
    # for teacher_choice in model_list:
    #     student_choice = 'student'  
    #     # 加载和预处理数据
    #     train_dataset, test_dataset, scaler = load_and_preprocess_data(test_size=0.3)

    #     # 创建数据加载器
    #     train_loader, test_loader = create_data_loaders(
    #         train_dataset, test_dataset, batch_size=batch_size
    #     )
    #     # 训练教师模型
    #     teacher, teacher_acc = get_teacher_model(
    #         train_loader, test_loader,
    #         input_dim=input_dim, num_classes=num_classes, epochs=3, use_pretrain=use_pretrain, choice=teacher_choice
    #     )
    #     # 初始化DFME训练框架
    #     dfme = DFME(
    #         teacher=teacher,
    #         student_choice=student_choice,
    #         latent_dim=latent_dim,
    #         input_dim=input_dim,
    #         num_classes=num_classes,
    #         m=1,
    #         epsilon=0.01,
    #         lr_G=0.0001,
    #         lr_S=0.001
    #     )
    #     # 执行攻击
    #     dfme.train(
    #         epochs=100,
    #         test_loader=test_loader,
    #         query_budget=query_budget,
    #         batch_size=batch_size
    #     )
    #     # 保存模型
    #     if not os.path.exists(student_path):
    #         os.makedirs(student_path)
    #     if not os.path.exists(generator_path):
    #         os.makedirs(generator_path)
    #     dfme.save_models(student_path, generator_path)

    #     # 最终评估
    #     final_acc = dfme.evaluate_student(test_loader)
    #     logger.info(f"{student_choice}/{teacher_choice} Final student accuracy: {final_acc:.4f}%")
    #     logger.info(f"{student_choice}/{teacher_choice} Teacher accuracy: {teacher_acc:.4f}%")
    #     logger.info(f"{student_choice}/{teacher_choice} Normalized accuracy: {final_acc / teacher_acc:.4f}")
    #     logger.info(f"{student_choice}/{teacher_choice} Total queries used: {dfme.query_count}")
            
    
    for student_choice in model_list:
        teacher_choice = 'student'  
        # 加载和预处理数据
        train_dataset, test_dataset, scaler = load_and_preprocess_data(test_size=0.3)

        # 创建数据加载器
        train_loader, test_loader = create_data_loaders(
            train_dataset, test_dataset, batch_size=batch_size
        )
        # 训练教师模型
        teacher, teacher_acc = get_teacher_model(
            train_loader, test_loader,
            input_dim=input_dim, num_classes=num_classes, epochs=3, use_pretrain=use_pretrain, choice=teacher_choice
        )
        # 初始化DFME训练框架
        dfme = DFME(
            teacher=teacher,
            student_choice=student_choice,
            latent_dim=latent_dim,
            input_dim=input_dim,
            num_classes=num_classes,
            m=1,
            epsilon=0.01,
            lr_G=0.0001,
            lr_S=0.001
        )
        # 执行攻击
        dfme.train(
            epochs=100,
            test_loader=test_loader,
            query_budget=query_budget,
            batch_size=batch_size
        )
        # 保存模型
        if not os.path.exists(student_path):
            os.makedirs(student_path)
        if not os.path.exists(generator_path):
            os.makedirs(generator_path)
        dfme.save_models(student_path, generator_path)

        # 最终评估
        final_acc = dfme.evaluate_student(test_loader)
        logger.info(f"{student_choice}/{teacher_choice} Final student accuracy: {final_acc:.4f}%")
        logger.info(f"{student_choice}/{teacher_choice} Teacher accuracy: {teacher_acc:.4f}%")
        logger.info(f"{student_choice}/{teacher_choice} Normalized accuracy: {final_acc / teacher_acc:.4f}")
        logger.info(f"{student_choice}/{teacher_choice} Total queries used: {dfme.query_count}")


if __name__ == "__main__":
    main()
    # experiment()
    