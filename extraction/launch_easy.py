import os.path

import torch
from loguru import logger
import yaml
from data_loader import load_and_preprocess_data, create_data_loaders
from launch import get_teacher_model
from models import StudentModel
from train import EME

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
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


def evaluate_student(student, test_loader):
    """评估学生模型性能"""
    student.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            # 将测试数据移到正确设备
            data = data.to(device)
            labels = labels.to(device)

            outputs = student(data)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def main():
    """使用EasyEME方法训练学生模型"""
    if config['is_log']:
        logger.add('extraction/logs/EME.log', colorize=False, level='DEBUG',
                format='<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> <red>|</red> <level>{level: <8}</level> ''<red>|</red> <level>{message}</level>')
        logger.add('extraction/logs/EME_info.log', colorize=False, level='INFO',
                format='<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> <red>|</red> <level>{level: <8}</level> ''<red>|</red> <level>{message}</level>')

    # 参数设置
    lr_S = 0.001  # 学习率
    n_S = 5  # 训练轮数
    synthesis_data_path = 'data'  # 合成数据集保存路径

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

    eme = EME(teacher=teacher,
              student_choice=student_choice,
              input_dim=input_dim,
              num_classes=num_classes,
              lr_S=lr_S, n_S=n_S)
    # 合成数据集（如果已经训练好就不用再合成）
    eme.synthesis_dataset(synthesis_data_path,query_budget=query_budget)

    # 开始训练学生模型
    sy_train_loader, sy_test_loader = eme.preprocess(synthesis_data_path)
    eme.train_student(sy_train_loader, sy_test_loader, student_path=student_path)

    # 独立测试学生模型的效果（分别与原始数据集和合成数据集来测试）
    student = StudentModel(input_dim=input_dim, num_classes=num_classes).to(device)
    student.load_state_dict(torch.load(os.path.join(student_path, "easy_student_model.pt")))
    acc = evaluate_student(student, test_loader)
    acc_syn = evaluate_student(student, sy_test_loader)

    logger.info("学生模型评估报告：")
    logger.info(f"在原始数据集上的准确度: {acc:.4f}%")
    logger.info(f"与教师模型的相似度: {acc_syn:.4f}%")


def experiment():
    # 参数设置
    if config['is_log']:
        logger.add('extraction/logs/EME.log', colorize=False, level='DEBUG',
                format='<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> <red>|</red> <level>{level: <8}</level> ''<red>|</red> <level>{message}</level>')
        logger.add('extraction/logs/EME_info.log', colorize=False, level='INFO',
                format='<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> <red>|</red> <level>{level: <8}</level> ''<red>|</red> <level>{message}</level>')
    lr_S = 0.001  # 学习率
    n_S = 5  # 训练轮数
    query_budget = 20000  # 查询预算
    student_path = "extraction/model"  # 学生模型保存路径
    synthesis_data_path = 'data'  # 合成数据集保存路径
    use_pretrain = True  # 教师模型是否使用预训练模型

    
    model_list = ['CNN', 'DNN', 'LSTM', 'RNN', 'GRU', 'Transformer']
    for teacher_choice in model_list:
        for student_choice in model_list:
            # 加载和预处理数据
            train_dataset, test_dataset, scaler = load_and_preprocess_data(test_size=0.3)

            # 创建数据加载器
            train_loader, test_loader = create_data_loaders(
                train_dataset, test_dataset, batch_size=batch_size
            )

            # 训练教师模型
            logger.info(f"当前实验组合：教师模型-{teacher_choice}，学生模型-{student_choice}")
            teacher, teacher_acc = get_teacher_model(
                train_loader, test_loader,
                input_dim=input_dim, num_classes=num_classes, epochs=3, use_pretrain=use_pretrain, choice=teacher_choice
            )

            eme = EME(teacher=teacher,
                      student_choice=student_choice,
                      input_dim=input_dim,
                      num_classes=num_classes,
                      lr_S=lr_S, n_S=n_S)
            # 合成数据集（如果已经训练好就不用再合成）
            eme.synthesis_dataset(synthesis_data_path,query_budget=query_budget)
            # 开始训练学生模型
            sy_train_loader, sy_test_loader = eme.preprocess(synthesis_data_path)
            eme.train_student(sy_train_loader, sy_test_loader, student_path=student_path)
            # 最终评估
            final_acc = eme.evaluate_student(test_loader)
            logger.success(f"{student_choice}/{teacher_choice} Experiment completed.")
            logger.info(f"{student_choice}/{teacher_choice} Final student accuracy: {final_acc:.4f}%")
            logger.info(f"{student_choice}/{teacher_choice} Teacher accuracy: {teacher_acc:.4f}%")
            logger.info(f"{student_choice}/{teacher_choice} Normalized accuracy: {final_acc / teacher_acc:.4f}")


if __name__ == "__main__":
    main()
    # experiment()