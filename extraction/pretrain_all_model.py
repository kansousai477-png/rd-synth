from train import pretrain_teacher_model
from data_loader import load_and_preprocess_data, create_data_loaders
from loguru import logger
from sys import stdout

# 设置日志
logger.remove(0)
logger.add(stdout, colorize=True, level='DEBUG',
           format='<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> <red>|</red> <level>{level: <8}</level> ''<red>|</red> <level>{message}</level>')

def main():
    # 加载和预处理数据
    train_dataset, test_dataset, scaler = load_and_preprocess_data(test_size=0.3)

    # 创建数据加载器
    train_loader, test_loader = create_data_loaders(
        train_dataset, test_dataset, batch_size=32
    )
    pretrain_teacher_model(train_loader, test_loader)
    
if __name__ == "__main__":
    main()