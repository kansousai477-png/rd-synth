from sys import stdout
from loguru import logger
import torch
import joblib
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from models import CNN, DNN, LSTM, RNN, GRU, Transformer, TeacherModel, StudentModel

# 设置日志
logger.remove(0)
logger.add(stdout, colorize=True, level='DEBUG',
           format='<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> <red>|</red> <level>{level: <8}</level> ''<red>|</red> <level>{message}</level>')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = 67  # 特征维度
num_classes = 2  # 类别数

def load_data():
    """加载数据并返回原始DataFrame、处理后的特征张量和标签张量"""
    csv_path = 'data/generated_data/perturbations.csv'
    data = pd.read_csv(csv_path)
    logger.info(f"加载数据完成: 共 {len(data)} 个样本")
    
    scaler_path = 'extraction/model/scaler.pkl'
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(data)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    return data, X_tensor

def get_model(choice):
    """根据选择加载对应的模型"""
    model = None
    if choice == 'CNN':
        model = CNN(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load('extraction/model/CNN.pt', map_location=device))
    elif choice == 'DNN':
        model = DNN(input_dim=input_dim, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load('extraction/model/DNN.pt', map_location=device))
    elif choice == 'LSTM':
        model = LSTM(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load('extraction/model/LSTM.pt', map_location=device))
    elif choice == 'RNN':
        model = RNN(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load('extraction/model/RNN.pt', map_location=device))
    elif choice == 'GRU':
        model = GRU(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load('extraction/model/GRU.pt', map_location=device))
    elif choice == 'Transformer':
        model = Transformer(input_dim=input_dim, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load('extraction/model/Transformer.pt', map_location=device))
    elif choice == 'teacher':
        model = TeacherModel(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load('extraction/model/teacher_model.pt', map_location=device))
    elif choice == 'student':
        model = StudentModel(input_dim=input_dim, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load('extraction/model/teacher_model.pt', map_location=device))
    model.eval()
    return model

def predict_with_model(model, X_tensor, batch_size=32):
    """使用模型进行预测并返回预测结果列表"""
    dataset = TensorDataset(X_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    with torch.no_grad():
        for batch in data_loader:
            # 取出特征并移到正确设备
            data = batch[0].to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            # 将预测结果转回CPU并转换为列表
            all_predictions.extend(predicted.cpu().numpy().tolist())
    
    return all_predictions

def main():
    result_df, X_tensor = load_data()
    output_path = 'data/generated_data/predictions_with_all_models.csv'
    models = ['CNN', 'DNN', 'LSTM', 'RNN', 'GRU', 'Transformer', 'teacher', 'student']
    
    for model_name in models:
        logger.info(f'正在使用 {model_name} 模型进行预测')
        try:
            model = get_model(model_name)
            predictions = predict_with_model(model, X_tensor)
            # 添加预测结果到DataFrame
            result_df[f'pred_{model_name}'] = predictions
            logger.info(f'{model_name} 模型预测完成')
        except Exception as e:
            logger.error(f'{model_name} 模型预测失败: {str(e)}')
    
    result_df.to_csv(output_path, index=False)
    logger.info(f'所有预测结果已保存到: {output_path}')

if __name__ == "__main__":
    main()
