import pandas as pd
import torch
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

config = yaml.safe_load(open('extraction/config.yaml'))
benign_path = config['benign_path']
malicious_path = config['malicious_path']
def load_and_preprocess_data(test_size=0.3):
    """
    加载并预处理表格数据
    """
    # 加载数据
    df1 = pd.read_csv(benign_path)
    df2 = pd.read_csv(malicious_path)
    data = pd.concat([df1, df2])

    # 去除不需要的特征
    unused_features = [
        'Unnamed: 0', 'id', 'expiration_id', 'src_ip', 'dst_ip', 'src_mac', 'dst_mac', 'src_oui',
        'dst_oui', 'vlan_id', 'tunnel_id', 'client_fingerprint', 'server_fingerprint',
        'application_is_guessed', 'application_confidence', 'requested_server_name',
        'user_agent', 'content_type', 'application_name', 'application_category_name',
    ]
    data = data.drop(columns=unused_features, errors='ignore')

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
    
    # 保存标准化器
    joblib.dump(scaler, 'extraction/model/scaler.pkl')

    return train_dataset, test_dataset, scaler


def create_data_loaders(train_dataset, test_dataset, batch_size=32):
    """创建训练和测试数据加载器"""
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader
