# -*- coding: utf-8 -*-
# ==============================================================
# RD_Synth_ME.py (Progress-enhanced version)
# ==============================================================

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_PATH = "../data/unsw/CICFlowMeter_preprocessed.csv"
MODEL_SAVE_DIR = "./extraction_all_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


# ==============================================================
# Step 0: 加载数据 + 日志
# ==============================================================

def load_dataset(csv_path=CSV_PATH):
    print("\n[Data] 开始加载数据...")
    t0 = time.time()

    df = pd.read_csv(csv_path)
    print(f"[Data] 数据读取完成，共 {len(df)} 行, {len(df.columns)} 列")

    assert "Label" in df.columns, "数据集中必须包含 Label 列"

    print("[Data] 正在分离特征和标签...")
    X = df.drop("Label", axis=1).values
    y = df["Label"].values

    print("[Data] 正在标准化...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, "scaler.pkl"))

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    print("[Data] 正在划分训练和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y
    )

    print("[Data] 正在构建 DataLoader...")
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

    print(f"[Data] 数据加载完成，用时 {time.time()-t0:.2f} 秒")
    return train_loader, test_loader, X.shape[1]


# ==============================================================
# Step 1: 定义所有模型...
# ==============================================================

class DNN(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(),
            nn.Linear(32,8), nn.ReLU(),
            nn.Linear(8,num_classes)
        )
    def forward(self,x): return self.net(x)

class StudentModel(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(input_dim,64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64,32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Linear(32,8), nn.BatchNorm1d(8), nn.ReLU()
        )
        self.cls = nn.Linear(8,num_classes)
    def forward(self,x):
        x=self.feat(x)
        return self.cls(x)

class TeacherModel(StudentModel):
    pass

class CNN(nn.Module):
    def __init__(self,input_dim,num_classes=2):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(1,32,3),nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32,64,3),nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * max(1, ((input_dim-4)//4)),128),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        x=x.unsqueeze(1)
        x=self.feat(x)
        return self.cls(x)

class RNN(nn.Module):
    def __init__(self,input_dim,num_classes=2):
        super().__init__()
        self.rnn = nn.RNN(1,32,batch_first=True)
        self.fc = nn.Linear(32,num_classes)
    def forward(self,x):
        x=x.unsqueeze(-1)
        out,_=self.rnn(x)
        return self.fc(out[:,-1,:])

class LSTM(nn.Module):
    def __init__(self,input_dim,num_classes=2):
        super().__init__()
        self.rnn = nn.LSTM(1,32,batch_first=True)
        self.fc = nn.Linear(32,num_classes)
    def forward(self,x):
        x=x.unsqueeze(-1)
        out,_=self.rnn(x)
        return self.fc(out[:,-1,:])

class GRU(nn.Module):
    def __init__(self,input_dim,num_classes=2):
        super().__init__()
        self.rnn = nn.GRU(1,32,batch_first=True)
        self.fc = nn.Linear(32,num_classes)
    def forward(self,x):
        x=x.unsqueeze(-1)
        out,_=self.rnn(x)
        return self.fc(out[:,-1,:])

class Transformer(nn.Module):
    def __init__(self,input_dim,num_classes=2):
        super().__init__()
        self.embed = nn.Linear(1,16)
        self.pos = nn.Parameter(torch.randn(input_dim,16))
        enc = nn.TransformerEncoderLayer(d_model=16,nhead=4)
        self.encoder = nn.TransformerEncoder(enc,1)
        self.cls = nn.Linear(16,num_classes)
    def forward(self,x):
        x=x.unsqueeze(-1)
        x=self.embed(x)+self.pos
        x=x.permute(1,0,2)
        x=self.encoder(x)
        return self.cls(x[0])

class DNN1_CNN1(nn.Module):
    def __init__(self,input_dim,num_classes=2):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(input_dim,64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(1,32,3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.cls = nn.Sequential(
            nn.Linear(32*((64-3+1)//2),64),
            nn.ReLU(),
            nn.Linear(64,num_classes)
        )
    def forward(self,x):
        x=self.dnn(x)
        x=x.unsqueeze(1)
        x=self.cnn(x)
        x=x.view(x.size(0),-1)
        return self.cls(x)


# ==============================================================
# Step 2: DFME 加强日志
# ==============================================================

class Generator(nn.Module):
    def __init__(self,latent_dim,output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim,128),nn.ReLU(),
            nn.Linear(128,256),nn.ReLU(),
            nn.Linear(256,512),nn.ReLU(),
            nn.Linear(512,output_dim)
        )
    def forward(self,z): return self.net(z)


class DFME:
    def __init__(self, teacher, input_dim, latent_dim=50):
        self.teacher = teacher.to(device)
        self.teacher.eval()
        self.student = StudentModel(input_dim).to(device)
        self.generator = Generator(latent_dim,input_dim).to(device)
        self.opt_s = optim.Adam(self.student.parameters(), lr=0.001)
        self.loss_fn = nn.L1Loss()
        self.latent_dim = latent_dim

    def train(self, epochs=10):
        print("\n[DFME] ===== DFME 训练开始 =====")
        for ep in range(epochs):
            z = torch.randn(32, self.latent_dim).to(device)
            x_fake = self.generator(z).detach()

            with torch.no_grad():
                t_prob = torch.softmax(self.teacher(x_fake),dim=1)

            s_out = self.student(x_fake)
            loss = self.loss_fn(s_out, t_prob)

            self.opt_s.zero_grad()
            loss.backward()
            self.opt_s.step()

            print(f"[DFME] Epoch {ep+1}/{epochs} | Loss={loss.item():.6f}")

        print("[DFME] ===== DFME 训练完成 =====")

    def evaluate(self,test_loader):
        print("\n[DFME] 开始评估 Student...")
        self.student.eval()
        correct,total = 0,0
        with torch.no_grad():
            for i,(x,y) in enumerate(test_loader):
                x,y=x.to(device),y.to(device)
                out = self.student(x)
                pred = out.argmax(1)
                total+=len(y)
                correct+=(pred==y).sum().item()
        acc = 100*correct/total
        print(f"[DFME] Student Accuracy = {acc:.2f}%")
        return acc


# ==============================================================
# Step 3: 教师模型训练 + 强化进度
# ==============================================================

def train_teacher(model, train_loader, test_loader, save_name):

    print(f"\n==============================")
    print(f"[Teacher] 开始训练模型 {save_name}")
    print("==============================")

    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1,4):
        print(f"[Teacher:{save_name}] Epoch {ep}/3")
        model.train()

        for batch_idx, (x,y) in enumerate(train_loader):
            x,y=x.to(device),y.to(device)
            out=model(x)
            loss=loss_fn(out,y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if batch_idx % 50 == 0:
                print(f"  └─ Batch {batch_idx}/{len(train_loader)} | Loss={loss.item():.4f}")

    print(f"[Teacher:{save_name}] 模型训练完毕，开始评估...")

    # test
    model.eval()
    correct,total = 0,0
    with torch.no_grad():
        for x,y in test_loader:
            x,y=x.to(device),y.to(device)
            out=model(x); pred=out.argmax(1)
            total+=len(y)
            correct+=(pred==y).sum().item()
    acc = 100 * correct / total

    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, save_name))
    print(f"[Teacher:{save_name}] 测试集准确率 = {acc:.2f}%")
    return acc, model


# ==============================================================
# Step 4: 主程序
# ==============================================================

def main():

    train_loader, test_loader, input_dim = load_dataset()

    model_zoo = {
        "DNN": lambda: DNN(input_dim),
        "CNN": lambda: CNN(input_dim),
        "RNN": lambda: RNN(input_dim),
        "LSTM": lambda: LSTM(input_dim),
        "GRU": lambda: GRU(input_dim),
        "Transformer": lambda: Transformer(input_dim),
        "DNN1_CNN1": lambda: DNN1_CNN1(input_dim),
        "TeacherModel": lambda: TeacherModel(input_dim)
    }

    results = {}

    print("\n===== 开始训练所有教师模型 =====")

    for name, build_fn in model_zoo.items():
        acc, _ = train_teacher(build_fn(), train_loader, test_loader, save_name=f"{name}.pt")
        results[name] = acc

    print("\n===== 开始 DFME 训练（TeacherModel 作为 victim） =====")
    victim = TeacherModel(input_dim)
    victim.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR,"TeacherModel.pt")))
    victim.eval()

    dfme = DFME(victim, input_dim)
    dfme.train(epochs=10)
    dfme_acc = dfme.evaluate(test_loader)

    results["DFME_Student"] = dfme_acc

    # ==========================================================
    # 最终性能对比表
    # ==========================================================
    print("\n================ 最终模型性能对比表 ================")
    print("{:<20}{}".format("Model", "Accuracy"))
    print("-"*40)
    for k,v in results.items():
        print(f"{k:<20}{v:.2f}%")
    print("====================================================")


if __name__ == "__main__":
    main()
