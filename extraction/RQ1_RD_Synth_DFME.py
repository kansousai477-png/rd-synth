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
    """
    严肃版 Data-Free Model Extraction:
    - teacher 仅通过前向查询（模拟黑盒）
    - generator 通过前向差分近似梯度更新
    - student 拟合 teacher 的“伪 logits”
    """

    def __init__(
        self,
        teacher,
        input_dim,
        num_classes=2,
        latent_dim=50,
        n_G=1,
        n_S=5,
        m=3,
        epsilon=0.01,
        lr_G=1e-4,
        lr_S=1e-3
    ):
        self.teacher = teacher.to(device)
        self.teacher.eval()  # 黑盒，不反向

        self.student = StudentModel(input_dim=input_dim, num_classes=num_classes).to(device)
        self.generator = Generator(latent_dim=latent_dim, output_dim=input_dim).to(device)

        self.optimizer_S = optim.Adam(self.student.parameters(), lr=lr_S, weight_decay=1e-4)
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=lr_G, betas=(0.5,0.999))

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.n_G = n_G        # 每轮更新 G 次数
        self.n_S = n_S        # 每轮更新 student 次数
        self.m = m            # 方向数量（forward-diff）
        self.epsilon = epsilon

        self.loss_fn = nn.L1Loss()

        self.student_acc_history = []
        self.student_loss_history = []
        self.query_count = 0

    # ================================================
    # helper：教师概率恢复伪 logits
    # ================================================
    def recover_logits(self, probabilities):
        log_probs = torch.log(probabilities + 1e-10)
        mean_logits = torch.mean(log_probs, dim=1, keepdim=True)
        approx_logits = log_probs - mean_logits
        return approx_logits

    # ================================================
    # forward difference 近似梯度
    # ================================================
    def forward_difference(self, z, loss_func):
        z = z.detach().requires_grad_(True)
        batch_size = z.size(0)

        x_base = self.generator(z)

        with torch.no_grad():
            teacher_probs = torch.softmax(self.teacher(x_base), dim=1)
            teacher_logits = self.recover_logits(teacher_probs)
            student_logits = self.student(x_base)
            base_loss = loss_func(student_logits, teacher_logits)

        grad_approx = torch.zeros_like(z)

        for _ in range(self.m):
            u = torch.randn_like(z, device=device)
            u_norm = u / torch.norm(u, dim=1, keepdim=True)

            z_perturbed = z + self.epsilon * u_norm
            x_perturbed = self.generator(z_perturbed)

            with torch.no_grad():
                t_probs_p = torch.softmax(self.teacher(x_perturbed), dim=1)
                t_logits_p = self.recover_logits(t_probs_p)
                s_logits_p = self.student(x_perturbed)
                perturbed_loss = loss_func(s_logits_p, t_logits_p)

            directional_derivative = (perturbed_loss - base_loss) / self.epsilon
            grad_approx += directional_derivative.view(-1, 1) * u_norm

        grad_approx /= self.m
        self.query_count += 2 * self.m * batch_size
        return grad_approx

    # ================================================
    # 更新 generator
    # ================================================
    def train_generator_step(self, batch_size=32):
        self.generator.train()
        self.student.eval()

        z = torch.randn(batch_size, self.latent_dim, device=device, requires_grad=True)

        def loss_func(student_logits, teacher_logits):
            return self.loss_fn(student_logits, teacher_logits)

        grad_approx = self.forward_difference(z, loss_func)

        self.optimizer_G.zero_grad()
        z.backward(grad_approx)
        self.optimizer_G.step()

    # ================================================
    # 更新 student
    # ================================================
    def train_student_step(self, batch_size=32):
        self.student.train()
        self.generator.eval()

        total_loss = 0.0

        for _ in range(self.n_S):
            z = torch.randn(batch_size, self.latent_dim, device=device)

            with torch.no_grad():
                x = self.generator(z)

            with torch.no_grad():
                teacher_probs = torch.softmax(self.teacher(x), dim=1)
                teacher_logits = self.recover_logits(teacher_probs)
                self.query_count += x.size(0)

            student_logits = self.student(x)
            loss = self.loss_fn(student_logits, teacher_logits)
            total_loss += loss.item()

            self.optimizer_S.zero_grad()
            loss.backward()
            self.optimizer_S.step()

        return total_loss / self.n_S

    # ================================================
    # 在真实 test_loader 上评估 student
    # ================================================
    def evaluate_student(self, test_loader):
        self.student.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = self.student(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100.0 * correct / total
        self.student_acc_history.append(acc)
        return acc

    # ================================================
    # 总训练流程
    # ================================================
    def train(self, epochs, test_loader, query_budget=20000, batch_size=32):
        print("[DFME] ===== Data-Free DFME 训练开始 =====")

        for epoch in range(1, epochs + 1):

            if self.query_count >= query_budget:
                print(f"[DFME] 查询预算 {query_budget} 已用尽，停止于 epoch {epoch}")
                break

            # 更新 Generator
            for _ in range(self.n_G):
                self.train_generator_step(batch_size=batch_size)

            # 更新 Student
            student_loss = self.train_student_step(batch_size=batch_size)
            self.student_loss_history.append(student_loss)

            # 在真实数据上评估一次
            acc = self.evaluate_student(test_loader)

            print(
                f"[DFME] Epoch {epoch}/{epochs} | "
                f"Loss={student_loss:.6f} | "
                f"Acc={acc:.2f}% | "
                f"Queries={self.query_count}/{query_budget}"
            )

        print("[DFME] ===== Data-Free DFME 训练完成 =====")

    def save_models(self):
        torch.save(self.student.state_dict(), os.path.join(MODEL_SAVE_DIR, "DFME_student.pt"))
        torch.save(self.generator.state_dict(), os.path.join(MODEL_SAVE_DIR, "DFME_generator.pt"))
        print("[DFME] DFME Student 和 Generator 已保存")



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
    print("\n[INFO] 加载数据...")
    train_loader, test_loader, input_dim = load_dataset()

    # =============================
    # 1. 加载已经训练好的 Teacher 模型
    # =============================
    print("\n[INFO] 加载已有的 Teacher 模型权重，不重新训练它们")
    model_paths = {
        "DNN":          "DNN.pt",
        "CNN":          "CNN.pt",
        "RNN":          "RNN.pt",
        "LSTM":         "LSTM.pt",
        "GRU":          "GRU.pt",
        "Transformer":  "Transformer.pt",
        "DNN1_CNN1":    "DNN1_CNN1.pt",
        "TeacherModel": "TeacherModel.pt"
    }

    # 每个模型的构造函数（必须一致）
    model_zoo = {
        "DNN":          lambda: DNN(input_dim),
        "CNN":          lambda: CNN(input_dim),
        "RNN":          lambda: RNN(input_dim),
        "LSTM":         lambda: LSTM(input_dim),
        "GRU":          lambda: GRU(input_dim),
        "Transformer":  lambda: Transformer(input_dim),
        "DNN1_CNN1":    lambda: DNN1_CNN1(input_dim),
        "TeacherModel": lambda: TeacherModel(input_dim)
    }

    results = {}

    # =============================
    # 2. 加载模型并测试它们
    # =============================
    for name, constructor in model_zoo.items():
        print(f"\n[INFO] 加载模型: {name}")
        model = constructor()
        model.load_state_dict(
            torch.load(
                os.path.join(MODEL_SAVE_DIR, model_paths[name]),
                map_location=device
            )
        )
        model.to(device)
        model.eval()

        # 测试它的效果
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(1)
                total += len(y)
                correct += (pred == y).sum().item()

        acc = 100.0 * correct / total
        print(f"[Teacher:{name}] Test Accuracy = {acc:.2f}%")
        results[name] = acc

    # =============================
    # 3. DFME（只有这个需要训练）
    # =============================
    print("\n===== 开始 Data-free DFME（Victim = TeacherModel） =====")
    victim = TeacherModel(input_dim)
    victim.load_state_dict(torch.load(
        os.path.join(MODEL_SAVE_DIR, "TeacherModel.pt"),
        map_location=device
    ))
    victim.eval()

    dfme = DFME(
        teacher=victim,
        input_dim=input_dim,
        num_classes=2,
        latent_dim=50,
        n_G=1,
        n_S=5,
        m=3,
        epsilon=0.01,
        lr_G=1e-4,
        lr_S=1e-3
    )

    dfme.train(
        epochs=100,
        test_loader=test_loader,
        query_budget=20000,
        batch_size=32
    )

    dfme_acc = dfme.evaluate_student(test_loader)
    results["DFME_Student"] = dfme_acc

    # =============================
    # 4. 输出最终对比表
    # =============================
    print("\n================ 最终模型性能对比表 ================")
    print("{:<20}{}".format("Model", "Accuracy"))
    print("-"*40)
    for k, v in results.items():
        print(f"{k:<20}{v:.2f}%")
    print("====================================================")



if __name__ == "__main__":
    main()
