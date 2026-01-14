# DFME_train.py
# 只负责训练 DFME + 在训练结束时立即输出 Student 的性能

import torch
import os
from RQ1_RD_Synth_DFME import (
    DFME, load_dataset, MODEL_SAVE_DIR,
    TeacherModel, device
)

def evaluate(model, test_loader):
    """用于 DFME 训练后的评估"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            total += len(y)
            correct += (pred == y).sum().item()
    return 100 * correct / total


def main():

    print("\n[DFME] 加载数据...")
    train_loader, test_loader, input_dim = load_dataset()

    # 加载 Teacher Model（Victim）
    print("\n[DFME] 加载 TeacherModel.pt ...")
    victim = TeacherModel(input_dim)
    victim.load_state_dict(torch.load(
        os.path.join(MODEL_SAVE_DIR, "TeacherModel.pt"),
        map_location=device
    ))
    victim.to(device)
    victim.eval()

    print("[DFME] TeacherModel 加载完毕，准备开始 Data-Free 训练...")

    # 创建 DFME 对象
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

    # 正式训练 DFME Student
    dfme.train(
        epochs=100,
        test_loader=test_loader,
        query_budget=20000,
        batch_size=32
    )

    # 保存 DFME 模型
    dfme.save_models()

    print("\n[DFME] 开始对 DFME Student 进行最终评估 ...")
    dfme_student = dfme.student
    dfme_acc = evaluate(dfme_student, test_loader)

    print("\n========== DFME 最终评估结果 ==========")
    print(f"DFME Student Accuracy = {dfme_acc:.2f}%")
    print("=======================================")

    # 可选：写入结果文件，便于 compare_models 读取
    with open(os.path.join(MODEL_SAVE_DIR, "DFME_result.txt"), "w") as f:
        f.write(f"DFME Student Accuracy = {dfme_acc:.2f}%\n")

    print(f"[DFME] 结果已写入 {MODEL_SAVE_DIR}/DFME_result.txt")


if __name__ == "__main__":
    main()
