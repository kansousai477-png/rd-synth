# compare_models.py

import torch, os
from RQ1_RD_Synth_DFME import (
    load_dataset, MODEL_SAVE_DIR,
    DNN, CNN, RNN, LSTM, GRU, Transformer, DNN1_CNN1,
    TeacherModel, StudentModel, device
)


def evaluate(model, test_loader):
    model.eval()
    correct = 0;
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            total += len(y)
            correct += (pred == y).sum().item()
    return 100 * correct / total


def main():
    print("[Eval] 加载数据...")
    _, test_loader, input_dim = load_dataset()

    model_zoo = {
        "DNN": DNN(input_dim),
        "CNN": CNN(input_dim),
        "RNN": RNN(input_dim),
        "LSTM": LSTM(input_dim),
        "GRU": GRU(input_dim),
        "Transformer": Transformer(input_dim),
        "DNN1_CNN1": DNN1_CNN1(input_dim),
        "TeacherModel": TeacherModel(input_dim),
        "DFME_Student": StudentModel(input_dim)
    }

    result = {}

    for name, model in model_zoo.items():
        path = os.path.join(MODEL_SAVE_DIR, f"{name}.pt")
        if not os.path.exists(path):
            print(f"[WARN] {name}.pt 不存在，跳过")
            continue

        print(f"\n[Eval] 加载模型 {name}.pt ...")
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)

        acc = evaluate(model, test_loader)
        print(f"[Eval] {name} = {acc:.2f}%")
        result[name] = acc

    print("\n=========== 最终对比表 ===========")
    for k, v in result.items():
        print(f"{k:<20}{v:.2f}%")
    print("=================================")


if __name__ == "__main__":
    main()
