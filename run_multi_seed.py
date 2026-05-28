import subprocess
import yaml
import os

# ==========================================
# 0. 全局配置
# ==========================================
# 你的基础配置文件路径 (请确保该文件存在并具有基本的 yaml 结构)
config_path = "config/test_config.yaml"
# 你的联邦学习主程序路径
main_script = "fl_main.py"

# ==========================================
# 1. 实验参数矩阵定义
# ==========================================
# 5 个数据集
datasets = [
    # 'MNIST',
    'CIFAR10',
    'FashionMNIST',
    'CIFAR100',
    # 'ImageNet',
    # 'LC25000'
]

# 数据集与模型的严格映射关系
dataset_model_map = {
    "MNIST": "LeNet",
    "FashionMNIST": "LeNet",
    "CIFAR10": "CNN",
    "CIFAR100": "CNN",
    "ImageNet": "ResNet18",
    "LC25000": "ResNet18"
}

# 3 个数据异构度 (Non-IID 程度)
# 0.01 极度 Non-IID, 0.1 中度 Non-IID, 1.0 轻度/接近 IID
alphas = [
    0.01,
    0.1,
    1.0
]

# 8 个核心算法 (经典 Baseline + GRO 变体)
algorithms = [
    "FedAvg",
    "Scaffold",
    "FedProx",
    "FedNova",
    "FedAvgGRO",
    "ScaffoldGRO",
    "FedProxGRO",
    "FedNovaGRO"
]

# 5 个随机种子 (用于多次实验求平均和置信区间)
seeds = [
    0,
    1,
    42,
    999,
    2026
]

# ==========================================
# 2. 自动化运行主循环
# ==========================================
# 总实验次数 = 5(数据集) * 3(异构度) * 8(算法) * 5(种子) = 600 次
total_experiments = len(datasets) * len(alphas) * len(algorithms) * len(seeds)
current_run = 0

print(f"🚀 初始化实验矩阵... 总计需运行 {total_experiments} 组实验。")

# 检查模板文件是否存在
if not os.path.exists(config_path):
    print(f"❌ 找不到配置文件模板: {config_path}，请先创建它！")
    exit(1)

for dataset in datasets:
    # 动态判定学习率：LC25000 使用 0.01（可根据需要微调）
    current_lr = 0.01

    for alpha in alphas:
        for algo in algorithms:
            for s in seeds:
                current_run += 1

                # 1. 读取 YAML 模板
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    if config is None:
                        config = {}

                # 确保一级字典 key 存在
                if "system" not in config: config["system"] = {}
                if "client" not in config: config["client"] = {}

                # 2. 动态修改配置矩阵
                config["system"]["dataset"] = dataset
                config["system"]["dirichlet_alpha"] = float(alpha)  # 强转为 float，确保 yaml 格式标准
                config["client"]["fed_algo"] = algo
                config["system"]["i_seed"] = s

                # 动态赋予对应的模型与学习率
                current_model = dataset_model_map[dataset]
                config["system"]["model"] = current_model
                config["client"]["lr"] = current_lr

                # ✨ 新增判断：当数据集为 LC25000 时，将客户端数量改为 5，其余数据集保持 10
                if dataset == "LC25000":
                    config["system"]["num_client"] = 5
                else:
                    config["system"]["num_client"] = 10

                # 3. 写回 YAML 文件
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(config, f, sort_keys=False)

                # 进度打印，加入 num_client 维度方便监控
                print(f"\n[{current_run}/{total_experiments}] ===== Dataset: {dataset} | Alpha: {alpha} | Model: {current_model} | LR: {current_lr} | Clients: {config['system']['num_client']} | Algo: {algo} | Seed: {s} =====")

                # 4. 运行主函数
                try:
                    subprocess.run(["python", main_script, "--config", config_path], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ 警告: 当前实验运行失败，已跳过。错误信息: {e}")
                    # 即使崩溃也继续下一个实验
                    continue

print("\n🎉 所有实验已全部运行完毕！可以开始进行数据分析与制图了！")