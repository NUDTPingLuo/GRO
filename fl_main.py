#!/usr/bin/env python
import os
import random
import json
import pickle
import argparse
import yaml
from json import JSONEncoder
from tqdm import tqdm

from fed_baselines.client_base import FedClient
from fed_baselines.client_fedprox import FedProxClient
from fed_baselines.client_scaffold import ScaffoldClient
from fed_baselines.client_fednova import FedNovaClient
from fed_baselines.client_gro import GROClient
from fed_baselines.client_scaffold_gro import ScaffoldGROClient
from fed_baselines.client_fednova_gro import FedNovaGROClient
from fed_baselines.client_fedprox_gro import FedProxGROClient
from fed_baselines.server_base import FedServer
from fed_baselines.server_scaffold import ScaffoldServer
from fed_baselines.server_fednova import FedNovaServer
from fed_baselines.server_gro import GROServer

from postprocessing.recorder import Recorder
from preprocessing.baselines_dataloader import divide_data_dirichlet
from utils.models import *

json_types = (list, dict, str, int, float, bool, type(None))

print("CUDA是否可用:", torch.cuda.is_available())
print("可用的GPU数量:", torch.cuda.device_count())
print("当前默认GPU设备:", torch.cuda.current_device())
print("当前默认设备名称:",
      torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "无GPU")


class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}


def fed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Yaml file for configuration')
    args = parser.parse_args()
    return args


# ==========================================
# 🚀 算法组件注册表 (工厂模式)
# ==========================================
CLIENT_REGISTRY = {
    "FedAvg": FedClient,
    "Scaffold": ScaffoldClient,
    "FedProx": FedProxClient,
    "FedNova": FedNovaClient,
    "FedAvgGRO": GROClient,
    "ScaffoldGRO": ScaffoldGROClient,
    "FedProxGRO": FedProxGROClient,
    "FedNovaGRO": FedNovaGROClient
}

SERVER_REGISTRY = {
    "FedAvg": FedServer,
    "FedProx": FedServer,
    "FedProxGRO": FedServer,
    "Scaffold": ScaffoldServer,
    "ScaffoldGRO": ScaffoldServer,
    "FedNova": FedNovaServer,
    "FedNovaGRO": FedNovaServer,
    "FedAvgGRO": GROServer
}


def fed_run():
    args = fed_args()
    with open(args.config, "r") as yaml_file:
        try:
            config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)

    fed_algo = config["client"]["fed_algo"]
    dataset = config["system"]["dataset"]
    model_name = config["system"]["model"]

    assert fed_algo in CLIENT_REGISTRY, "The federated learning algorithm is not supported"
    assert dataset in ['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN', 'CIFAR100', 'ImageNet',
                       'LC25000','HAM10000'], "The dataset is not supported"
    assert model_name in ["LeNet", 'CNN_MNIST', 'AlexCifarNet', "ResNet18", 'VGG11', "CNN",
                          'ResNet18_HAM10000'], "The model is not supported"

    np.random.seed(config["system"]["i_seed"])
    torch.manual_seed(config["system"]["i_seed"])
    random.seed(config["system"]["i_seed"])

    recorder = Recorder()
    trainset_config, testset = divide_data_dirichlet(
        num_client=config["system"]["num_client"],
        alpha=config["system"]["dirichlet_alpha"],
        dataset_name=dataset,
        i_seed=config["system"]["i_seed"]
    )

    max_acc = 0
    max_f1 = 0.0
    max_auc = 0.0
    client_dict = {}

    # ==========================================
    # ✅ 优雅地初始化 Clients (已彻底移除 Beta)
    # ==========================================
    client_class = CLIENT_REGISTRY[fed_algo]
    base_kwargs = {
        "dataset_id": dataset,
        "epoch": config["client"]["num_local_epoch"],
        "model_name": model_name,
        "lr": config["client"]["lr"],
        "batch_size": config["client"]["batch_size"],
        "momentum": config["client"]["momentum"]
    }

    for client_id in trainset_config['users']:
        # 动态实例化对应的 Client
        client_dict[client_id] = client_class(name=client_id, **base_kwargs)
        client_dict[client_id].load_trainset(trainset_config['user_data'][client_id])

    # ==========================================
    # ✅ 优雅地初始化 Server
    # ==========================================
    server_class = SERVER_REGISTRY[fed_algo]
    fed_server = server_class(
        client_list=trainset_config['users'],
        dataset_id=dataset,
        model_name=model_name,
        batch_size=config["client"]["batch_size"]
    )
    fed_server.load_testset(testset)

    global_state_dict = fed_server.state_dict()
    scv_state = fed_server.scv.state_dict() if "Scaffold" in fed_algo else None

    # ==========================================
    # ✅ 主训练循环
    # ==========================================
    max_rounds = config["system"]["num_round"]
    pbar = tqdm(range(max_rounds))

    for global_round in pbar:

        for client_id in trainset_config['users']:

            client = client_dict[client_id]

            # ==========================================
            # 1. Dynamic Update
            # ==========================================
            if "Scaffold" in fed_algo:
                client.update(
                    global_round,
                    max_rounds,
                    global_state_dict,
                    scv_state
                )
            else:
                client.update(
                    global_round,
                    max_rounds,
                    global_state_dict
                )

            # ==========================================
            # 2. Dynamic Train
            # ==========================================
            if fed_algo == 'FedAvgGRO':

                global_avg_grad = getattr(
                    fed_server,
                    "global_avg_grad",
                    None
                )

                results = client.train(
                    global_avg_grad=global_avg_grad
                )

            else:
                results = client.train()

            # ==========================================
            # 3. Receive Client Updates
            # ==========================================
            fed_server.rec(client.name, *results)

        # ==========================================
        # 4. Server Aggregation
        # ==========================================
        fed_server.select_clients()

        agg_results = fed_server.agg()

        global_state_dict = agg_results[0]
        avg_loss = agg_results[1]

        if "Scaffold" in fed_algo:
            scv_state = agg_results[3]

        # ==========================================
        # 5. Test
        # ==========================================
        metrics = fed_server.test()

        fed_server.flush()

        acc = metrics["Acc"]
        f1 = metrics["F1"]
        recall = metrics["Recall"]
        precision = metrics["Precision"]
        auc = metrics["AUC"]

        # ==========================================
        # 6. Record Metrics
        # ==========================================
        recorder.res['server']['Acc'].append(acc)
        recorder.res['server']['F1'].append(f1)
        recorder.res['server']['Recall'].append(recall)
        recorder.res['server']['Precision'].append(precision)
        recorder.res['server']['AUC'].append(auc)

        recorder.res['server']['train_loss'].append(avg_loss)

        max_acc = max(max_acc, acc)
        max_f1 = max(max_f1, f1)
        max_auc = max(max_auc, auc)

        recorder.res['server']['max_Acc'] = max_acc
        recorder.res['server']['max_F1'] = max_f1
        recorder.res['server']['max_AUC'] = max_auc

        # ==========================================
        # 7. Progress Bar
        # ==========================================
        pbar.set_description(
            f'Round: {global_round} | '
            f'Loss: {avg_loss:.4f} | '
            f'Acc: {acc:.4f} | '
            f'F1: {f1:.4f} | '
            f'AUC: {auc:.4f} | '
            f'Max Acc: {max_acc:.4f}'
        )

    # ==========================================
    # ✅ 保存结果
    # ==========================================
    alpha = config["system"]["dirichlet_alpha"]
    lr = config["client"]["lr"]
    random_seed = config["system"]["i_seed"]
    algo_clean = fed_algo.replace("GRO", "")

    file_name = f"['{fed_algo}','{dataset}','alpha{alpha}','lr{lr}','seed{random_seed}']"

    # 严格保留你要求的路径结构
    save_dir = os.path.join(
        config["system"]["res_root"],
        dataset,
        f"alpha{alpha}",
        f"lr{lr}",
        algo_clean
    )

    # save_dir = os.path.join(config["system"]["res_root"])

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file_name)

    with open(file_path, "w") as jsfile:
        json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)


if __name__ == "__main__":
    fed_run()