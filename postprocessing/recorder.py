import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from json import JSONEncoder
import pickle
import re
from matplotlib import rcParams
import seaborn as sns
import os
from collections import defaultdict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

json_types = (list, dict, str, int, float, bool, type(None))


class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct


class Recorder(object):
    def __init__(self):
        self.res_list = []

        # 预设全指标标准模版：仅保留 server 侧核心指标
        self.res = {
            'server': {
                'Acc': [],
                'F1': [],
                'Recall': [],
                'Precision': [],
                'AUC': [],
                'train_loss': [],
                'max_Acc': None,
                'max_F1': None,
                'max_AUC': None
            }
        }

    def load(self, filename, label):
        """
        加载结果文件，并对 server 侧的其余核心变量进行稳健提取与兜底
        """
        with open(filename) as json_file:
            res = json.load(json_file, object_hook=as_python_object)

        if 'server' not in res:
            res['server'] = {}

        all_server_keys = ['Acc', 'F1', 'Recall', 'Precision', 'AUC', 'train_loss', 'max_Acc', 'max_F1', 'max_AUC']
        for key in all_server_keys:
            if key not in res['server']:
                if key.startswith('max_'):
                    base_key = key.replace('max_', '')
                    if base_key in res['server'] and isinstance(res['server'][base_key], list) and len(
                            res['server'][base_key]) > 0:
                        res['server'][key] = max(res['server'][base_key])
                    else:
                        res['server'][key] = None
                else:
                    res['server'][key] = []

        self.res_list.append((res, label))

    def _parse_label(self, label):
        """
        内部辅助函数：精准解析无 beta 的标准 5 段式文件名标签
        格式如: ['FedAvg','CIFAR10','alpha0.01','lr0.01','seed0']
        """
        matches = re.findall(r"'([^']+)'", label)

        info = {
            'Algorithm': matches[0] if len(matches) > 0 else 'Unknown',
            'Dataset': matches[1] if len(matches) > 1 else 'Unknown',
            'Dirichlet': matches[2] if len(matches) > 2 else 'Unknown',
            'lr': matches[3] if len(matches) > 3 else 'Unknown',
            'seed': matches[4] if len(matches) > 4 else 'seed0'
        }
        return info

    def plot(self, figsize=(6, 5)):
        """
        Plot testing accuracy (mean ± range) across different seeds.
        """
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42
        plt.rcParams['font.family'] = 'DejaVu Sans'

        sns.set_style("white")
        color_palette = sns.color_palette("colorblind")

        base_color_map = {
            'FedAvg': color_palette[3],
            'Scaffold': color_palette[0],
            'FedProx': color_palette[9],
            'FedNova': color_palette[1],
            'FedAvgGRO': color_palette[2],
            'ScaffoldGRO': color_palette[2],
            'FedNovaGRO': color_palette[2],
            'FedProxGRO': color_palette[2],
        }

        line_style_map = {
            'FedAvg': '-',
            'Scaffold': '--',
            'FedProx': '-.',
            'FedNova': ':',
            'FedAvgGRO': '-',
            'ScaffoldGRO': '-',
            'FedNovaGRO': '-',
            'FedProxGRO': '-',
        }

        grouped_results = defaultdict(list)
        for res, label in self.res_list:
            info = self._parse_label(label)
            key = (info['Algorithm'], info['Dataset'], info['Dirichlet'])
            grouped_results[key].append(np.array(res['server']['Acc']))

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        Dataset, Dirichlet, lr = 'Unknown', 'Unknown', 'Unknown'

        for (Algorithm, Dataset, Dirichlet), acc_list in grouped_results.items():
            if len(acc_list) == 0:
                continue
            min_len = min(len(a) for a in acc_list)
            acc_array = np.array([a[:min_len] for a in acc_list])

            mean_acc = acc_array.mean(axis=0)
            max_acc = acc_array.max(axis=0)
            min_acc = acc_array.min(axis=0)

            color = base_color_map.get(Algorithm, None)
            linestyle = line_style_map.get(Algorithm, '-')
            alpha_value = 0.4 if "GRO" in Algorithm else 0.2
            linewidth_value = 3.0 if "GRO" in Algorithm else 2.0

            ax.plot(mean_acc, label=Algorithm, alpha=1, linewidth=linewidth_value, color=color, linestyle=linestyle)
            ax.fill_between(range(min_len), min_acc, max_acc, color=color, alpha=alpha_value)

        # 创建 inset axes（局部放大图）
        inset_ax = inset_axes(ax, width="30%", height="30%", bbox_to_anchor=(-0.15, -0.55, 1, 1),
                              bbox_transform=ax.transAxes, borderpad=0)
        start_i, end_i = 90, 100

        for (Algorithm, Dataset, Dirichlet), acc_list in grouped_results.items():
            if len(acc_list) == 0:
                continue
            min_len = min(len(a) for a in acc_list)
            acc_array = np.array([a[:min_len] for a in acc_list])

            actual_end = min(end_i, min_len)
            actual_start = min(start_i, actual_end)

            if actual_end - actual_start > 0:
                mean_acc = acc_array.mean(axis=0)[actual_start:actual_end]
                color = base_color_map.get(Algorithm, None)
                linestyle = line_style_map.get(Algorithm, '-')
                linewidth_value = 3.0 if "GRO" in Algorithm else 2.0
                x_range = np.arange(actual_start, actual_end)

                inset_ax.plot(x_range, mean_acc, color=color, linestyle=linestyle, linewidth=linewidth_value)

        inset_ax.tick_params(labelsize=8)
        inset_ax.grid(alpha=0.3)

        try:
            ax.indicate_inset_zoom(inset_ax, edgecolor="black", linewidth=1.2)
        except Exception as e:
            print("⚠️ inset zoom connection unavailable:", e)

        ax.set_xlabel('Epochs', size=12)
        ax.set_ylabel('Testing Accuracy', size=12)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(alpha=0.3)

        for res, label in self.res_list:
            info = self._parse_label(label)
            lr = info['lr']
            break

        # ✨ 改进1：动态组合所有画在当前图表上的算法名称
        all_algs = sorted(list(set(k[0] for k in grouped_results.keys())))
        algs_str = "_".join(all_algs)

        save_dir = os.path.join('..', 'plot', Dataset, Dirichlet, lr)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{Dataset}_{Dirichlet}_{lr}_{algs_str}.pdf")

        # 🌟 修复2：调整顺序，先保存图片，再 show 展现
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_mean(self, figsize=(6, 5)):
        """
        Plot only the mean testing accuracy for all methods on the same figure.
        """
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42
        plt.rcParams['font.family'] = 'DejaVu Sans'

        sns.set_style("white")
        color_palette = sns.color_palette("colorblind")

        base_color_map = {
            'FedAvg': color_palette[2],
            'Scaffold': color_palette[0],
            'FedProx': color_palette[9],
            'FedNova': color_palette[1],
        }

        gro_color_map = {
            'FedAvgGRO': base_color_map['FedAvg'],
            'ScaffoldGRO': base_color_map['Scaffold'],
            'FedNovaGRO': base_color_map['FedNova'],
            'FedProxGRO': base_color_map['FedProx'],
        }

        grouped_results = defaultdict(list)
        meta_info = {'Dataset': 'Unknown', 'Dirichlet': 'Unknown', 'lr': 'Unknown'}

        for res, label in self.res_list:
            info = self._parse_label(label)
            key = (info['Algorithm'], info['Dataset'], info['Dirichlet'])
            grouped_results[key].append(np.array(res['server']['Acc']))

            meta_info['Dataset'] = info['Dataset']
            meta_info['Dirichlet'] = info['Dirichlet']
            meta_info['lr'] = info['lr']

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        for (Algorithm, Dataset, Dirichlet), acc_list in grouped_results.items():
            if len(acc_list) == 0:
                continue
            min_len = min(len(a) for a in acc_list)
            acc_array = np.array([a[:min_len] for a in acc_list])
            mean_acc = acc_array.mean(axis=0)

            if 'GRO' in Algorithm:
                color = gro_color_map.get(Algorithm, 'black')
                linestyle = '-'
            else:
                color = base_color_map.get(Algorithm, 'black')
                linestyle = '--'

            ax.plot(mean_acc, label=Algorithm, color=color, linestyle=linestyle, linewidth=2.5)

        ax.set_xlabel('Epochs', size=12)
        ax.set_ylabel('Testing Accuracy', size=12)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(alpha=0.3)
        ax.legend(prop={'size': 12}, loc='lower right')
        plt.suptitle(meta_info['Dataset'], size=16, fontweight='bold')

        # ✨ 改进1：动态组合所有算法名称进文件名中
        all_algs = sorted(list(set(k[0] for k in grouped_results.keys())))
        algs_str = "_".join(all_algs)

        save_dir = os.path.join('..', 'plot', meta_info['Dataset'], meta_info['Dirichlet'], meta_info['lr'])
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir,
                                 f"{meta_info['Dataset']}_{meta_info['Dirichlet']}_{meta_info['lr']}_{algs_str}_mean.pdf")

        # 🌟 修复2：调整顺序，先保存图片，再 show 展现
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()