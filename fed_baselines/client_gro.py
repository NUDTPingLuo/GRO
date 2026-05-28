import copy
import torch
import math
from fed_baselines.client_base import FedClient
from utils.models import *
from torch.utils.data import DataLoader


class GROClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name, lr, batch_size, momentum):
        super().__init__(name, epoch, dataset_id, model_name, lr, batch_size, momentum)

    def parameters_to_vector(self, parameters):
        """使用列表推导式优雅地展平梯度，榨干 GPU 效率"""
        vec = [param.grad.view(-1) for param in parameters if param.grad is not None]
        return torch.cat(vec) if len(vec) > 0 else torch.tensor([])

    def vector_to_parameters(self, vec, parameters):
        """将一维巨型张量写回模型的 .grad 中"""
        pointer = 0
        for param in parameters:
            if param.grad is not None:
                num_param = param.numel()
                param.grad.data.copy_(vec[pointer:pointer + num_param].view_as(param.grad).data)
                pointer += num_param

    def train(self, global_avg_grad=None):
        self.model.to(self._device)

        # 关闭 PyTorch 优化器自带动量，由我们手动接管
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=0.0)
        loss_func = torch.nn.CrossEntropyLoss()

        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True,
                                  num_workers=8, pin_memory=True, persistent_workers=False)

        initial_state = copy.deepcopy(self.model.state_dict())

        # EMA 探路针与手动动量缓冲区
        short_term_vec = None
        momentum_vec = None

        ema_alpha = 0.9

        step_count = 0
        loss_val = 0.0

        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                b_x, b_y = x.to(self._device), y.to(self._device)

                self.model.train()
                optimizer.zero_grad()
                output = self.model(b_x)
                loss = loss_func(output, b_y.long())
                loss.backward()
                loss_val = loss.item()

                # 提取当前的“原始瞬时梯度”
                curr_grad_vec = self.parameters_to_vector(self.model.parameters())

                if step_count == 0:
                    momentum_vec = curr_grad_vec.clone()
                    short_term_vec = curr_grad_vec.clone()
                    optimizer.step()
                else:
                    # 1. 在纯数据流形上进行正常的动量累加
                    momentum_vec = self._momentum * momentum_vec + curr_grad_vec

                    # ========================================================
                    # 2. 【核心魔法：刚性镜像反射 (Rigid Geometric Reflection)】
                    # ========================================================
                    dot_product = torch.dot(momentum_vec, short_term_vec)
                    norm_short_sq = torch.dot(short_term_vec, short_term_vec)
                    norm_mom = torch.norm(momentum_vec)
                    norm_short = torch.sqrt(norm_short_sq)

                    if dot_product < 0 and norm_short > 1e-8 and norm_mom > 1e-8:
                        # 放弃自适应衰减，只要发生冲突（点积 < 0），直接进行 100% 镜像反向
                        # 物理意义：垂直于探路针的探索分量 100% 保留，平行于探路针的背离分量直接乘 -1 反转。
                        gamma = 2.0

                        # # ---> DEBUG 打印输出 <---
                        # print(f"🐝 [{self.name} | Epoch {epoch} Step {step}] 触发刚性反射! (平行向量完全反向, Gamma=2.0)")

                        momentum_vec -= gamma * (dot_product / norm_short_sq) * short_term_vec

                    # 3. 将纠偏后的步伐写回模型并供 SGD 消费
                    self.vector_to_parameters(momentum_vec, self.model.parameters())

                    # 4. 更新探路针 (追踪真实的综合运动轨迹)
                    short_term_vec.mul_(ema_alpha).add_(momentum_vec, alpha=1 - ema_alpha)

                    # 5. 执行参数更新
                    optimizer.step()

                step_count += 1

        # 提取并返回本轮本地训练的 Delta W
        final_state = self.model.state_dict()
        final_grad = {}
        for name in initial_state:
            if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
                final_grad[name] = final_state[name]
            else:
                final_grad[name] = initial_state[name].to(self._device) - final_state[name]

        return final_grad, self.n_data, loss_val