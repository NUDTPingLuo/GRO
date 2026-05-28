import copy
import torch
import math
from fed_baselines.client_base import FedClient
from utils.models import *
from torch.utils.data import DataLoader
from utils.fed_utils import init_model


class ScaffoldGROClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name, lr, batch_size, momentum):
        super().__init__(name, epoch, dataset_id, model_name, lr, batch_size, momentum)

        self.scv = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        self.ccv = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        self.init_lr = self._lr

        ccv_state = self.ccv.state_dict()
        for key in ccv_state:
            ccv_state[key] = torch.zeros_like(ccv_state[key])
        self.ccv.load_state_dict(ccv_state)

    def update(self, global_round, max_rounds, model_state_dict, scv_state):
        eta_min = 0.0
        self._lr = eta_min + 0.5 * (self.init_lr - eta_min) * (
                1 + math.cos(math.pi * global_round / max_rounds)
        )

        self.model.load_state_dict(model_state_dict)
        self.scv.load_state_dict(scv_state)

    def parameters_to_vector(self, parameters):
        vec = [param.grad.view(-1) for param in parameters if param.grad is not None]
        return torch.cat(vec) if len(vec) > 0 else torch.tensor([])

    def vector_to_parameters(self, vec, parameters):
        pointer = 0
        for param in parameters:
            if param.grad is not None:
                num_param = param.numel()
                param.grad.data.copy_(vec[pointer:pointer + num_param].view_as(param.grad).data)
                pointer += num_param

    def train(self):
        self.model.to(self._device)
        self.scv.to(self._device)
        self.ccv.to(self._device)

        scv_state = self.scv.state_dict()
        ccv_state = self.ccv.state_dict()
        initial_state = copy.deepcopy(self.model.state_dict())

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=0.0)
        loss_func = torch.nn.CrossEntropyLoss()

        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True, num_workers=8,
                                  pin_memory=True, persistent_workers=False)

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

                curr_grad_vec = self.parameters_to_vector(self.model.parameters())

                if step_count == 0:
                    momentum_vec = curr_grad_vec.clone()
                    short_term_vec = curr_grad_vec.clone()
                    self.vector_to_parameters(momentum_vec, self.model.parameters())
                else:
                    # 1. 纯数据流形上的动量累加
                    momentum_vec = self._momentum * momentum_vec + curr_grad_vec

                    # 2. 基于余弦相似度的自适应反射 (Significance-Aware Adaptive Reflection)
                    dot_product = torch.dot(momentum_vec, short_term_vec)
                    norm_short_sq = torch.dot(short_term_vec, short_term_vec)
                    norm_mom = torch.norm(momentum_vec)
                    norm_short = torch.sqrt(norm_short_sq)

                    if dot_product < 0 and norm_short > 1e-8 and norm_mom > 1e-8:
                        # cos_sim = dot_product / (norm_mom * norm_short)
                        # cos_sim = torch.clamp(cos_sim, -1.0, 0.0)
                        #
                        # # 冲突越严重，gamma越接近2 (强反射)；冲突越轻微，gamma越接近1 (温和归零)
                        # gamma = 1.0 + torch.abs(cos_sim)
                        gamma = 2.0
                        momentum_vec -= gamma * (dot_product / norm_short_sq) * short_term_vec

                    self.vector_to_parameters(momentum_vec, self.model.parameters())

                    # 3. 提取 SCAFFOLD 的宏观修正向量，进行轨迹对齐
                    correction_vec = []
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            diff = scv_state[name].to(self._device) - ccv_state[name].to(self._device)
                            correction_vec.append(diff.view(-1))
                    correction_vec = torch.cat(correction_vec) if len(correction_vec) > 0 else torch.tensor([]).to(
                        self._device)

                    # 4. 探路针追踪真实的综合运动轨迹
                    total_effective_trajectory = momentum_vec + correction_vec
                    short_term_vec.mul_(ema_alpha).add_(total_effective_trajectory, alpha=1 - ema_alpha)

                # 5. 在优化器执行前，强行追加 SCAFFOLD 修正
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            param.grad.data.add_(scv_state[name].to(self._device) - ccv_state[name].to(self._device))

                optimizer.step()
                step_count += 1

        state_dict = self.model.state_dict()
        new_ccv_state = copy.deepcopy(self.ccv.state_dict())
        delta_ccv_state = {}

        for key in initial_state:
            if key in new_ccv_state:
                if "running" in key or "num_batches" in key:
                    delta_ccv_state[key] = torch.zeros_like(ccv_state[key])
                else:
                    delta_w = initial_state[key].to(self._device) - state_dict[key]
                    new_ccv_state[key] = ccv_state[key] - scv_state[key] + delta_w / (step_count * self._lr)
                    delta_ccv_state[key] = new_ccv_state[key] - ccv_state[key]

        self.ccv.load_state_dict(new_ccv_state)

        return state_dict, self.n_data, loss_val, delta_ccv_state