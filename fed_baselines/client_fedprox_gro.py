import copy
import torch
from fed_baselines.client_base import FedClient
from utils.models import *
from torch.utils.data import DataLoader


class FedProxGROClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name, lr, batch_size, momentum, mu=0.1):
        super().__init__(name, epoch, dataset_id, model_name, lr, batch_size, momentum)
        self.mu = mu

    def parameters_to_vector(self, parameters):
        vec = [param.grad.view(-1) for param in parameters if param.grad is not None]
        return torch.cat(vec) if len(vec) > 0 else torch.tensor([])

    def weights_to_vector(self, parameters):
        vec = [param.data.view(-1) for param in parameters]
        return torch.cat(vec) if len(vec) > 0 else torch.tensor([])

    def vector_to_parameters(self, vec, parameters):
        pointer = 0
        for param in parameters:
            if param.grad is not None:
                num_param = param.numel()
                param.grad.data.copy_(vec[pointer:pointer + num_param].view_as(param.grad).data)
                pointer += num_param

    def train(self, global_avg_grad=None):
        self.model.to(self._device)
        global_weights_vec = self.weights_to_vector(self.model.parameters()).clone()

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

                    prox_grad_vec = self.mu * (self.weights_to_vector(self.model.parameters()) - global_weights_vec)
                    self.vector_to_parameters(momentum_vec + prox_grad_vec, self.model.parameters())
                    optimizer.step()
                else:
                    # 1. 动量累加
                    momentum_vec = self._momentum * momentum_vec + curr_grad_vec

                    # 2. 自适应反射
                    dot_product = torch.dot(momentum_vec, short_term_vec)
                    norm_short_sq = torch.dot(short_term_vec, short_term_vec)
                    norm_mom = torch.norm(momentum_vec)
                    norm_short = torch.sqrt(norm_short_sq)

                    if dot_product < 0 and norm_short > 1e-8 and norm_mom > 1e-8:
                        # cos_sim = dot_product / (norm_mom * norm_short)
                        # cos_sim = torch.clamp(cos_sim, -1.0, 0.0)
                        # gamma = 1.0 + torch.abs(cos_sim)
                        gamma = 2.0
                        momentum_vec -= gamma * (dot_product / norm_short_sq) * short_term_vec

                    # 3. 计算 FedProx 的先验拉力
                    current_weights_vec = self.weights_to_vector(self.model.parameters())
                    prox_grad_vec = self.mu * (current_weights_vec - global_weights_vec)

                    # 4. 探路针追踪【数据动量 + Proximal 拉力】
                    total_effective_trajectory = momentum_vec + prox_grad_vec
                    short_term_vec.mul_(ema_alpha).add_(total_effective_trajectory, alpha=1 - ema_alpha)

                    # 5. 最终梯度供优化器消费
                    final_grad_to_apply = momentum_vec + prox_grad_vec
                    self.vector_to_parameters(final_grad_to_apply, self.model.parameters())
                    optimizer.step()

                step_count += 1

        return self.model.state_dict(), self.n_data, loss_val