import copy
import torch
from fed_baselines.client_base import FedClient
from utils.models import *
from torch.utils.data import DataLoader


class FedNovaGROClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name, lr, batch_size, momentum):
        super().__init__(name, epoch, dataset_id, model_name, lr, batch_size, momentum)
        self.rho = 0.9

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
                    optimizer.step()
                else:
                    momentum_vec = self._momentum * momentum_vec + curr_grad_vec

                    # 引入感知显著性的自适应反射引擎
                    dot_product = torch.dot(momentum_vec, short_term_vec)
                    norm_short_sq = torch.dot(short_term_vec, short_term_vec)
                    norm_mom = torch.norm(momentum_vec)
                    norm_short = torch.sqrt(norm_short_sq)

                    if dot_product < 0 and norm_short > 1e-8 and norm_mom > 1e-8:
                        # cos_sim = dot_product / (norm_mom * norm_short)
                        # cos_sim = torch.clamp(cos_sim, -1.0, 0.0)
                        #
                        # gamma = 1.0 + torch.abs(cos_sim)
                        gamma = 2.0
                        momentum_vec -= gamma * (dot_product / norm_short_sq) * short_term_vec

                    self.vector_to_parameters(momentum_vec, self.model.parameters())
                    short_term_vec.mul_(ema_alpha).add_(momentum_vec, alpha=1 - ema_alpha)
                    optimizer.step()

                step_count += 1

        state_dict = self.model.state_dict()
        norm_grad = {}
        coeff = float(step_count) if step_count > 0 else 1.0

        for key in initial_state:
            # 放行所有张量（包含 BN）用于生成 norm_grad
            delta_w = initial_state[key].to(self._device) - state_dict[key]
            norm_grad[key] = delta_w / coeff

        return state_dict, self.n_data, loss_val, coeff, norm_grad