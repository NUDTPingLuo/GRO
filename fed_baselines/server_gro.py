from fed_baselines.server_base import FedServer
import copy
import torch

class GROServer(FedServer):
    def __init__(self, client_list, dataset_id, model_name, batch_size, select_ratio=0.5):
        super().__init__(client_list, dataset_id, model_name, batch_size)
        self.dataset_id = dataset_id

    def agg(self):
        """
        聚合客户端上传的 final_grad (Delta W)。
        """
        client_num = len(self.selected_clients)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0

        self.model.to(self._device)
        global_model = self.model.state_dict()
        new_model = copy.deepcopy(global_model)
        avg_loss = 0

        grad_sum = {}

        # 聚合客户端上传的梯度 (Delta W)
        for i, name in enumerate(self.selected_clients):
            client_grad = self.client_state[name]["final_grad"]
            weight = self.client_n_data[name] / self.n_data

            for key in client_grad:
                if i == 0:
                    grad_sum[key] = weight * client_grad[key].to(self._device)
                else:
                    grad_sum[key] += weight * client_grad[key].to(self._device)

            avg_loss += self.client_loss[name] * weight

        # 用聚合后的平均更新量更新全局模型
        for key in global_model:
            if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
                new_model[key] = grad_sum[key]
            else:
                new_model[key] = global_model[key] - grad_sum[key]

        self.model.load_state_dict(new_model)
        self.round += 1

        return new_model, avg_loss, self.n_data

    def rec(self, name, final_grad, n_data, loss):
        self.n_data += n_data
        self.client_state[name] = {"final_grad": final_grad}
        self.client_n_data[name] = n_data
        self.client_loss[name] = loss

    def flush(self):
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}