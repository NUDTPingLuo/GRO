from utils.fed_utils import init_model
from fed_baselines.server_base import FedServer
import copy
import torch


class ScaffoldServer(FedServer):
    def __init__(self, client_list, dataset_id, model_name, batch_size):
        super().__init__(client_list, dataset_id, model_name, batch_size)
        # server control variate (即全局 c)
        self.scv = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        scv_state = self.scv.state_dict()
        for key in scv_state:
            scv_state[key] = torch.zeros_like(scv_state[key])
        self.scv.load_state_dict(scv_state)
        # 记录客户端传上来的控制变量增量 (Delta c_i)
        self.client_ccv_state = {}

    def agg(self):
        """
        Server aggregates normalized models from connected clients using SCAFFOLD
        :return: Updated global model after aggregation, Averaged loss value, Number of the local data points, server control variate
        """
        client_num = len(self.selected_clients)

        # 修复：提前退出时必须返回 4 个参数，补上 self.scv.state_dict()
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0, self.scv.state_dict()

        self.scv.to(self._device)
        self.model.to(self._device)

        # 获取上一轮的全局模型和全局控制变量 c
        model_state = self.model.state_dict()
        scv_state = self.scv.state_dict()

        # 初始化一个全是 0 的字典，专门用来累加本轮客户端上传的 Delta c
        delta_c_sum = {key: torch.zeros_like(val).to(self._device) for key, val in scv_state.items()}

        avg_loss = 0

        for i, name in enumerate(self.selected_clients):
            weight = self.client_n_data[name] / self.n_data
            if name not in self.client_state:
                continue

            for key in self.client_state[name]:
                # 1. 聚合模型权重
                if i == 0:
                    model_state[key] = self.client_state[name][key] * weight
                else:
                    model_state[key] = model_state[key] + self.client_state[name][key] * weight

                # 2. 累加客户端传上来的增量 Delta c_i (此时存在 self.client_ccv_state 中)
                delta_c_sum[key] = delta_c_sum[key] + self.client_ccv_state[name][key].to(self._device) * weight

            avg_loss = avg_loss + self.client_loss[name] * weight

        # 3. 将累加好的加权增量更新到全局控制变量 c 上
        # 公式: c_new = c_old + sum(weight * delta_c_i)
        for key in scv_state:
            # 【高级提示】如果在你的实验中是 Partial Participation (比如10个client只选4个参与)
            # 严格按照 SCAFFOLD 论文，这里的增量还应该乘以一个 (选中的client数 / 总client数) 的比例
            # 即: scv_state[key] = scv_state[key] + delta_c_sum[key] * (client_num / total_clients)
            # 这里先按照你的原生逻辑直接加上 weight * delta_c_sum
            scv_state[key] = scv_state[key] + delta_c_sum[key]

        self.model.load_state_dict(model_state)
        self.scv.load_state_dict(scv_state)
        self.round = self.round + 1

        return model_state, avg_loss, self.n_data, scv_state

    def rec(self, name, state_dict, n_data, loss, ccv_state):
        """
        Server receives the local updates from the connected client k.
        """
        self.n_data = self.n_data + n_data
        self.client_state[name] = {}
        self.client_n_data[name] = {}
        self.client_ccv_state[name] = {}

        self.client_state[name].update(state_dict)
        self.client_n_data[name] = n_data
        self.client_loss[name] = {}
        self.client_loss[name] = loss
        self.client_ccv_state[name].update(ccv_state)

    def flush(self):
        """
        Flushing the client information in the server
        """
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}
        self.client_ccv_state = {}