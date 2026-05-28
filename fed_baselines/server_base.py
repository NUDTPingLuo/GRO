from utils.models import *
import torch
from torch.utils.data import DataLoader
from utils.fed_utils import assign_dataset, init_model
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F



class FedServer(object):
    def __init__(self, client_list, dataset_id, model_name, batch_size):
        """
        Initialize the server for federated learning.
        :param client_list: List of the connected clients in networks
        :param dataset_id: Dataset name for the application scenario
        :param model_name: Machine learning model name for the application scenario
        """
        # Initialize the dict and list for system settings
        self.client_state = {}
        self.client_loss = {}
        self.client_n_data = {}
        self.selected_clients = []
        # self._lr = 0.01
        # batch size for testing
        self._batch_size = batch_size
        self.client_list = client_list

        # Initialize the test dataset
        self.testset = None

        # Initialize the hyperparameter for federated learning in the server
        self.round = 0
        self.n_data = 0
        self._dataset_id = dataset_id

        # Testing on GPU
        gpu = 0
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

        # Initialize the global machine learning model
        self._num_class, self._image_dim, self._image_channel = assign_dataset(dataset_id)
        self.model_name = model_name
        self.model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)

    def load_testset(self, testset):
        """
        Server loads the test dataset.
        :param data: Dataset for testing.
        """
        self.testset = testset

    def state_dict(self):
        """
        Server returns global model dict.
        :return: Global model dict
        """
        return self.model.state_dict()

    def test(self):
        """
        Server tests the global model.
        Returns:
            Acc / F1 / Recall / Precision / AUC
        """
        test_loader = DataLoader(
            self.testset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=False
        )
        self.model.to(self._device)
        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self._device)
                y = y.to(self._device)
                outputs = self.model(x)
                # =========================
                # 防止 NaN / Inf
                # =========================
                outputs = torch.nan_to_num(
                    outputs,
                    nan=0.0,
                    posinf=1e6,
                    neginf=-1e6
                )
                probs = F.softmax(outputs, dim=1)
                probs = torch.nan_to_num(
                    probs,
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0
                )
                _, predicted = torch.max(probs, 1)
                all_labels.extend(y.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        # =========================
        # Metrics
        # =========================
        acc = (all_preds == all_labels).mean()
        f1 = f1_score(
            all_labels,
            all_preds,
            average='macro',
            zero_division=0
        )
        recall = recall_score(
            all_labels,
            all_preds,
            average='macro',
            zero_division=0
        )
        precision = precision_score(
            all_labels,
            all_preds,
            average='macro',
            zero_division=0
        )
        # =========================
        # AUC
        # =========================
        try:
            if len(np.unique(all_labels)) == 2:
                auc = roc_auc_score(
                    all_labels,
                    all_probs[:, 1]
                )
            else:
                auc = roc_auc_score(
                    all_labels,
                    all_probs,
                    multi_class='ovr'
                )
        except Exception as e:
            print(f"[Warning] AUC computation failed: {e}")
            auc = 0.0
        return {
            "Acc": float(acc),
            "F1": float(f1),
            "Recall": float(recall),
            "Precision": float(precision),
            "AUC": float(auc)
        }

    def select_clients(self, connection_ratio=1):
        """
        Server selects a fraction of clients.
        :param connection_ratio: connection ratio in the clients
        """
        # select a fraction of clients
        self.selected_clients = []
        self.n_data = 0
        for client_id in self.client_list:
            b = np.random.binomial(np.ones(1).astype(int), connection_ratio)
            if b:
                self.selected_clients.append(client_id)
                self.n_data += self.client_n_data[client_id]

    def agg(self):
        """
        Server aggregates models from connected clients.
        :return: model_state: Updated global model after aggregation
        :return: avg_loss: Averaged loss value
        :return: n_data: Number of the local data points
        """
        client_num = len(self.selected_clients)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0

        # Initialize a model for aggregation
        model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        model_state = model.state_dict()
        avg_loss = 0

        # Aggregate the local updated models from selected clients
        for i, name in enumerate(self.selected_clients):
            if name not in self.client_state:
                continue
            for key in self.client_state[name]:
                if i == 0:
                    model_state[key] = self.client_state[name][key] * self.client_n_data[name] / self.n_data
                else:
                    model_state[key] = model_state[key] + self.client_state[name][key] * self.client_n_data[
                        name] / self.n_data

            avg_loss = avg_loss + self.client_loss[name] * self.client_n_data[name] / self.n_data

        # Server load the aggregated model as the global model
        self.model.load_state_dict(model_state)
        self.round = self.round + 1
        n_data = self.n_data

        return model_state, avg_loss, n_data

    def rec(self, name, state_dict, n_data, loss):
        """
        Server receives the local updates from the connected client k.
        :param name: Name of client k
        :param state_dict: Model dict from the client k
        :param n_data: Number of local data points in the client k
        :param loss: Loss of local training in the client k
        """
        self.n_data = self.n_data + n_data
        self.client_state[name] = {}
        self.client_n_data[name] = {}

        self.client_state[name].update(state_dict)
        self.client_n_data[name] = n_data
        self.client_loss[name] = {}
        self.client_loss[name] = loss

    def flush(self):
        """
        Flushing the client information in the server
        """
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}
