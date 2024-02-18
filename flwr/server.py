# from collections import OrderedDict
# from typing import Dict, List, Optional, Tuple
# import flwr as fl
# from flwr.common import Metrics
# import numpy as np

# # Define metric aggregation function
# # def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
# #     # Multiply accuracy of each client by number of examples used
# #     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
# #     examples = [num_examples for num_examples, _ in metrics]
# #     print("accuracies = ", accuracies)
# #     print("examples = ", examples)
# #     # Aggregate and return custom metric (weighted average)
# #     return {"accuracy": sum(accuracies) / sum(examples)}

# # Define strategy
# #strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# NUM_CLIENTS = 100

# def get_parameters(net) -> List[np.ndarray]:
#     return [val.cpu().numpy() for _, val in net.state_dict().items()]

# def evaluate(
#     server_round: int,
#     parameters: fl.common.NDArrays,
#     config: Dict[str, fl.common.Scalar],
# ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
#     net = Net().to(DEVICE)
#     valloader = valloaders[0]
#     set_parameters(net, parameters)  # Update model with the latest parameters
#     loss, accuracy = test(net, valloader)
#     print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
#     return loss, {"accuracy": accuracy}


# # Pass parameters to the Strategy for server-side parameter initialization
# strategy = fl.server.strategy.FedAvg(
#     fraction_fit=0.3,
#     fraction_evaluate=0.3,
#     min_fit_clients=3,
#     min_evaluate_clients=3,
#     min_available_clients=NUM_CLIENTS,
#     #initial_parameters=fl.common.ndarrays_to_parameters(params),
# )

# # Start Flower server
# fl.server.start_server(
#     server_address="0.0.0.0:8088",
#     config=fl.server.ServerConfig(num_rounds=10),
#     strategy=strategy,
# )

    # trainpart = list(range(int(0.34*50000), int(0.67*50000), 1))
    # testpart = list(range(int(0.34*10000), int(0.67*10000), 1)) 

    # trainpart = list(range(int(0.67*50000), int(1*50000), 1))
    # testpart = list(range(int(0.67*10000), int(1*10000), 1)) 

    # trainpart = list(range(0, int(0.34*len(trainset)), 1))
    #testpart = list(range(0, int(0.34*len(testset)), 1))

    # def load_data():

#     trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     trainset = CIFAR10("./data", train=True, download=True, transform=trf)
#     testset = CIFAR10("./data", train=False, download=True, transform=trf)
#     trainpart = list(range(0, int(0.34*len(trainset)), 1))
#     testpart = list(range(0, int(0.33*len(testset)), 1))  
#     trainset_1 = torch.utils.data.Subset(trainset, trainpart)
#     testset_1 = torch.utils.data.Subset(testset, testpart)
#     return DataLoader(trainset_1, batch_size=32, shuffle=True), DataLoader(testset_1)
 # Make sure numpy is installed


# Initialize logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("FlowerServer")

# # Custom FedAvg strategy
# class CustomFedAvg(fl.server.strategy.FedAvg):
#     def aggregate_fit(
#         self,
#         rnd: int,
#         results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#         failures: List[BaseException],
#     ) -> Optional[List[np.ndarray]]:
#         # Log received model updates
#         for client, fit_res in results:
#             logger.info(
#                 f"Received model update from client {client.cid} "
#                 f"with metrics {fit_res.metrics}"
#             )
#         # Proceed with the usual aggregation
#         return super().aggregate_fit(rnd, results, failures)

# # Define strategy
# strategy = CustomFedAvg()

# # Start Flower server
# fl.server.start_server(
#     server_address="0.0.0.0:8088",
#     config=fl.server.ServerConfig(num_rounds=3),
#     strategy=strategy,
# )



# # Initialize logging
# logging.basicConfig(
#     filename='flower_server2.log',  # Log file name
#     filemode='a',  # Append to log file if it exists, create if it doesn't
#     level=logging.INFO,  # Logging level
#     format='%(asctime)s - %(levelname)s - %(message)s'  # Format of log messages
# )
# logger = logging.getLogger("FlowerServer")

def calculate_stats_and_flag_anomalies(client_weights, threshold=0.3):
    # Assuming client_weights is a list of arrays, one per client
    if not client_weights:
        return []

    # Calculate the mean and standard deviation
    stacked_weights = np.stack(client_weights)
    mean_weights = np.mean(stacked_weights, axis=0)
    std_weights = np.std(stacked_weights, axis=0)

    anomalies = []
    for i, weights in enumerate(client_weights):
        z_scores = np.abs((weights - mean_weights) / std_weights)
        if np.any(z_scores > threshold):
            anomalies.append(i)
    
    return anomalies

# # Custom FedAvg strategy
# class CustomFedAvg(fl.server.strategy.FedAvg):
#     def aggregate_fit(
#         self,
#         rnd: int,
#         results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
#         failures: List[BaseException],
#     ) -> Optional[List[np.ndarray]]:
#         client_weights = []
#         for client, fit_res in results:
#             if fit_res is not None:
#                 weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
#                 client_weights.append(np.concatenate([w.flatten() for w in weights]))
#                 logger.info(f"Round {rnd}, Client {client.cid}: Received model update")
            
#             else:
#                 logger.warning(f"Round {rnd}, Client {client.cid}: Failed to receive model update")

#         # Perform anomaly detection after collecting all client weights
#         anomalies = calculate_stats_and_flag_anomalies(client_weights)
#         print(client_weights)
#         for anomaly_index in anomalies:
#             client_id = results[anomaly_index][0].cid
#             logger.warning(f"Round {rnd}, Client {client_id}: Potential anomaly detected in weights")

#         return super().aggregate_fit(rnd, results, failures)

# # Define strategy
# strategy = CustomFedAvg()

# # Start Flower server
# fl.server.start_server(
#     server_address="0.0.0.0:8088",
#     config=fl.server.ServerConfig(num_rounds=3),
#     strategy=strategy,
# )

import logging
import warnings
from typing import List, Optional, Tuple
import flwr as fl
from flwr.common import FitRes
import numpy as np

# Initialize logging to capture warnings
logging.basicConfig(
    filename='flower_server2.log',  # Log file name
    filemode='a',  # Append to log file if it exists, create if it doesn't
    level=logging.INFO,  # Logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Format of log messages
)
logger = logging.getLogger("FlowerServer")
logging.captureWarnings(True)

def calculate_stats_and_flag_anomalies(client_weights, threshold=2.0):
    num_clients = len(client_weights)
    num_layers = len(client_weights[0])
    anomalies = []

    # Calculate layer-wise mean and std dev
    layer_means = [np.mean([client_weights[i][l] for i in range(num_clients)], axis=0) for l in range(num_layers)]
    layer_stds = [np.std([client_weights[i][l] for i in range(num_clients)], axis=0) for l in range(num_layers)]

    # Calculate z-scores and flag anomalies
    for i, weights in enumerate(client_weights):
        try:
            for l, layer_weights in enumerate(weights):
                z_scores = np.abs((layer_weights - layer_means[l]) / layer_stds[l])
                if np.any(z_scores > threshold):
                    anomalies.append(i)
                    raise ValueError(f"Anomaly detected in client {i}, layer {l}")
        except RuntimeWarning as e:
            logger.warning(f"RuntimeWarning during anomaly detection: {e}")
            logger.info(f"Problematic weights: {weights}")

    return anomalies

# Custom FedAvg strategy
class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[List[np.ndarray]]:
        client_weights = []
        for client, fit_res in results:
            if fit_res is not None:
                weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
                client_weights.append(np.concatenate([w.flatten() for w in weights]))
                logger.info(f"Round {rnd}, Client {client.cid}: Received model update")
            else:
                logger.warning(f"Round {rnd}, Client {client.cid}: Failed to receive model update")

        # Perform anomaly detection after collecting all client weights
        anomalies = calculate_stats_and_flag_anomalies(client_weights)
        for anomaly_index in anomalies:
            client_id = results[anomaly_index][0].cid
            logger.warning(f"Round {rnd}, Client {client_id}: Potential anomaly detected in weights")

        return super().aggregate_fit(rnd, results, failures)

# Define strategy
strategy = CustomFedAvg()

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8089",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
