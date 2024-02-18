from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import flwr as fl
from flwr.common import Metrics
import numpy as np
from simple_blockchain import Blockchain
import logging

# # Configure logging
# logging.basicConfig(
#     filename='flower_server.log',  # Log file name
#     filemode='a',  # Append to log file if it exists, create if it doesn't
#     level=logging.INFO,  # Logging level
#     format='%(asctime)s - %(levelname)s - %(message)s'  # Format of log messages
# )

# # Create a logger object
# logger = logging.getLogger('FlowerServerLogger')

# blockchain = Blockchain()

# if __name__ == "__main__":
#     # Start Flower server with a simple FedAvg strategy
#     strategy = fl.server.strategy.FedAvg()
#     fl.server.start_server(server_address="0.0.0.0:8089", config=fl.server.ServerConfig(num_rounds=3), strategy=strategy)

# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     try:
#         accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#         examples = [num_examples for num_examples, _ in metrics]
#         aggregated_accuracy = sum(accuracies) / sum(examples)
#         logger.info("Aggregated accuracy: %f", aggregated_accuracy)
#         return {"accuracy": aggregated_accuracy}
#     except Exception as e:
#         logger.error("Error during metric aggregation: %s", e)
#         raise

# # Define strategy
# strategy = fl.server.strategy.FaultTolerantFedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# # Start Flower server
# if __name__ == "__main__":
#     logger.info("Starting Flower server")
#     fl.server.start_server(
#         server_address="0.0.0.0:8089",
#         config=fl.server.ServerConfig(num_rounds=3),
#         strategy=strategy,
#     )


# if __name__ == "__main__":
#     # Start Flower server with a simple FedAvg strategy
#     strategy = fl.server.strategy.FedAvg()
#     fl.server.start_server(server_address="0.0.0.0:8089", config=fl.server.ServerConfig(num_rounds=3), strategy=strategy)


import flwr as fl
from typing import List, Tuple, Optional
import numpy as np
import logging

# Setup server logging
logging.basicConfig(
    filename='serverflwr.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

server_logger = logging.getLogger('server_logger')

class CustomAggregationStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[List[np.ndarray]]:
        # Iterate through the results and log model parameters
        for client_proxy, fit_res in results:
            # Convert Parameters to numpy arrays
            client_weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
            
            # Now you can iterate over client_weights as it's a list of numpy arrays
            for idx, weight in enumerate(client_weights):
                server_logger.info(f"Client {client_proxy.cid}: Weight {idx} - shape: {weight.shape}, mean: {np.mean(weight)}, std: {np.std(weight)}, weight: {weight}")

        # Proceed with the standard FedAvg aggregation
        return super().aggregate_fit(rnd, results, failures)


# Use the custom strategy when starting the server
if __name__ == "__main__":
    strategy = CustomAggregationStrategy()
    fl.server.start_server(
        server_address="0.0.0.0:8089",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )

