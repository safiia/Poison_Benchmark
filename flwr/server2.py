from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import flwr as fl
from flwr.common import Metrics
import numpy as np
from simple_blockchain import Blockchain
import logging

# Configure logging
logging.basicConfig(
    filename='flower_server.log',  # Log file name
    filemode='a',  # Append to log file if it exists, create if it doesn't
    level=logging.INFO,  # Logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Format of log messages
)

# Create a logger object
logger = logging.getLogger('FlowerServerLogger')

blockchain = Blockchain()

# # Define metric aggregation function
# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     # Multiply accuracy of each client by number of examples used
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]
#     # Aggregate and return custom metric (weighted average)
#     return {"accuracy": sum(accuracies) / sum(examples)}

# #Define strategy
# strategy = fl.server.strategy.FaultTolerantFedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# # Start Flower server
# fl.server.start_server(
#     server_address="0.0.0.0:8088",
#     config=fl.server.ServerConfig(num_rounds=3),
#     strategy=strategy,
# )


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    try:
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        aggregated_accuracy = sum(accuracies) / sum(examples)
        logger.info("Aggregated accuracy: %f", aggregated_accuracy)
        return {"accuracy": aggregated_accuracy}
    except Exception as e:
        logger.error("Error during metric aggregation: %s", e)
        raise

# Define strategy
strategy = fl.server.strategy.FaultTolerantFedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# Start Flower server
if __name__ == "__main__":
    logger.info("Starting Flower server")
    fl.server.start_server(
        server_address="0.0.0.0:8089",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
    logger.info("Flower server shutdown")
