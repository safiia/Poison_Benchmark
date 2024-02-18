import flwr as fl

# Define a custom strategy to handle aggregation with Byzantine resilience
class ByzantineResilientFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        # Filter out malicious updates based on some criteria, e.g., too far from the mean
        # This example naively assumes the first two clients are malicious for demonstration
        filtered_results = results[2:]  # Ignoring updates from the first two clients
        
        # Proceed with the usual aggregation on filtered results
        return super().aggregate_fit(rnd, filtered_results, failures)


if __name__ == "__main__":
    strategy = ByzantineResilientFedAvg()
    fl.server.start_server(
        server_address="0.0.0.0:8089",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )
