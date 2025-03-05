import flwr as fl
import torch
from model import Net

def get_on_fit_config(server_round):
    return {
        "server_round": server_round,
        "local_epochs": 1,
    }

def get_evaluate_fn(model):
    # Helper function to set the model parameters
    def set_parameters(model, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict)
    
    def evaluate(server_round, parameters, config):
        set_parameters(model, parameters)
        # Global evaluation logic can be added here.
        # For now, we return dummy evaluation metrics.
        loss = 0.5
        accuracy = 0.9
        return loss, {"accuracy": accuracy, "server_round": server_round}
    
    return evaluate

def main():
    model = Net()
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,         # Sample 10% of available clients for training
        fraction_evaluate=0.1,      # Sample 10% of available clients for evaluation
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        on_fit_config_fn=get_on_fit_config,
        evaluate_fn=get_evaluate_fn(model)
    )

    fl.server.start_server(
        server_address="localhost:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=10)
    )

if __name__ == "__main__":
    main()
