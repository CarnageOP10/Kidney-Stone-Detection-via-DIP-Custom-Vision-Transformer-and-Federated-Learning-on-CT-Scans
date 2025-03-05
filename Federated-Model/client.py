import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Net

class MnistClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # move model to the detected device
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        for epoch in range(1):
            for batch_idx, (data, target) in enumerate(self.trainloader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.NLLLoss()
        correct, total, loss = 0, 0, 0.0

        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                loss += criterion(outputs, target).item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return loss / len(self.testloader), correct / total, {}

def load_data(is_train):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(
        "./data", train=is_train, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=32, shuffle=True)

def main():
    model = Net()
    trainloader = load_data(is_train=True)
    testloader = load_data(is_train=False)

    client = MnistClient(model, trainloader, testloader)
    # Convert the NumPyClient to a Client instance using .to_client()
    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client()
    )

if __name__ == "__main__":
    main()
