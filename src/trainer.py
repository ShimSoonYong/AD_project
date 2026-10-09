from src.utils import visualize_IO, Scorer

Trainer:
    def __init__(self, model, epochs, train_loader, test_loader, criterion, optimizer, device):
        print("Initializing Trainer")

        self.model = model
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self):
        self.model.train()
        print("Training starts")

        for epoch in range(self.epochs):
            print(f"{epohc} epoch start")
            for step, (x, label) in enumerate(self.train_loader):
                x = x.to(self.device)

                decoded = autoencoder(x)

                loss = criterion(decoded, x) 
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
