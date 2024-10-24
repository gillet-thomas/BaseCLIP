import torch
from tqdm import tqdm

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

class Trainer():
    def __init__(self, config, model, dataset):
        self.config = config
        self.model = model
        self.data = dataset.train_data
        self.test_data = dataset.test_data
        self.device = config['device']

        self.batch_size = config['batch_size']
        self.epochs = config['epochs']

    def run(self):
        self.model.to(self.config['device'])
        print(f'CLIP model {sum(p.numel() for p in self.model.parameters())/1e6} M parameters running on {self.device}')

        self.dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)

        for epoch in tqdm(range(self.epochs)):
            print(f"Epoch {epoch+1}/{self.epochs}")
            self.train(epoch)
            # self.validate(epoch)

        torch.save(self.model.state_dict(), f'./CLIP_model.pth')
        print("Model saved to ./CLIP_model.pth")


    def train(self, epoch):
        self.model.train()
        running_loss = 0.0
        
        optimizer = torch.optim.AdamW(self.model.parameters(), weight_decay=0.)

        print(f"Number of batches: {len(self.dataloader)} of size {self.batch_size}")
        for i, (sources, targets) in enumerate(self.dataloader):
            sources, targets = sources.to(self.device), targets.to(self.device)  ## (batch_size, 2048) and (batch_size, 768)
            loss = self.model(sources, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


    def validate(self, epoch):
        self.model.eval()
        loss_meter = AvgMeter()

        tqdm_object = tqdm(self.test_dataloader, total=len(self.test_dataloader))
        for batch in tqdm_object:
            batch = {k: v.to(self.config["device"]) for k, v in batch.items() if k != "caption"}
            loss = self.model(batch)

            count = batch["image"].size(0)
            loss_meter.update(loss.item(), count)

            tqdm_object.set_postfix(valid_loss=loss_meter.avg)
        return loss_meter

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]