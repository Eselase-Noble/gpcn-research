import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """
    Temperature scaling for post-hoc calibration.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def set_temperature(self, model, val_loader, device):
        """
        Tune temperature on validation set.
        """
        model.eval()

        logits_list = []
        labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    images, labels = batch
                else:
                    images, labels = batch[0], batch[1]

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                logits_list.append(outputs)
                labels_list.append(labels)

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        print(f"Optimal temperature: {self.temperature.item():.4f}")
        return self