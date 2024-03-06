import torch
import torch.nn as nn

class ConsisLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nonline = nn.Softmax(dim=1)
        self.smooth = 1e-5
    
    def forward(self, x1, x2, y):
        assert x1.shape == x2.shape, 'two estimates must have the same size'
        with torch.no_grad():
            y_onehot = torch.zeros_like(x1)
            y_onehot = y_onehot.scatter(1, y.long(), 1)
        axes = [0] + list(range(2, len(x1.shape)))

        x1 = self.nonline(x1)
        x2 = self.nonline(x2)

        numerator = 3. * (x1 * x2 * y_onehot).sum(axes) + self.smooth
        denominator = (x1 + x2 + y_onehot).sum(axes) + self.smooth

        cons = numerator / (denominator + 1e-8)
        return -cons[1:].mean()