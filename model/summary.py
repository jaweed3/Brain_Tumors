from torchinfo import summary
from u_net_model import UNet

model = UNet(n_channels=1, n_classes=2)
batch_size = 32
summary(model, input_size=(batch_size, 1, 28, 28))