"""Profiling PCN with constraint."""

from types import SimpleNamespace
from functools import partial

import torch

from tqdm.notebook import tqdm

from cpcn import PCNetwork, load_mnist, Trainer

# setup
device = torch.device("cpu")
torch.manual_seed(123)

batch_size = 100
dataset = load_mnist(n_validation=500, batch_size=batch_size, device=device)

# create net, run training
n_batches = 100
dims = [784, 5, 10]

z_it = 50
z_lr = 0.1

torch.manual_seed(123)

net = PCNetwork(
    dims,
    activation=lambda _: _,
    lr_inference=z_lr,
    it_inference=z_it,
    constrained=True,
    rho=0.015,
    fast_optimizer=torch.optim.Adam,
    bias=False,
)
net = net.to(device)

trainer = Trainer(net, dataset["train"], dataset["validation"])
# trainer.set_classifier("linear")

trainer.set_optimizer(torch.optim.Adam, lr=0.001)
# trainer.add_scheduler(partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.9))

results = trainer.run(n_batches=n_batches)
