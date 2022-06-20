"""Profiling PCN with constraint."""

from types import SimpleNamespace
from functools import partial

import torch
import time

from tqdm.notebook import tqdm

from cpcn import *

# setup
device = torch.device("cpu")
torch.manual_seed(123)

batch_size = 100
dataset = load_mnist(n_validation=500, batch_size=batch_size, device=device)

# create net, run training
n_batches = 2000
dims = [784, 5, 10]

z_it = 50
z_lr = 0.1

t0 = time.time()
torch.manual_seed(123)

net0 = PCNetwork(
    dims,
    activation=lambda _: _,
    z_lr=z_lr,
    z_it=z_it,
    constrained=True,
    rho=0.015,
    fast_optimizer=torch.optim.Adam,
    bias=False,
)

net = PCWrapper(net0, "linear").to(device)
optimizer = torch.optim.Adam(net.pc_net.parameters(), lr=0.001)
predictor_optimizer = torch.optim.Adam(net.predictor.parameters())
lr_power = 1.0
lr_rate = 4e-4
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda batch: 1 / (1 + lr_rate * batch ** lr_power)
)
trainer = Trainer(dataset["train"], invalid_action="warn+stop")
trainer.metrics["accuracy"] = dot_accuracy
for batch in tqdmw(trainer(n_batches)):
    if batch.every(10):
        batch.evaluate(dataset["validation"]).run(net)
        batch.weight.report("W", net.pc_net.W)

    ns = batch.feed(net, latent_profile=True)
    batch.latent.report_batch("z", ns.z)
    if batch.count(4):
        batch.fast.report_batch("z", [_.transpose(0, 1) for _ in ns.profile.z])

    optimizer.step()
    predictor_optimizer.step()
    scheduler.step()

results = trainer.history
print(f"Training PCN took {time.time() - t0:.1f} seconds.")
