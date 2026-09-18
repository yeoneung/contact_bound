"""Fresh, pre-specified Ridge1D critics for rigorous post-training verification.

Both use the original fitted-value-iteration trainer, seed 1, 4000 updates,
4096 samples, h=0.02. This run saves the weights (the old E1 run did not).
"""
from pathlib import Path
import hashlib
import json
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"code"))
from hjrl.critic import FourierMLP, MinBranchCritic
from hjrl.problems import Ridge1D
from hjrl.train import train_sl_td


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    problem = Ridge1D(device)
    states, records = {}, []
    for name, branches in (("MLP", 1), ("Min2", 2)):
        torch.manual_seed(1)
        model = (FourierMLP(1, n_freq=8, width=64, depth=3) if branches == 1
                 else MinBranchCritic(1, K=2, n_freq=8, width=64, depth=3)).to(device)
        start = time.perf_counter()
        history = train_sl_td(problem, model, steps=4000, batch=4096, h=0.02,
                              seed=1, log_every=1000, n_eval=8192)
        states[name] = {k:v.detach().cpu() for k,v in model.state_dict().items()}
        records.append({"critic":name, "branches":branches,
                        "seconds":time.perf_counter()-start, "history":history.rows})
        print(json.dumps(records[-1]), flush=True)
        torch.save(states, ROOT/"results/verified_neural_models.pt")
    p=ROOT/"results/verified_neural_training.json"
    p.write_text(json.dumps({"device":str(device), "seed":1, "steps":4000,
                            "batch":4096, "hold":0.02, "n_freq":8,"width":64,"depth":3,
                            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "records":records},indent=2)+"\n",encoding="utf8")


if __name__ == "__main__":main()
