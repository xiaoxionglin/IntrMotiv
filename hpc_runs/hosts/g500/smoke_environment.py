"""Bounded GPU and custom DMLab checks; run after sourcing activate.sh.

This verifies binary compatibility, pretrained weights, rendering and engine
steps. It does not qualify an experiment or launch training.
"""

import json
import os
import sys

import deepmind_lab
import numpy as np
import torch
import torchvision
from torchvision.models import ResNet18_Weights, resnet18


def main():
    torch.set_num_threads(1)
    assert torch.cuda.is_available(), "CUDA unavailable"
    devices = []
    for index in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{index}")
        x = torch.randn(64, 64, device=device, requires_grad=True)
        (x @ x.T).square().mean().backward()
        assert torch.isfinite(x.grad).all()
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 72, 96, device=device))
        assert output.shape == (1, 1000) and torch.isfinite(output).all()
        torch.cuda.synchronize(device)
        devices.append({"index": index, "name": torch.cuda.get_device_name(index),
                        "capability": torch.cuda.get_device_capability(index)})
        del model, x, output
        torch.cuda.empty_cache()

    lab = deepmind_lab.Lab(
        "openfield_map2_fixed_loc3_fixedlength_noreward", ["RGB_INTERLEAVED"],
        config={"width": "96", "height": "72", "fps": "60"}, renderer="software",
    )
    try:
        lab.reset(seed=99)
        frame = lab.observations()["RGB_INTERLEAVED"]
        assert frame.shape == (72, 96, 3) and frame.std() > 0
        action = np.zeros(len(lab.action_spec()), dtype=np.intc)
        reward = sum(float(lab.step(action, num_steps=8)) for _ in range(16))
        assert reward == 0 and lab.is_running()
    finally:
        lab.close()

    print(json.dumps({"python": sys.version, "torch": torch.__version__,
                      "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                      "devices": devices, "dmlab_steps": 128, "reward": reward,
                      "torch_home": os.environ["TORCH_HOME"], "status": "passed"}, indent=2))


if __name__ == "__main__":
    main()
