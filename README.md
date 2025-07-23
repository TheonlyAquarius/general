# SynthNet Research Playground

This repository explores training diffusion-based weight generators. The `mnist_weight_diffusion` directory contains the original MNIST proof of concept. The new `synthnet` package now includes a Perceiver IO policy with a Mixture-of-Experts (MoE) head.

The policy uses the open source [perceiver-pytorch](https://github.com/lucidrains/perceiver-pytorch) implementation. Install dependencies with:

```bash
pip install perceiver-pytorch
```

Run the skeleton trainer with:

```bash
python train_synthnet.py
```

This uses the default configs under `synthnet/configs/` and runs a placeholder GRPO loop.
