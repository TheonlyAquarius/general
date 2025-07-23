# SynthNet Research Playground

This repository explores training diffusion-based weight generators. The
`mnist_weight_diffusion` directory contains the original MNIST proof of concept.
The new `synthnet` package starts scaffolding a more flexible system using Hydra
configurations and a `PolicyAPI` abstraction.

Run the skeleton trainer with:

```bash
python train_synthnet.py
```

This uses the default configs under `synthnet/configs/` and prints progress for a
placeholder GRPO loop.
