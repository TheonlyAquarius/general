# SynthNet: A Generative Framework for Neural Network Weight Synthesis

This repository contains the official research and implementation for the SynthNet project. Its purpose is to explore and develop a novel method for generating high-performing neural network weights directly from a random initialization, bypassing traditional data-driven training at inference time.

---

## Core Concepts

SynthNet is built on a specific set of foundational principles that distinguish it from other research in learned optimizers or model generation.

### Weight Synthesis, Not Trajectory Replication

The primary objective of SynthNet is to learn a direct, few-step mapping from a chaotic, randomly initialized weight-state to a structured, high-performing one. It is **not** a "trajectory replicator" designed to mimic the specific weight updates of an SGD-based training process. The goal is to synthesize the *solution*, not to reproduce the *path*.

### Empirical Diffusion

The generative mechanism is a form of diffusion. However, instead of relying on artificial Gaussian noise, SynthNet uses an **empirical noise model**. The "noise" is the inherent randomness of the initial weight tensors themselves. The denoising process is therefore a learned transformation that imposes the statistical structure of a well-trained network onto this initial chaotic state.

### Reinforcement Learning with GRPO

SynthNet's policy is trained using **Group Relative Policy Optimization (GRPO)**. This is a reinforcement learning paradigm where the reward signal is derived directly from the functional performance (e.g., validation accuracy) of the synthesized weights when loaded into a target architecture. This allows the model to optimize for performance without needing a differentiable loss function related to the weights themselves.

---

## Architectural Blueprint

The SynthNet architecture is designed from first principles to be modular, scalable, and structurally aware.

#### 1. Perceiver IO Backbone

To avoid the naive flattening of weights used in the initial Proof-of-Concept, the core architecture uses a **Perceiver IO** backbone. This model is explicitly designed to handle input as a *structured set of tensors*, making it the ideal choice for processing the contents of a `.safetensors` file while preserving architectural information.

#### 2. Mixture-of-Experts (MoE) for Specialization

To enable the model to handle different parameter types (e.g., convolutional kernels, layer norm biases) differently, **Mixture-of-Experts (MoE)** layers are integrated into the Perceiver backbone. This allows for **emergent specialization**, where a gating network learns to route information to different expert sub-networks based on the properties of the weight tokens, rather than relying on hard-coded logic.

---

## Project Structure

* **/unified_framework/**: Contains the main, modular, and up-to-date implementation of the SynthNet architecture and training loop. This is the primary area of development.
* **/mnist_weight_diffusion/**: The original, simplified Proof-of-Concept based on a simple MLP. This is preserved for historical and comparative purposes.
* **/configs/**: The Hydra configuration directory, which separates the experimental setup (the "what") from the implementation logic (the "how").
* **train_synthnet.py**: The main entry point for running experiments, orchestrated by Hydra.

---

## Getting Started

### 1. Installation

First, run the setup script to install all necessary dependencies into your Python environment. It is highly recommended to use a virtual environment.

```bash
bash setup.sh