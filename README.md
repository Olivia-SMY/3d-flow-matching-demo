# 3D Point Cloud Flow Matching 🧬

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vhKdlpIrlynr4uF2QLXAonUoiSrSIT5q?usp=sharing)

A minimalist implementation of **Continuous Normalizing Flows (CNF)** designed to generate complex 3D manifolds (DNA double helices) from Gaussian noise via **ODE dynamics**.

![Demo](dna_flow_matching.gif)

## Key Features
* **From Scratch:** Implemented the MLP architecture, Flow Matching loss, and ODE solver using raw **PyTorch** (no high-level diffusion libraries).
* **Deterministic Sampling:** Uses a custom Euler-method solver to simulate particle trajectories along the learned vector field.
* **Geometric Deep Learning:** Captures multi-modal 3D distributions efficiently, validating the optimal transport path.

## Method
The model minimizes the **Conditional Flow Matching** objective to learn a time-dependent vector field $v_t(x)$:

$$\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t, x_1, x_0} || v_\theta(\psi_t(x_0), t) - (x_1 - x_0) ||^2$$

