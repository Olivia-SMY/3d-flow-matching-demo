# 3D Point Cloud Flow Matching 🧬

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vhKdlpIrlynr4uF2QLXAonUoiSrSIT5q?usp=sharing)

A minimalist implementation of **Continuous Normalizing Flows (CNF)** designed to generate complex 3D manifolds (DNA double helices) from Gaussian noise via **ODE dynamics**.

![Demo](dna_flow_matching.gif)

## Key Features
* **From Scratch:** Implemented the MLP architecture, Flow Matching loss, and ODE solver using raw **PyTorch** (no high-level diffusion libraries).
* **Deterministic Sampling:** Uses a custom Euler-method solver to simulate particle trajectories along the learned vector field.
* **Geometric Deep Learning:** Captures multi-modal 3D distributions efficiently, validating the optimal transport path.

## Method
The model minimizes the **Conditional Flow Matching (CFM)** objective to regress the vector field $v_\theta$ to the conditional flow $u_t(x|x_1)$:

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0,1)} \mathbb{E}_{x_1 \sim q(x_1)} \mathbb{E}_{x_0 \sim p(x_0)} \left[ \left\| v_\theta(\psi_t(x_0), t) - \frac{\mathrm{d}}{\mathrm{d}t}\psi_t(x_0) \right\|^2 \right]$$

Where the optimal transport path is defined as $\psi_t(x_0) = (1 - t)x_0 + t x_1$, yielding a constant target velocity $x_1 - x_0$.
