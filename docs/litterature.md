# Literature Review: Analytic Continuation with Neural Networks

Compiled 2026-05-25. Covers NN-based analytic continuation and temperature-agnostic approaches relevant to the `deep_continuation` project.

---

## Neural Network-Based Analytic Continuation

### Foundational papers

- **Yoon, Sim, Han (2018)** — *Analytic continuation via domain-knowledge free machine learning*, Phys. Rev. B 98, 245101. [arXiv:1806.03841](https://arxiv.org/abs/1806.03841). First CNN approach; 10^4–10^5× faster than MaxEnt, no physical priors needed.

- **Fournier, Wang, Yazyev, Wu (2020)** — *Artificial Neural Network Approach to the Analytic Continuation Problem*, PRL 124, 056401. [arXiv:1810.00913](https://arxiv.org/abs/1810.00913). Supervised learning framework; matches MaxEnt at low noise, substantially outperforms it at high noise.

- **Xie, Bao, Maier, Webster (2019/2021)** — *Analytic Continuation of Noisy Data Using Adams Bashforth ResNet*. [arXiv:1905.10430](https://arxiv.org/abs/1905.10430). Adams-Bashforth ResNet; model-independent, no prior spectral information needed.

### Addressing training bias

- **Zhang, Merkel, Beck, Ederer (2022)** — *Training biases in machine learning for the analytic continuation of quantum many-body Green's functions*, Phys. Rev. Research 4, 043082. [arXiv:2206.07493](https://arxiv.org/abs/2206.07493). Directly examines bias from using Gaussian/Lorentzian/Beta synthetic training data — highly relevant to this project's design choices.

- **Yao, Wang, Yao, Zhai (2022)** — *Noise Enhanced Neural Networks for Analytic Continuation*, Mach. Learn.: Sci. Technol. [arXiv:2111.12266](https://arxiv.org/abs/2111.12266). Adding noise to training data improves generalization; captures multi-peak structures.

- **Sun, Wang (2023)** — *Neural Network Analytic Continuation for Monte Carlo: Improvement by Statistical Errors*. [arXiv:2302.11317](https://arxiv.org/abs/2302.11317). Training data noise statistics must match real input data.

### Recent advances

- **Zhao, Xu, Wang, Yang (2024)** — *Analytic Continuation by Feature Learning*. [arXiv:2411.17728](https://arxiv.org/abs/2411.17728). Feature Learning Network (FL-net); ≥20% improvement over MaxEnt and prior NN methods.

- **Assaad et al. (2026)** — *Analytic continuation of Green's functions with a neural network*. [arXiv:2602.18561](https://arxiv.org/abs/2602.18561). CNN with enforced positive semidefiniteness; compared against MaxEnt on Hubbard and SSH models.

---

## Temperature-Agnostic / Domain-Agnostic Ideas

- **2024 ScienceDirect paper** — *Deep learning route to quantum materials: A domain agnostic analytic continuation mapping*, Computational Materials Science. Directly addresses domain-agnostic (temperature-independent) continuation — the closest paper in spirit to this project.

- The **training-bias paper** (Zhang et al. 2022 above) is also relevant here: it discusses how training over a distribution of spectral function types (as this project does with BetaMix) affects generalization.

---

## Optical Conductivity Specifically

- **Gunnarsson, Haverkort, Sangiovanni (2010)** — *Analytical continuation of imaginary axis data for optical conductivity*, Phys. Rev. B 82, 165125. [arXiv:1012.5934](https://arxiv.org/abs/1012.5934). Systematic comparison of MaxEnt, SVD, Padé for the optical conductivity case — useful reference for the problem formulation used here.

---

## Diffusion / Generative Models for Analytic Continuation

**No paper has yet applied diffusion models specifically to the Matsubara → σ(ω) analytic continuation problem.** The sections below cover the closest methodological neighbors.

### Why diffusion models are promising here

Analytic continuation is maximally ill-posed: many spectral functions are consistent with the same Matsubara data. Diffusion models naturally sample the full posterior P(σ | Π), giving an ensemble of plausible spectra rather than a single point estimate — something MaxEnt approximates crudely and feedforward NNs cannot do at all.

### Conditional score-based / diffusion models for physics inverse problems

- **Holzschuh, Vegetti, Thuerey (2023)** — *CoCoGen: Physically-Consistent and Conditioned Score-based Generative Models for Forward and Inverse Problems*. [arXiv:2312.10527](https://arxiv.org/abs/2312.10527). Score-based models with explicit physics consistency constraints — relevant since σ(ω) ≥ 0 must be enforced.

- **Pakravan et al. (2024)** — *Conditional score-based diffusion models for solving inverse problems in mechanics*. [arXiv:2406.13154](https://arxiv.org/abs/2406.13154). Direct framework for using conditional score-based diffusion for physics inverse problems; the closest methodological template for applying diffusion to analytic continuation.

- **2026 paper** — *Physics-informed diffusion models in spectral space*. [arXiv:2602.09708](https://arxiv.org/abs/2602.09708). Works in the spectral/modal domain rather than grid space — could map naturally onto frequency-space spectral functions.

- **Meng, Kabashima (2023)** — *Solving Inverse Problems with Score-Based Generative Priors learned from Noisy Data*. [arXiv:2305.01166](https://arxiv.org/abs/2305.01166). Score-based priors learned from noisy training data — relevant given that QMC Matsubara data is always noisy.

### Bayesian posterior sampling framing

- **2025 paper** — *Bayesian Analysis and Analytic Continuation of Scattering Amplitudes from Lattice QCD*. [arXiv:2506.16161](https://arxiv.org/abs/2506.16161). Frames analytic continuation as Bayesian posterior sampling with Nevanlinna-Pick interpolation — the natural framing for a diffusion-based approach.

### Diffusion for other spectral/ill-posed inverse problems (methodological references)

- **2024 paper** — *Back-Projection Diffusion: Solving the Wideband Inverse Scattering Problem with Diffusion Models*. [arXiv:2408.02866](https://arxiv.org/abs/2408.02866). Probabilistic posterior sampling from limited data; honors physics symmetries.

- **2025 paper** — *Solving geophysical inverse problems with measurement-guided diffusion models*. [arXiv:2501.04881](https://arxiv.org/abs/2501.04881). Seismic inversion with diffusion models — good reference for ill-posed physics inversion.

- **2026 paper** — *Score-based diffusion models for diffuse optical tomography with uncertainty quantification*. [arXiv:2602.03449](https://arxiv.org/abs/2602.03449). Ill-posed optical inverse problem with uncertainty quantification via score-based diffusion.

---

## Grid-Free and Pole/Lehmann-Native Representations

The natural arena for spectral functions is the Lehmann (pole) representation G(z) = Σᵢ wᵢ/(z − εᵢ), not a fixed frequency grid. The papers below work directly in this space.

### Minimal pole representation

- **Zhang, Gull (2024)** — *Minimal Pole Representation and Controlled Analytic Continuation of Matsubara Response Functions*, Phys. Rev. B 110, 035154. [arXiv:2312.10576](https://arxiv.org/abs/2312.10576). Directly parametrizes G(z) by pole positions and weights; systematically improvable by adding poles; no grid required.

- **Zhang, Erpenbeck, Yu, Gull (2025)** — *Minimal Pole Representation for Spectral Functions*. [arXiv:2504.01163](https://arxiv.org/abs/2504.01163). Extends the minimal-pole approach to spectral densities directly. Per-instance optimization (not yet amortized by a neural network — that combination is open).

### Lehmann-native neural network

- **Kakizawa, Terasaki, Shinaoka (2024)** — *Physics-informed neural network model for quantum impurity problems based on Lehmann representation*. [arXiv:2411.18835](https://arxiv.org/abs/2411.18835). PINN that outputs pole weights and positions rather than grid values; 7.8× error reduction over grid-based NNs. Applied to Anderson impurity models, not optical conductivity.

### Nevanlinna and stochastic pole methods

- **Iskakov, Hampel, Wentzell, Gull (2023)** — *TRIQS/Nevanlinna: Implementation of the Nevanlinna Analytic Continuation method*. [arXiv:2309.01407](https://arxiv.org/abs/2309.01407). Open-source, pole-respecting continuation method; alternative to MaxEnt for noise-free data.

- **Huang, Liang (2023)** — *Stochastic pole expansion method*. [arXiv:2307.11324](https://arxiv.org/abs/2307.11324). Stochastic optimization over pole parameters; captures sharp features while preserving causality.

### Continued fraction and rational neural networks

- **Wang et al. (2026)** — *CFNN: Continued Fraction Neural Network*. [arXiv:2603.20634](https://arxiv.org/abs/2603.20634). Network whose computation graph is a continued fraction; Padé approximants and Nevanlinna constructions are special cases — the right inductive bias for analytic functions with pole structure. Not yet applied to analytic continuation.

- **Boullé (2022)** — *Data-driven discovery of Green's functions*. [arXiv:2210.16016](https://arxiv.org/abs/2210.16016). Rational neural networks with learnable rational activation functions; naturally approximate singularities via poles.

- **Lechien, Dudal (2022)** — *Neural network approach to reconstructing spectral functions and complex poles of confined particles*. [arXiv:2203.03293](https://arxiv.org/abs/2203.03293). NN that directly outputs complex pole pairs; applied to lattice QCD gluon propagators.

---

## Flow Matching and Modern Generative Models for Inverse Problems

Flow matching has emerged as the successor to diffusion for inverse problems: fewer neural function evaluations, exact likelihood via instantaneous change-of-variables, and straighter transport paths. **None of this has been applied to analytic continuation yet.**

### Flow matching for physics inverse problems

- **2026 paper** — *Solving Physics-Constrained Inverse Problems with Conditional Flow Matching*. [arXiv:2603.14135](https://arxiv.org/abs/2603.14135). Conditional flow matching learns a velocity field transporting a source distribution to the Bayesian posterior conditioned on measurements; supports nonlinear forward models.

- **FLOWER (2025)** — *FLOWER: A Flow-Matching Solver for Inverse Problems*. [arXiv:2509.26287](https://arxiv.org/abs/2509.26287). Dedicated flow-matching inverse problem solver; empirically outperforms diffusion in efficiency and quality.

- **2025 paper** — *Bayesian Inverse Problems Meet Flow Matching: Efficient and Flexible Inference via Transformers*. [arXiv:2503.01375](https://arxiv.org/abs/2503.01375). Combines flow matching with transformer conditioning for flexible Bayesian inverse inference.

- **Lipman et al. (2023)** — *Flow Matching with Minibatch Optimal Transport (OT-CFM)*. [arXiv:2302.00482](https://arxiv.org/abs/2302.00482). Foundational OT-CFM paper; geodesic generative paths with minimal transport cost.

### Consistency models (ultra-fast alternative)

- **2025 paper** — *Consistency Models as Plug-and-Play Priors for Inverse Problems*. [arXiv:2509.22736](https://arxiv.org/abs/2509.22736). 1–4 step generation vs. hundreds for diffusion; competitive quality for inverse problems.

---

## Operator Learning and Grid-Free Neural Architectures

### DeepONet — grid-free on the output side

- **Lu et al. (2021)** — *Learning nonlinear operators via DeepONet*, Nature Machine Intelligence. [arXiv:1910.03193](https://arxiv.org/abs/1910.03193). Branch-trunk architecture learns the operator Π(iωₙ) → σ(ω); trunk network evaluates at arbitrary query frequencies — inherently grid-free on output.

- **φ-DeepONet (2026)** — *A Discontinuity Capturing Neural Operator*. [arXiv:2604.08076](https://arxiv.org/abs/2604.08076). Extends DeepONet to handle discontinuities — relevant for sharp Drude peaks or coherence peaks.

### State space models for long sequences

- **Gu et al. (2022)** — *S4: Structured State Spaces for Sequence Modeling*. HiPPO framework; efficient O(N) long-range modeling — natural for the Matsubara frequency sequence as input.

- **2024 paper** — *State-space models are accurate and efficient neural operators for dynamical systems*. [arXiv:2409.03231](https://arxiv.org/abs/2409.03231). SSMs (including Mamba) as neural operators; O(N) vs O(N²) for transformers on long sequences.

---

## Toolkits for Comparison

- **Kaufmann, Held (2021)** — *ana_cont: Python package for analytic continuation*. [arXiv:2105.11211](https://arxiv.org/abs/2105.11211). Open-source Python package (Padé + MaxEnt).

- **Huang (2023)** — *ACFlow: An Open Source Toolkit for Analytical Continuation*, Comput. Phys. Commun. [arXiv:2211.16692](https://arxiv.org/abs/2211.16692). Comprehensive toolkit (MaxEnt, stochastic, Nevanlinna, etc.) — useful benchmark baseline.
