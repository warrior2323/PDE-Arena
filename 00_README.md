# Physics-Informed Learning: Three Ways to Learn a PDE, and Where Each One Breaks

A comprehensive 100-hour machine learning project comparing **PINNs**, **FNOs**, and **Neural ODEs** for solving PDEs.

---

## 🗂️ Project Structure

```
📦 FINAL_100hrs_ML/
├── 00_README.md                    ← You are here
├── 01_exact_solutions.ipynb        ← Closed-form solutions for all 4 PDEs
├── 02_harness.ipynb                ← Measurement harness (accuracy + cost)
├── 03_pinn.ipynb                   ← PINN implementation (all 4 PDEs)
├── 04_fno.ipynb                    ← Fourier Neural Operator implementation
├── 05_neural_ode.ipynb             ← Neural ODE implementation
├── 06_main_benchmark.ipynb         ← Main benchmark: PINN vs FNO cost analysis
├── 07_investigation_advection_cliff.ipynb   ← Inv. 1: PINN failure at high speeds
├── 08_investigation_fno_spectral.ipynb      ← Inv. 2: FNO spectral bias & frequency
├── 09_investigation_resolution.ipynb        ← Inv. 3: FNO resolution invariance
├── 10_investigation_ood.ipynb               ← Inv. 4: Out-of-distribution generalization
├── 11_water_waves_simulation.ipynb          ← Bonus: Water wave dispersion study
├── 12_neural_ode_rollout.ipynb              ← Bonus: Neural ODE rollout stability
└── 13_inverse_problem.ipynb                ← Extension: Inverse problem (PINN strength)
```

---

## 🧠 Three Methods, One Question

| Method | What It Is | Training Data | Key Strength |
|--------|-----------|---------------|--------------|
| **PINN** | Neural network solver (collocation) | None (uses physics loss) | No data needed |
| **FNO** | Amortized operator learner | Many solved examples | Fast at scale |
| **Neural ODE** | Latent dynamics learner | Observed trajectories | Extrapolation |

---

## 🔬 Four Problems

1. **Heat Equation** — `u_t = α·u_xx` — Sanity check
2. **Advection** — `u_t + c·u_x = 0` — PINN failure mode
3. **Linear Water Waves** — Laplace's equation — Fluid mechanics
4. **Damped Oscillator** — `z̈ + 2ζω₀ż + ω₀²z = 0` — Neural ODE's home turf

---

## 🚀 Quick Start

Run notebooks in order (01 → 13). Each notebook is standalone.

### Dependencies
```bash
pip install torch numpy matplotlib scipy
```

---

## 📋 Key Findings

- PINN fails at advection speed `c ≥ 20` due to **optimization failure** (not capacity)
- FNO is **exact** for heat equation (Fourier space multiplication)  
- FNO's resolution invariance claim is **partially verified** (holds within ±1 octave)
- Neural ODE rollout degrades gracefully — error grows with horizon

---

## ⚠️ Honest Limitations

This project compares neural methods against **exact solutions only**.  
No comparison against conventional numerical solvers was performed.  
See McGreivy & Hakim (2024) for context on why that matters.

---

## 📚 References

- Raissi et al. (2019) — PINNs — arXiv:1711.10561
- Li et al. (2021) — FNO — arXiv:2010.08895
- Chen et al. (2018) — Neural ODEs — arXiv:1806.07366
- Krishnapriyan et al. (2021) — PINN failure modes — arXiv:2109.01050
- McGreivy & Hakim (2024) — Weak baselines — Nature Machine Intelligence
