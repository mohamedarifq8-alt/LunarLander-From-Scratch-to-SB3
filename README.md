# 🚀 Mastering PPO: From Scratch to Production (LunarLanderContinuous-v3)

This repository documents a complete engineering journey of training an AI agent to control a spacecraft landing using the **Proximal Policy Optimization (PPO)** algorithm. The project doesn't just rely on out-of-the-box libraries; it progresses from building the algorithm mathematically and programmatically from scratch, to accelerating training via vectorized environments, and finally implementing industrial production standards.

---


## 🛤️ The Journey (Project Phases)

This project was built in three main phases to deeply understand and apply Deep Reinforcement Learning (RL) concepts:

### Phase 1: Custom PPO Implementation (From Scratch)
* **Description:** Writing the core architecture of the algorithm using `PyTorch` to solidify the mathematical understanding.
* **Techniques Used:** 
  * Designing an **Actor-Critic** neural network architecture.
  * Implementing the **PPO Clipped Objective** to prevent catastrophic forgetting and policy collapse.
  * Resolving "Trajectory Mixing" by strictly separating episodes for accurate cumulative return calculations.
  * Utilizing an **Independent Log-Std** parameter to maintain healthy exploration and avoid the "safe hovering" trap.
* **Result:** Achieved a stable and successful landing with a score of **~150 points**.

### Phase 2: Architectural Acceleration (Vectorized Environments)
* **Description:** Transitioning the agent from sequential single-play to matrix processing by controlling multiple landers simultaneously to drastically reduce training time.
* **Techniques Used:**
  * Running **16 Parallel Environments** (Vectorized) to gather thousands of steps in seconds.
  * Implementing **Learning Rate Annealing** to ensure weight stability when approaching the optimal solution.
  * Applying **Entropy Decay** to gradually eliminate mechanical jitter.
* **Result:** Massive training speedup, resulting in a highly accurate, safety-first policy (scoring **~110 points**) with a near 100% success rate of landing precisely between the flags.

### Phase 3: Production-Grade (Stable Baselines3)
* **Description:** Moving to standard industrial tools after mastering the mathematical fundamentals.
* **Techniques Used:**
  * Integrating `Stable Baselines3` with `Gymnasium Vectorized Environments`.
  * Using highly optimized hyperparameters for maximum efficiency.
  * Extracting a **Deterministic Policy** during evaluation to eliminate residual randomness, producing a flawless, cinematic landing video.
* **Result:** Completely solved the environment (Environment Solved), breaking the **200-point** barrier with a stunning, fuel-efficient landing.

---

## 🧠 Core Concepts Explored

* **Policy Gradient Methods:** Optimizing policies directly while keeping updates within a Trust Region.
* **Advantage Estimation:** Evaluating the quality of instantaneous actions compared to expected values, rather than relying solely on the final outcome.
* **Local Optima Escape:** Overcoming cowardly survival strategies (like hovering to waste time) to reach optimal strategies (fast, precise landings).
* **Sim2Real Foundations:** Understanding domain randomization and physics engines (Box2D) as a first step toward transferring models to real-world robots.

---

## 🛠️ Installation & Repository Structure

### Prerequisites

```bash
pip install torch numpy gymnasium stable-baselines3[extra] imageio imageio-ffmpeg
```

### Repository Structure

```text
📦 PPO-LunarLander
 ┣ 📂 scripts
 ┃ ┣ 📜 LunarLander.py                  # Phase 1: From-scratch implementation
 ┃ ┣ 📜 PPO_with_Vectorized.py          # Phase 2: 16 parallel envs with LR Annealing
 ┃ ┗ 📜 ppo_with_Stable_Baselines3.py   # Phase 3: Final Stable Baselines3 code
 ┣ 📂 models                            # Saved Weights (Pre-trained Models)
 ┃ ┣ 📜 perfect_ppo_lunar.pth              # Weights for Phase 1
 ┃ ┣ 📜 perfect_ppo_lunar_vectorized.pth   # Weights for Phase 2
 ┃ ┗ 📜 ppo_lunar_sb3_production.zip       # Final SB3 Production Model
 ┣ 📂 videos                            # Rendered Agent Landings
 ┃ ┣ 🎬 perfect_landing.mp4                # Phase 1 Landing
 ┃ ┣ 🎬 perfect_landing_vectorized.mp4     # Phase 2 Landing
 ┃ ┗ 🎬 sb3_perfect_landing.mp4            # Phase 3 Landing (Cinematic & Solved)
 ┗ 📜 README.md
```

---

## 🏋️ Training and Evaluation

To train the final production version (Phase 3):

```bash
python scripts/ppo_with_Stable_Baselines3.py
```

To evaluate a trained agent and render a video:

```bash
python scripts/evaluate_and_render.py
```
