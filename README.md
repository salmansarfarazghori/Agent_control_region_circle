### Simulation GIFs

| ![Full Animation](https://github.com/salmanghori/Agent_control_region_circle/blob/main/animation__ITE_1500_R_90_Rnd_3_SD_3_BUI_7_T_2024-09-26--13-25-52.gif) | ![Resized Animation](https://github.com/salmanghori/Agent_control_region_circle/blob/main/animation__ITE_1500_R_90_Rnd_3_SD_3_BUI_7_T_2024-09-26--13-25-52-ezgif.com-resize.gif) |
|---|---|
| Full Simulation GIF | Resized GIF |

---

## Simulation Description

The GIF demonstrates the simulation of cross-flow management at an intersection where two orthogonal lanes intersect at the origin, forming a constrained space. A **green circle** with a radius of 90 meters represents the **control region** managed by a centralized system. As agents enter this region, their acceleration and movement are actively managed to ensure efficient and collision-free crossings at the intersection.

This simulation aligns with the study's focus on the **"early versus late management"** approach to managing autonomous agents in constrained spaces. **Early management** involves proactive control of agents using larger control regions before they reach critical areas, while **late management** intervenes closer to the intersection with smaller control regions.

The research investigates how intervention timing affects operational constraints, aiming to determine the optimal strategy for cross-flow management. The simulation supports the hypothesis that **increasing the control region radius beyond a certain threshold leads to diminishing returns** in system performance. This emphasizes the need to find the **optimal control region size** for maximizing efficiency and safety.




# Agent Control Region — Poisson Flow Intersection Simulation

**Core script:** `RH_intersection_possion_flow.py`

---

## About

This repository contains the official implementation of the simulation framework used in the paper:

> **Early Intervention Strategies to Enhance Fairness and Efficiency in Autonomous Traffic Flow Management**  
> Submitted to **IEEE Transactions on Intelligent Transportation Systems (T-ITS)**.  
> Authors: *Salman Ghori*, *Ania Adil*, *Melkior Ornik*, *Eric Feron*.

The code implements a **receding-horizon MILP controller** for agents approaching a two-flow orthogonal intersection inside a **circular control zone**. The simulator supports different control zone radii, fairness constraints, and Poisson-based arrival processes to evaluate delay, fairness, and efficiency metrics.

---

## Simulation Video & Repository

- 📄 [GitHub Repository README](https://github.com/salmansarfarazghori/Agent_control_region_circle/blob/main/README.md)  
- 🎥 [YouTube Demonstration](https://www.youtube.com/watch?v=mqgiasBgRtE)  

The GIF in the repository shows the simulator's functionality, with corresponding space-time diagrams provided in the paper.

---

## Citation & Usage

If you use this code or any part of it in your research, **you must cite** either:

### **Preferred (Paper)** — once the paper is published:
```bibtex
@article{ghori2025early,
  title   = {Early Intervention Strategies to Enhance Fairness and Efficiency in Autonomous Traffic Flow Management},
  author  = {Ghori, Salman and Adil, Ania and Feron, Eric and Ornik, Melkior},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  year    = {2025},
  note    = {to appear}
}


License & Intellectual Property
This software is subject to KAUST Intellectual Property rights.
Non-commercial research and educational use only is permitted.
For commercial use, licensing, or derivative works, written permission from KAUST is required.
All rights reserved © 2025, KAUST and the authors.
