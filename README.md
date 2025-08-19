# CosmoSim-ΛCDM

A high-performance 3D ΛCDM N-body simulator written in C++20. This code implements a Barnes–Hut tree for gravity in comoving coordinates with cosmological expansion, OpenMP parallelization, and CSV snapshot output. It’s intended as a foundation for extending into TreePM solvers, MPI scaling, and cosmological initial condition generation.

---

## Features
- **3D N-body simulation** in a periodic cubic box.
- **ΛCDM expansion:** configurable Ωm, ΩΛ, Ωr; H(a) = H0√(Ωm a⁻³ + Ωr a⁻⁴ + ΩΛ).
- **Barnes–Hut octree** for O(N log N) force evaluation with Plummer softening.
- **KDK leapfrog** integrator in comoving coordinates with Hubble drag.
- **OpenMP parallelism** for multi-core CPU acceleration.
- **Snapshot output** as CSV files (`snapshots/a_xxxxxx.csv`) containing positions and velocities.

---

## Build
Requirements:
- C++20 compiler (GCC ≥ 10, Clang ≥ 12)
- OpenMP (optional, but recommended for performance)

```bash
g++ -O3 -march=native -fopenmp -std=c++20 cosmosim.cpp -o cosmosim
```

---

## Run
Default parameters are built into the executable. You can override them via CLI arguments:

```bash
./cosmosim [N=20000] [steps=3000] [theta=0.6] [soft=0.04] [L=100.0] [a_start=0.02] [a_end=1.0] [dump_every=50]
```

Example:
```bash
./cosmosim 100000 4000 0.6 0.04 200 0.02 1.0 40
```

This produces snapshots in `./snapshots/` as CSV files, e.g.:
```
snapshots/a_00200.csv
snapshots/a_00400.csv
...
```

---

## Visualization
You can view snapshots in:
- **ParaView** or **VisIt** (CSV import, scatter plot).
- Custom Python scripts (e.g., using `matplotlib` or `plotly`).

Example Python quickview:
```python
import pandas as pd, matplotlib.pyplot as plt

snap = pd.read_csv("snapshots/a_01000.csv", comment='#')
plt.scatter(snap.x, snap.y, s=1, alpha=0.5)
plt.gca().set_aspect('equal')
plt.show()
```

---

## Roadmap
- [ ] Add **TreePM** long-range solver with FFTW.
- [ ] Implement **Zel’dovich approximation** initial condition generator.
- [ ] Replace CSV with **HDF5 snapshot format**.
- [ ] Add **MPI domain decomposition** for scaling across nodes.
- [ ] Validation: compare power spectrum, 2-pt correlation, halo mass function against CAMB/CLASS.

---

## Disclaimer
This is a **research-grade toy code**, not a replacement for production cosmology simulators like **GADGET**, **RAMSES**, or **ENZO**. It’s designed for learning, experimentation, and extension.

---

## License
MIT License. See [LICENSE](LICENSE) for details.
