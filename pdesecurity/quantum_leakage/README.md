# Quantum Leakage

A modular research codebase for studying whether hidden scientific properties of quantum workloads leak through provider-visible compilation artefacts.

## Project goals

This repository investigates whether a semi-honest cloud provider can infer hidden workload attributes from metadata generated during compilation and execution. The current experiments focus on four questions:

1. **Boundary topology leakage**  
   Can hidden boundary conditions such as Dirichlet vs Periodic be inferred from compilation artefacts?

2. **Scale leakage**  
   Can hidden scientific scale or resolution be inferred from routed depth, extra two-qubit overhead, and related features?

3. **Drift stability and feature-group ablation**  
   Which feature groups carry the leakage signal, and how stable is that signal under hardware drift?

4. **Veracity / accuracy leakage**  
   Can hidden accuracy requirements be inferred from provider-visible compilation overheads?

---

## Repository structure

```text
quantum_leakage/
├── experiments/
│   ├── exp1_boundary_topology.py
│   ├── exp2_scale_leakage.py
│   ├── exp3_drift_ablation.py
│   └── exp4_veracity_leakage.py
├── quantum/
│   ├── topologies.py
│   ├── circuits_pde.py
│   ├── circuits_scale.py
│   ├── circuits_veracity.py
│   ├── controls.py
│   ├── transpilation.py
│   ├── verification.py
│   └── features.py
├── data/
│   ├── builders_boundary.py
│   ├── builders_scale.py
│   ├── builders_veracity.py
│   ├── builders_drift.py
│   └── schemas.py
├── analysis/
│   ├── stats.py
│   ├── mi.py
│   ├── classifiers.py
│   ├── ordinal.py
│   ├── paired_tests.py
│   ├── drift.py
│   └── scaling.py
├── viz/
│   ├── plots_mi.py
│   ├── plots_distributions.py
│   ├── plots_confusion.py
│   ├── plots_drift.py
│   └── plots_scaling.py
├── configs/
│   ├── exp1_config.py
│   ├── exp2_config.py
│   ├── exp3_config.py
│   └── exp4_config.py
├── outputs/
├── README.md
├── config.py
└── main.py
