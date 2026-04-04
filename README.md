If I had to give you the top 5 priorities
1.	Use real PDE-derived logical circuits 
2.	Add multiple qubit sizes 
3.	Add more hardware topologies 
4.	Tighten pair matching on logical volume 
5.	Add execution-side/noise-derived observables



project/
├── experiments/
│   ├── exp1_boundary_topology.py
│   ├── exp2_scale_leakage.py
│   ├── exp3_drift_ablation.py
│   └── exp4_veracity_leakage.py
│


├── quantum/
│   ├── topologies.py
│   ├── circuits_pde.py
│   ├── circuits_scale.py
│   ├── circuits_veracity.py
│   ├── controls.py
│   ├── transpilation.py
│   ├── verification.py
│   └── features.py
│


├── data/
│   ├── builders_boundary.py
│   ├── builders_scale.py
│   ├── builders_veracity.py
│   ├── builders_drift.py
│   └── schemas.py
│


├── analysis/
│   ├── stats.py
│   ├── mi.py
│   ├── classifiers.py
│   ├── ordinal.py
│   ├── paired_tests.py
│   ├── drift.py
│   └── scaling.py
│


├── viz/
│   ├── plots_mi.py
│   ├── plots_distributions.py
│   ├── plots_confusion.py
│   ├── plots_drift.py
│   └── plots_scaling.py
│


├── configs/
│   ├── exp1_config.py
│   ├── exp2_config.py
│   ├── exp3_config.py
│   └── exp4_config.py
│


├── outputs/
├── config.py
└── main.py
