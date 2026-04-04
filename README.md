If I had to give you the top 5 priorities
1.	Use real PDE-derived logical circuits 
2.	Add multiple qubit sizes 
3.	Add more hardware topologies 
4.	Tighten pair matching on logical volume 
5.	Add execution-side/noise-derived observables

project/
├── experiments/
│   ├── exp1.py
│   ├── exp2.py
│   ├── exp3.py
│   └── exp4.py
├── quantum/
│   ├── topologies.py
│   ├── circuits.py
│   ├── controls.py
│   ├── transpilation.py
│   └── features.py
├── analysis/
│   ├── stats.py
│   ├── mi.py
│   ├── classifiers.py
│   └── paired_tests.py
├── viz/
│   └── plots.py
├── data/
│   └── builders.py
├── config.py
└── main.py
