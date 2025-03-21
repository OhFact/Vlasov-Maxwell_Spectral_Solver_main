# Vlasov-Maxwell Spectral Solver

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Running the Code](#running-the-code)
  - [Testing](#testing)

---

## Overview

This project simulates 1D and 3D evolving plasma distributions that start off roughly gaussian with a perturbation.
The project can be downloaded from its [GitHub repository](https://github.com/uwplasma/Vlasov-Maxwell_Spectral_Solver).

---

## Features

This Vlasov-Maxwell Spectral Solver runs using the JAX framework, utilizing optimizations within the module. It currently takes initial hermite coefficients or initial electric and magnetic fields, calculated by an initialization function, and evolves them using the Vlasov-Maxwell equations. You can see some examples that do this through the files in the `Examples` folder.

These parameters aim to model:
- Kelvin-Helmholtz Instabilities
- Landau Damping in 1D and 3D Cases
- Density Perturbations
- Pressure Anisotropy in 1D

The `JAX_VM_solver.py` file defines the necessary functions for modeling, while `Run_JAX_VM_solver.py` implements those functions in separate blocks for Landau Damping and Kelvin-Helmholtz Instabilities.

---

## Project Structure

```
└── Vlasov-Maxwell_Spectral-main
    ├── Examples
    │   ├── Examples_1D
    │   │   ├── density_perturbation_1D.py
    │   │   ├── Landau_Damping_1D.py
    │   │   ├── pressure_anisotropy_HF_1D.py
    │   │   └── Two_Stream_1D.py
    │   └── Examples_2D
    │   │   ├── Kelvin_Helmholtz_2D.py
    │   │   └── Orszag_Tang.py
    ├── ExampleParams
    │   ├── plasma_parameters_density_perturbation.json
    │   ├── plasma_parameters_Kelvin_Helmholtz_1D.json
    │   ├── plasma_parameters_Kelvin_Helmholtz_2D.json
    │   ├── plasma_parameters_Landau_damping_1D.json
    │   ├── plasma_parameters_Landau_damping_HF_1D.json
    │   └── pressure_anisotropy.json
    ├── JAX_Vlasov_Maxwell_solver
    │   ├── __init__.py
    │   ├── energy.py
    │   └── JAX_VM_solver.py
    ├── Run_JAX_VM_solver.py
    ├── requirements.txt
    └── CodeTests
        ├── LandauDampingRate.py
        ├── WolframCodeCSV
        │   ├── damping_rate_1000000.py
        │   └── omega.py
        ├── TestingHermite.py
        └── Compute_C_nmp_test.py
```

---

## Getting Started

### Prerequisites

This repository has requirements stored in [`requirements.txt`](requirements.txt).

### Running the Code

Once you have your prerequisites installed, navigate to the `Run_JAX_VM_solver.py` file and execute the program. Ensure that the correct parameters are set and that the appropriate sections of the code are enabled. Adjust which plots to view through the comments in the file, and if you want to print initial conditions, they can be dumped into a text file.

### Testing


---

