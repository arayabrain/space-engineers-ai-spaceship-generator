# 🚀 Space Engineers AI Spaceship Generator 🚀
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Lines of code](https://img.shields.io/tokei/lines/github/arayabrain/space-engineers-research)
![GitHub issues](https://img.shields.io/github/issues-raw/arayabrain/space-engineers-research)
![GitHub forks](https://img.shields.io/github/forks/arayabrain/space-engineers-research?style=social)
![GitHub Repo stars](https://img.shields.io/github/stars/arayabrain/space-engineers-research?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/arayabrain/space-engineers-research)

A collection of research code for the Space Engineers PCG project developed under the GoodAI research grant.

## Installation
Create the executable file by running the `build_main_webapp.bat` file inside the `user-study` folder. The executable file (`Space Engineers PCG.exe`) will be created in the `user-study\dist` folder.

## How to use the webapp
_TODO: write this section of the README._

## Development
Install the `PCGSEPy` library by first installing the requirements:
```
pip install -r requirements.txt
```
And then install the library:
```
pip install -e .
```

## Contents overview
- `pcgsepy`: this directory contains the main Python PCGSEPy library.
- `steam-workshop-downloader`: this directory contains the code used to download the spaceships from the Steam Workshop and extract the metrics of interest.
- `l-system`: this directory contains the code used for the L-system and FI-2Pop experiments.
- `icmap-elites`: this directory contains the code used for the Interactive Constrained MAP-Elites experiments and the user study.
- `user-study`: this directory contains additional code used in the user study.

For more information, refer to the `README`s in each directory.

## Publications
This library was used in the following publications:
```
Roberto Gallotta, Kai Arulkumaran, and L. B. Soros. 2022. Evolving spaceships with a hybrid L-system constrained optimisation evolutionary algorithm. In Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO '22). Association for Computing Machinery, New York, NY, USA, 711–714. https://doi.org/10.1145/3520304.3528775
```
```
Gallotta, Roberto, Kai Arulkumaran and Lisa B. Soros. “Surrogate Infeasible Fitness Acquirement FI-2Pop for Procedural Content Generation.” ArXiv abs/2205.05834 (2022): 4 pag.
```