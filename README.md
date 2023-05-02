# 🚀 Space Engineers AI Spaceship Generator 🚀
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Lines of code](https://img.shields.io/tokei/lines/github/arayabrain/space-engineers-ai-spaceship-generator)
![GitHub issues](https://img.shields.io/github/issues-raw/arayabrain/space-engineers-ai-spaceship-generator)
![GitHub forks](https://img.shields.io/github/forks/arayabrain/space-engineers-ai-spaceship-generator?style=social)
![GitHub Repo stars](https://img.shields.io/github/stars/arayabrain/space-engineers-ai-spaceship-generator?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/arayabrain/space-engineers-ai-spaceship-generator)

<p align="center">
  <img src="media/pcgsepy_banner.png" alt="pcgsepy_banner" height="120"/>
</p>

### Download

#### Download AI Spaceship Generator for Windows from [the releases page](https://github.com/GoodAI/space-engineers-ai-spaceship-generator/releases).

### Introduction

Apps and code (`PCGSEPy` library) developed for the [Space Engineers](https://www.spaceengineersgame.com/) PCG project, supported by a [GoodAI research grant](https://www.goodai.com/using-open-ended-algorithms-to-generate-video-game-content-in-space-engineers/). The main app is the AI Spaceship Generator, which creates spaceships for Space Engineers using AI (for more information, see our [research explainer](#research) and [publications](#publications)). The second app is a Spaceships Ranker, which is used for conducting a user study. The AI Spaceship Generator and Spaceships Ranker apps are available for Windows on the [releases page](https://github.com/GoodAI/space-engineers-ai-spaceship-generator/releases).

## Apps
The following is a quick overview of the apps (AI Spaceship Generator and Spaceships Ranker). Further documentation is available within the apps themselves.

### AI Spaceship Generator (user mode)
<p align="center">
  <img src="media/UI_usermode_preview.png" alt="ui_usermode_preview" height="300"/>
</p>
The default mode for the app (outside of the user study period). The AI generates an initial "population" of spaceships (top left). When a spaceship is selected from the population it is visualised (top middle) and its properties are displayed (top right). You can choose to "evolve" a new set of spaceships based on either the selected spaceship or a random spaceship (the "evolution" process tries to construct new spaceships based on an existing spaceship). You can also re-initialise the population of spaceships.

### AI Spaceship Generator (user study mode)
<p align="center">
  <img src="media/UI_userstudy_preview.jpg" alt="ui_userstudy_preview" height="300"/>
</p>
The default mode for the app (during the user study period). You evolve spaceships for a fixed number of iterations, for different configurations of the AI system. At the end of the user study, the app automatically switches to user mode. The data collected from the user study will be used to improve the AI system in the next app release.

### AI Spaceship Generator (developer mode)
<p align="center">
  <img src="media/UI_devmode_preview.jpg" alt="ui_devmode_preview" height="300"/>
</p>
An advanced mode, with full access to every part of the system that can be changed during the evolution process.

### Spaceships Ranker
<p align="center">
  <img src="media/UI_comparator_preview.jpg" alt="ui_comparator_preview" height="300"/>
</p>
This app is used for the user study. You can upload and rank different spaceships from the AI Spaceship Generator.

## Roadmap
This project will be actively supported until the end of December 2022.

The user study is planned to run from the duration of November 2022.

## Development
This project requires Python 3. The `PCGSEPy` library (including its requirements) can be installed by running `pip install -e .`. To use PyTorch in the library (required for some research experiments, but not the apps), first set the `use_torch` flag in `configs.ini`.

We recommend creating a conda environment for the application. Make sure to install [orca](https://github.com/plotly/orca) via `conda install -c plotly plotly-orca` in case the spaceship download hangs (spinner on the top right remains visible for over 30 seconds to a minute).

### Building the apps
The apps can be built using the provided `.py` files placed in the `user-study` folder. The executable files will be created in the `user-study\dist` folder.

The AI Spaceship Generator can be built by running `python build_main_webapp.py`. The Spaceships Ranker can be built by running `python build_comparator.py`.

### Documentation
An explorable documentation is provided in `docs`. You can build the documentation by first installing `pdoc3` (`pip install pdoc3`) and then running the command `pdoc --html pcgsepy --force --output-dir  ./docs`.

#### Modifiable files
Some files can be modified before building the apps. These are:
- The estimators under the `estimators` folders: these are `.pkl` files and can be generated by running the `steam-workshop-downloader\spaceships-analyzer.ipynb` notebook (additional details are provided in the notebook itself).
- The `configs.ini` file: different settings can be specified in this file, making sure to respect the setting formats specified in `pcgsepy\config.py`.
- `block_definitions.json`: this file can be excluded from the application building file (`user-study\build_main_webapp.bat`), but it is required to run the application. It can be recreated when the application is first launched if an instance of Space Engineers is open and the [iv4xr API](https://github.com/iv4xr-project/iv4xr-se-plugin) is installed.
- `hlrules`, `llrules`: these files define the expansion rules of the underlying L-system used in the application. `hlrules` determines the tile placements when creating spaceships, whereas `llrules` determines the game blocks used in a tile. If you want to add your own tiles in the application, please follow the instructions reported in `l-system\rules-extractor.ipynb` and remember to also update the `hl_atoms.json` file.

### Codebase overview
- `pcgsepy`: this directory contains the main Python PCGSEPy library.
- `steam-workshop-downloader`: this directory contains the code used to download the spaceships from the Steam Workshop and extract the metrics of interest.
- `l-system`: this directory contains the code used for the L-system and FI-2Pop experiments.
- `icmap-elites`: this directory contains the code used for the Interactive Constrained MAP-Elites experiments and the user study.
- `user-study`: this directory contains additional code used in the user study.

For more information, refer to the `README`s in each directory.

## Research

The AI Spaceship Generator consists of several components, allowing it to generate both functional and aesthetically-pleasing spaceships for Space Engineers. The basis is a rule-based procedural content generation algorithm that generates working ships, which is combined with an evolutionary algorithm that optimises their appearance, resulting in a novel hybrid evolutionary algorithm. The basic heuristics for the evolutionary algorithm to optimise were derived by analysing the properties of spaceships available on the Space Engineers’ Steam Workshop page.

This is then combined with a novel algorithm that finds a diverse set of spaceships, making more choices available. Finally, we have built a graphical interface for these algorithms so that you can influence the spaceship generator’s choices, alongside another novel algorithm that tries to tune the system to better follow the your ideas.

Starting from an initial population of vessels, you can select a spaceship to "evolve" - the underlying evolutionary algorithm then "mutates" this, creating new spaceships with similar properties (shape, size, etc.). This process can be repeated to continuously generate new spaceships.

## Publications

> [Gallotta, R., Arulkumaran, K., & Soros, L. B. (2022). Evolving Spaceships with a Hybrid L-system Constrained Optimisation Evolutionary Algorithm. In _Genetic and Evolutionary Computation Conference Companion_.](https://dl.acm.org/doi/abs/10.1145/3520304.3528775)

> [Gallotta, R., Arulkumaran, K., & Soros, L. B. (2022). Surrogate Infeasible Fitness Acquirement FI-2Pop for Procedural Content Generation. In _IEEE Conference on Games_.](https://ieeexplore.ieee.org/document/9893592)

> [Gallotta, R., Arulkumaran, K., & Soros, L. B. (2022). Preference-Learning Emitters for Mixed-Initiative Quality-Diversity Algorithms. _IEEE Transactions on Games_.](https://doi.org/10.1109/TG.2023.3264457)
