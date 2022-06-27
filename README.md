# space-engineers-research
A collection of research code for the Space Engineers project.

## Contents:
- Notebooks / Utilities:
    - `l-system`: Contains a notebook that shows how to place an arbitrary structure defined by an L System in Space Engineers. Additional information in the `README.md` file. All the code was used for the "Evolving Spaceships with a Hybrid L-system Constrained Optimisation Evolutionary Algorithm" poster presented at GECCO 2022.
    - `steam-workshop-downloader`: Contains a notebook to download items from the Steam Workshop. Additional information in the `README.md` file.
    - `cppn-system`: Contains a notebook to train a CPPN to produce user content as downloaded from the Steam Workshop. Additional information in the `README.md` file. Currently not in development.
    - `icmap-elites`: Contains notebooks to run an interactive CMAP-Elites webapp, as well as the results for the "Surrogate Infeasible Fitness Acquirement FI-2Pop for Procedural Content Generation" poster presented at CoG 2022. Additional information in the `README.md` file.
- `PCGSEPy` package folders:
    - `common`: Contains a collection of modules that are shared across projects (mainly interfacing to the Space Engineers API).
    - `evo`: Contains all code for anything EC.
    - `fi2pop`: Contains all code and necessary components for the `FI-2Pop` GA.
    - `guis`: Contains the web applications for interfacing with the CMAP-Elites object as well as the interfaces for the user study.
    - `lsystem`: Contains a collection of modules that define data types and methods used n the L-System demo.
    - `mapelites`: Contains the code for all necessary components of a `CMAP-Elites` system.
    - `nn`: Contains the `PyTorch` code for defining and training neural networks.
    - `config.py`: Contains shared configurations defined in the `configs.ini` file.
    - `hullbuilder.py`: Defines the `HullBuilder` class and its methods.
    - `setup_utils.py`: Contains methods to define a basic L-system as well as configuring `matplotlib`
    - `structure.py`: Contains the definitions of the `Structure` and `Block` objects, as well as the basic methods for these objects.
    - `xml_conversion.py`: Contains the method to convert an XML file to its corresponding `Structure` object.