# space-engineers-research
A collection of research code for the Space Engineers project.

## Contents:
- Notebooks / Utilities:
    - `l-system`: Contains a notebook that shows how to place an arbitrary structure defined by an L System in Space Engineers. Additional information in the `README.md` file.
    - `steam-workshop-downloader`: Contains a notebook to download items from the Steam Workshop. Additional information in the `README.md` file.
    - `cppn-system`: Contains a notebook to train a CPPN to produce user content as downloaded from the Steam Workshop. Additional information in the `README.md` file.
- `PCGSEPy` package folders:
    - `common`: Contains a collection of modules that are shared across projects (mainly interfacing to the Space Engineers API).
    - `lsystem`: Contains a collection of modules that define data types and methods used n the L-System demo.
    - `config.py`: Contains shared configurations defined in the `configs.ini` file.
    - `structure.py`: Contains the definitions of the `Structure` and `Block` objects, as well as the basic methods for these objects.
    - `xml_conversion.py`: Contains the method to convert an XML file to its corresponding `Structure` object.