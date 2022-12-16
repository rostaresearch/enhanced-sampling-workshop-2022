# Enhanced Sampling Simulation Methods for Thermodynamics, Kinetics, and Pathways

### 19-20 December 2022

## Download

You can download the repository as a `.zip` archive here. If you choose to clone, we suggest to use a shallow copy to
ignore the extensive history of the repository by
```bash
$ git clone --depth=1 <url>
```

## Getting started

You need python3 and the setup is optimised for anaconda. If you need to install it, you find instructions
[here](https://docs.anaconda.com/anaconda/install/index.html).

Then create a new environment:
```bash
$ conda env create --name <env_name> --file conda_env.yml
```
Wait until it finishes, depending on your system it may take a couple of minutes. Activate and code away.
```bash
$ conda activate <env_name>
```
Most of the code is ordered in jupyter notebooks, so start with that.
```bash
$ jupyter notebook
```

----
### Notes
If you have not already, you can add the `conda-forge` channel to your database:
```bash
$ conda config --add channels conda-forge
```
