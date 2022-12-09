# enhanced-sampling-workshop-2022

## Don't forget to update the `requirements.txt` before publishing

## Getting started

You need python3 and the setup is optimised for anaconda. If you have not already, please add the `conda-forge` channel
to your database:
```bash
$ conda config --add channels conda-forge
```
Then create a new environment:
```bash
$ conda create --name <env_name> --file requirements.txt
```
Wait until it finishes, depending on your system it may take a couple of minutes. Activate and code away.
```bash
$ conda activate <env_name>
```
Most of the code is ordered in jupyter notebooks, so start with that.
```bash
$ jupyter notebook
```