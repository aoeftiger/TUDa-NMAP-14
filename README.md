# Numerical Methods of Accelerator Physics

MSc lecture at TU Darmstadt, etit, TEMF by Adrian Oeftiger in 2022/23.

Fourteenth part of a jupyter notebook lecture series, guest lecture held by Dr. Andrea Santamaria Garcia on 10.02.2023.

Find the rendered HTML slides [here](https://aoeftiger.github.io/TUDa-NMAP-14/).

---

## Run online

Run this notebook talk online, interactively on mybinder.org:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aoeftiger/TUDa-NMAP-14/v1.0)

The `lecture.ipynb` notebook will work out-of-the-box.

---

## Run on TU Darmstadt jupyterhub

If you have a TU ID, access the [local TU Darmstadt jupyterhub](https://tu-jupyter-i.ca.hrz.tu-darmstadt.de/) using your TU ID.

A possible way to upload and run this lecture repository is the following:

1. Open a terminal by clicking on the top right "New" -> "Terminal".

2. A new tab opens with a terminal, click into the black area and enter (copy&pasting):

``` bash
wget https://github.com/aoeftiger/TUDa-NMAP-14/archive/refs/heads/main.zip
unzip main.zip
cd TUDa-NMAP-14-main
```

3. You have downloaded, unzipped and entered the lecture repository. As a last step, install the dependencies:

``` bash
export TMPDIR="`pwd`"
pip install -r requirements_noversions.txt --prefix="`pwd`"/requirements
``` 

Close the terminal tab and open the `lecture.ipynb` notebook inside the repository directory on the jupyterhub main page.

---

## Run locally

The notebook can of course also be run on your local computer using your own jupyter notebook server. Install such an environment e.g. via the extensive [Anaconda distribution](https://www.anaconda.com/products/distribution), the minimalistic [Miniconda distribution](https://docs.conda.io/en/main/miniconda.html) or the extremely fast [Mamba package manager](https://mamba.readthedocs.io/en/latest/). (The order indicates preference by simplicity in installation and usage.)

You may find all required packages in the `requirements.txt` file.
