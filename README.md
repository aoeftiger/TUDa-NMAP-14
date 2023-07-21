# Numerical Methods of Accelerator Physics

MSc lecture at TU Darmstadt, etit, TEMF by Adrian Oeftiger in 2022/23.

Fourteenth part of a jupyter notebook lecture series, guest lecture held by Dr. Andrea Santamaria Garcia and Chenran Xu on 10.02.2023.

Find the rendered HTML slides [here](https://aoeftiger.github.io/TUDa-NMAP-14/).

---

## Run online

Run this notebook talk online, interactively on mybinder.org:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aoeftiger/TUDa-NMAP-14/v1.1)

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

---

## Overview Lecture Series

1. [Lecture 01: basic concepts (accelerators, time scales, modelling a pendulum)](https://github.com/aoeftiger/TUDa-NMAP-01)
2. [Lecture 02: basic concepts (rms emittance, emittance preservation & filamentation, discrete frequency analysis & NAFF)](https://github.com/aoeftiger/TUDa-NMAP-02)
3. [Lecture 03: basic concepts (chaos and early indicators, numerical artefacts)](https://github.com/aoeftiger/TUDa-NMAP-03)
4. [Lecture 04: longitudinal beam dynamics (acceleration with rf cavities, longitudinal tracking equations)](https://github.com/aoeftiger/TUDa-NMAP-04)
5. [Lecture 05: longitudinal beam dynamics (Monte-Carlo technique, synchrotron Hamiltonian, phase space initialisation)](https://github.com/aoeftiger/TUDa-NMAP-05)
6. [Lecture 06: longitudinal beam dynamics (simulating transition crossing, equilibrium distributions, emittance growth)](https://github.com/aoeftiger/TUDa-NMAP-06)
7. [Lecture 07: transverse beam dynamics (dipole / quadrupole / sextupole magnetic fields, betatron matrices)](https://github.com/aoeftiger/TUDa-NMAP-07)
8. [Lecture 08: transverse beam dynamics (Hill differential equation, Floquet theory, optics, off-momentum particles)](https://github.com/aoeftiger/TUDa-NMAP-08)
9. [Lecture 09: longitudinal tomography](https://github.com/aoeftiger/TUDa-NMAP-09)
10. [Lecture 10: closed orbit correction (local and global)](https://github.com/aoeftiger/TUDa-NMAP-10)
11. [Lecture 11: reinforcement learning (Q-learning, actor-critic methods)](https://github.com/aoeftiger/TUDa-NMAP-11)
12. [Lecture 12: collective effects (space charge, lambda-prime model, microwave instability)](https://github.com/aoeftiger/TUDa-NMAP-12)
13. [Lecture 13: summary](https://github.com/aoeftiger/TUDa-NMAP-13)
14. [Lecture 14: Bayesian optimisation](https://github.com/aoeftiger/TUDa-NMAP-14)
