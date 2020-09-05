# Efficient Sampling of Equilibrium States Using Boltzmann Generators (CS 230 Final Project) 
Final project for Stanford CS 230 - Deep learning (Autumn 2019). Our goal is to apply [boltzmann generators](https://arxiv.org/abs/1812.01729) to new molecular systems. A description of our progress on this project can be found in [our project writeup](https://github.com/jbinagia/CS-230-Final-Project/blob/master/CS_230_Final_Report.pdf).

## Installation and Usage
To view our analysis, one can perform the following steps. 
- Activate a virtual environment and install the required libraries, e.g.:
```shell
virtualenv -p python .env
source .env/bin/activate 
pip install -r requirements.txt
```
- If using Anaconda, the commands are: 
```shell
conda create -n .env pip
conda activate .conda-env
pip install -r requirements.txt
```
- Then open up one of the Jupyter notebooks located in the [notebooks](https://github.com/jbinagia/CS-230-Final-Project/tree/master/notebooks) folder, such as `double_well.ipynb' or 'analytical_example.ipynb'
