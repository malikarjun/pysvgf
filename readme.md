# Offline A-SVGF using Jax

This is a differentiable implementation of the [A-SVGF](https://cg.ivd.kit.edu/publications/2018/adaptive_temporal_filtering/adaptive_temporal_filtering.pdf) paper using Jax.

# Installation
## Python modules
```
pip install --upgrade pip
pip install requirements.txt
pip install --upgrade "jax[cpu]"
```
## Input data
LaJolla is used to generate data. [This](https://github.com/malikarjun/lajolla/tree/pysvgf#generate-data-for-differentiable-a-svgf) doc can be referred for more details.


# Run
`python main.py`
The result will be in the `output` folder.