Mirror Descent Search
===

This is the experimental code for Mirror Descent Search in our paper -- [Mirror Descent Search and Acceleration](https://arxiv.org/abs/1709.02535).

## Dependency
* Python3
    * Numpy
    * Matplotlib
    * Scipy
    * Pyglet
    * PyQt4 / PyQt5

`pip install -r requirements.txt`

## Usage
`python main.py`

The output directory is created in `./mds_out/` with timestamp.

### Parameter
See parameter description.
`python main.py -h`
* Example: set agent AMDS
`python main.py --agent AMDS`

### Correspondence between this code and our paper
- G-AMDS in our paper: `--agent AMDS`
- G-MDS in our paper: `--agent GMDS`
- PI2 in our paper: `--agent PI2`
- REPS in our paper: `--agent REPS`

### Output file
Above files are created in the output directory:
- args.txt: Run arguments.
- command.txt: Run command.
- cost.csv: Cost per the number of rollouts.
- cost.pdf: Plot of correlation between cost and number of rollouts.
- cost.png: Plot of correlation between cost and number of rollouts.
- git-*.txt: Git command.
- trajectory.mp4: Movie of agent's trajectory.
- rollout.csv: The number of rollouts.
- x.csv: X-axis trajectory per time
- y.csv: Y-axis trajectory per time

### Rendering
If your environment has $DISPLAY, rendering is enable.
`python main.py --render`

## Reference
* [Policy Improvement with Path Integrals](http://www-clmc.usc.edu/software/git/gitweb.cgi?p=matlab/pi2.git) by Stefan Schaal
* [Dynamic Movement Primitives](http://www-clmc.usc.edu/software/git/gitweb.cgi?p=matlab/dmp.git) by Stefan Schaal
* [Accelerated Mirror Descent](https://github.com/walidk/AcceleratedMirrorDescent) by Walid Krichene
* [ChainerRL](https://github.com/chainer/chainerrl)
