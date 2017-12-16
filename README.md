# ShapePusherEnv
Simple mujoco environment where a sphere agent pushes objects of various shapes and colors.

Example observations of this environment (8 observations of different trajectories are shown).

![Alt Text](https://github.com/alexlee-gk/shape_pusher_env/raw/master/images/example.gif)

## Install mujoco-py dependency
For correct behavior of setting colors, apply patch to mujoco-py source and install from source.
```
git clone --branch 0.5.7 git@github.com:openai/mujoco-py.git
cd mujoco-py
git apply ../mujoco-py.patch
python setup.py install
```

## Install other dependencies
```
pip install -r requirements.txt
```
