# shape_pusher_env

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
