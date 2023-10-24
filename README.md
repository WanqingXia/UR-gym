# UR-gym

Set of robotic environments based on PyBullet physics engine and gymnasium.
This repository is the reinforcement learning (SAC+HER) code for UR5e robot
Code developed base on panda-gym https://github.com/qgallouedec/panda-gym

## Installation
```bash
git clone https://github.com/WanqingXia/UR-gym.git
```

## Usage

### Simple demo (provided in demo.py)
```python
import sys
import gymnasium
sys.modules["gym"] = gymnasium
import UR_gym

env = gymnasium.make('UR5OriReach-v1', render="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

### Train the model
```python
python3 train.py
```

### Run our trained model
```python
python3 model_test.py
```