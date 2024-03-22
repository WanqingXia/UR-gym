# UR-gym

Set of robotic environments based on PyBullet physics engine and gymnasium.
This repository is the code for training and testing UR5e robot with Soft-Actor Critic algorithm. Our paper have been accepted by SME NAMRC52 and waiting to be published.
Code developed base on panda-gym https://github.com/qgallouedec/panda-gym

In this project, we constructed four environments for the UR5e robot, they are: UR5OriReach-v1, UR5ObsReach-v1, UR5StaReach-v1, UR5DynReach-v1.
UR5OriReach-v1: Train the robot to reach a designated coordinate aligning both position and orientation
UR5ObsReach-v1: Train the robot to reach a designated coordinate aligning position only in a environment with static obstacle.
UR5StaReach-v1: Train the robot to reach a designated coordinate aligning both position and orientation in a environment with static obstacle.
UR5DynReach-v1: Train the robot to reach a designated coordinate aligning both position and orientation in a environment with dynamic obstacle.
```
[![IMAGE ALT TEXT HERE](./UR_gym/assets/cover.png)](https://www.youtube.com/watch?v=Tq03JQw-MHw)
```

## Installation
```bash
git clone https://github.com/WanqingXia/UR-gym.git
```

## Project Structure

<pre>
UR-gym
|──Trained_Models
	|──Trained_Dyn $${\color{red}# Trained model with UR5DynReach-v1 environment}$$
	|──Trained_Obs # Trained model with UR5ObsReach-v1 environment
	|──Trained_Ori # Trained model with UR5OriReach-v1 environment
	└──Trained_Sta # Trained model with UR5StaReach-v1 environment
|──UR_gym # Main code for environment construction
	|──assets
	|──envs
		|──robots
			|──meshes
			|──urdf
			└──UR5.py
		|──tasks
			└──reach.py
		|──core.py
		└──ur_tasks.py
	|──pyb_setup.py
	└──utils.py
|──utils
	|──callbackFunctions.py
	└──generate.py
|──demo.py # demo code
|──model_test.py
|──robot_show.py
|──setup.py
|──show_traj.py
└──train.py
</pre>
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
# Train a SAC controlled UR5e robot to avoid dynamic obstacles
python train.py
```

### Run our trained model
```python
python model_test.py
```