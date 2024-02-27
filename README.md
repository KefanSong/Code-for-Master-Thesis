Implementation for Multi-Task Group Fairness Reinforcement Learning


### Requirements
[pytorch >= 1.3.1](https://pytorch.org/) <br>
[gymnasium >= 0.27.0](https://github.com/openai/gym](https://gymnasium.farama.org/gymnasium_release_notes/index.html)) <br>
[mujoco >= 3.1.2](https://github.com/openai/mujoco-py](https://mujoco.org)) <br>



### Implementation
Example: HalfCheetah group and BigFoot HalfCheetah group under two tasks, one with default reward, the other with added penalty for action magnitude.

```
python3 group_fairness_one_task.py --env-id='big_foot_half_cheetah' --batch-size=1024 --constraint='group fairness' --max-iter-num=100 --seed=1 --group-fairness-threshold=500

```


Return plots of one seed is shown in the file [Plotting Return.ipynb](https://github.com/KefanSong/Multi-Task-Group-Fairness-Reinforcement-Learning/blob/main/Plotting%20Return.ipynb)

The implementation of group fairness reinforcement learning in a single task is in the file [group_fairness_one_task.py](https://github.com/KefanSong/Code-for-Master-Thesis/blob/main/group_fairness_one_task.py)
