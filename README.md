# Experiments

## Movies

Single-task:  
![AirRaid (Single-task clipped)](movies/AirRaid_050009600_AirRaidNoFrameskip-v4.gif)
![Carnival (Single-task clipped)](movies/Carnival_050002560_CarnivalNoFrameskip-v4.gif)
![DemonAttack (Single-task clipped)](movies/DemonAttack_050001280_DemonAttackNoFrameskip-v4.gif)
![Pong (Single-task clipped)](movies/Pong_050013440_PongNoFrameskip-v4.gif)
![SpaceInvaders (Single-task clipped)](movies/SpaceInvaders_050001280_SpaceInvadersNoFrameskip-v4.gif)
  
Multi-task (clipped):  
![AirRaid (Multi-task clipped)](movies/MultiTask_300014720_AirRaidNoFrameskip-v4.gif)
![Carnival (Multi-task clipped)](movies/MultiTask_300014720_CarnivalNoFrameskip-v4.gif)
![DemonAttack (Multi-task clipped)](movies/MultiTask_300014720_DemonAttackNoFrameskip-v4.gif)
![Pong (Multi-task clipped)](movies/MultiTask_300014720_PongNoFrameskip-v4.gif)
![SpaceInvaders (Multi-task clipped)](movies/MultiTask_300014720_SpaceInvadersNoFrameskip-v4.gif)
  
Multi-task PopArt:  
![AirRaid (Multi-task PopArt)](movies/MultiTaskPopart_300010240_AirRaidNoFrameskip-v4.gif)
![Carnival (Multi-task PopArt)](movies/MultiTaskPopart_300010240_CarnivalNoFrameskip-v4.gif)
![DemonAttack (Multi-task PopArt)](movies/MultiTaskPopart_300010240_DemonAttackNoFrameskip-v4.gif)
![Pong (Multi-task PopArt)](movies/MultiTaskPopart_300010240_PongNoFrameskip-v4.gif)
![SpaceInvaders (Multi-task PopArt)](movies/MultiTaskPopart_300010240_SpaceInvadersNoFrameskip-v4.gif)

Saliency:
![AirRaid](movies/Saliency_AirRaidNoFrameskip-v4.gif)
![Carnival](movies/Saliency_CarnivalNoFrameskip-v4.gif)
![DemonAttack](movies/Saliency_DemonAttackNoFrameskip-v4.gif)
![Pong](movies/Saliency_PongNoFrameskip-v4.gif)
![SpaceInvaders)](movies/Saliency_SpaceInvadersNoFrameskip-v4.gif)


## Trained models

| Name | Environments | Steps (millions) |
| ---- |------------- | ---------------- |
| AirRaid | AirRaidNoFrameskip-v4 | 50 |
| Carnival | CarnivalNoFrameskip-v4  | 50 |
| DemonAttack | DemonAttackNoFrameskip-v4 | 50 |
| NameThisGame| ameThisGameNoFrameskip-v4 | 50 |
| Pong| PongNoFrameskip-v4 | 50 |
| SpaceInvaders | SpaceInvadersNoFrameskip-v4 | 50 |
| MultiTask | AirRaid,Carnival,DemonAttack,NameThisGame,Pong,SpaceInvaders NoFrameskip-v4 | 300 |
| MultiTask3Games | Carnival,DemonAttack,AirRaid NoFrameskip-v4 | 100 |
| MultiTaskPopArt | AirRaid,Carnival,DemonAttack,NameThisGame,Pong,SpaceInvaders NoFrameskip-v4 | 300 |


# Running the code
# Train
```bash
python -m torchbeast.polybeast --mode test --xpid MultiTask --env PongNoFrameskip-v4 --savedir=./logs/torchbeast
python -m torchbeast.polybeast --mode test_render --xpid MultiTask --env PongNoFrameskip-v4 --savedir=./logs/torchbeast
```


# Test
```bash
python -m torchbeast.polybeast --mode test --xpid MultiTask --env PongNoFrameskip-v4 --savedir=./logs/torchbeast
python -m torchbeast.polybeast --mode test_render --xpid MultiTask --env PongNoFrameskip-v4 --savedir=./logs/torchbeast
```

# Saliency
```bash
python -m torchbeast.saliency --xpid MultiTask --env PongNoFrameskip-v4 --resolution=75 --num_frames 5 --savedir=./logs/torchbeast
```

## References
TorchBeast
```
@article{torchbeast2019,
  title={{TorchBeast: A PyTorch Platform for Distributed RL}},
  author={Heinrich K\"{u}ttler and Nantas Nardelli and Thibaut Lavril and Marco Selvatici and Viswanath Sivakumar and Tim Rockt\"{a}schel and Edward Grefenstette},
  year={2019},
  journal={arXiv preprint arXiv:1910.03552},
  url={https://github.com/facebookresearch/torchbeast},
}
```
```
@article{greydanus2017visualizing,
  title={Visualizing and Understanding Atari Agents},
  author={Greydanus, Sam and Koul, Anurag and Dodge, Jonathan and Fern, Alan},
  journal={arXiv preprint arXiv:1711.00138},
  year={2017},
  url={https://github.com/greydanus/visualize_atari},
}
```
