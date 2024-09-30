# Robotaxi User Study Environment
Game Logic Modified from repository: 
- [snake-ai-reinforcement](https://github.com/YuriyGuts/snake-ai-reinforcement)
- [RobotaxiEnv](https://github.com/Pearl-UTexas/RobotaxiEnv)

![robotaxi_env](https://github.com/kang1121/RoboTaxi/blob/main/.robotaxi_env.png)

## Requirements

- Recommanded: [install anaconda](https://docs.anaconda.com/anaconda/install/)
- Python 3.10 or above. 
- Install all Python dependencies, run:
```
conda create -n robotaxi python=3.10
conda activate robotaxi
python -m pip install --upgrade -r requirements.txt
conda install pyaudio seaborn 
```

## Play Robotaxi

Navigation
```
python play.py --mode navigation --level robotaxi/levels/23x23-obstacles.json --num-episodes 100
```

Surveillance
```
python play.py --mode surveillance --level robotaxi/levels/23x23-obstacles.json --num-episodes 100
```

