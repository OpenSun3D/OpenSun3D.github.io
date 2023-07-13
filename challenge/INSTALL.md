# Install


## Create virutal environment

```
conda create --name opensun3d python=3.8
conda activate opensun3d
pip install pyviz3d numpy pandas opencv-python scipy
```

## Download example scene

```
python download_data_opensun3d.py --data_type=challenge_development_set --download_dir=data
```

## Visualize scene 

```
python opensun3d_challenge/demo_dataloader_lowres.py
```
