# ELECTS: End-to-End Learned Early Classification of Time Series for In-Season Crop Type Mapping

<img width="100%" src="png/elects.png">

please cite
> Marc Rußwurm, Nicolas Courty, Remi Emonet, Sebastien Lefévre, Devis Tuia, and Romain Tavenard (2023). End-to-End Learned Early Classification of Time Series for In-Season Crop Type Mapping. To appear in ISPRS Journal of Photogrammetry and Remote Sensing.

preprint available at https://arxiv.org/pdf/1901.10681.pdf

## Dependencies

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Getting Started:

Test model predictions on the evaluation set with Jupyter
Notebook provided in `elects.ipynb`

<img height="200px" src="./png/elects_notebook.png">

## Run Training Loop

### Monitor training visally (optional)

start [visdom](https://github.com/fossasia/visdom) server for visual training progress
```bash
❯ visdom
Checking for scripts.
It's Alive!
INFO:root:Application Started
You can navigate to http://localhost:8097
```
and navigate to [http://localhost:8097/](http://localhost:8097/) in the browser of your choice.

<img height="200px" src="./png/visdom.png">

### Start training loop

To start the training loop run
```
❯ python train.py
Setting up a new session...
epoch 100: trainloss 1.70, testloss 1.97, accuracy 0.87, earliness 0.48. classification loss 7.43, earliness reward 3.48: 100%|███| 100/100 [06:34<00:00,  3.95s/it]
```
The BavarianCrops dataset is automatically downloaded.
Additional options (e.g., `--alpha`, `--epsilon`, `--batchsize`) are available with `python train.py --help`.

## Docker

It is also possible to install dependencies in a docker environment
```
docker build -t elects .
```
and run the training script
```
docker run elects python train.py
```


python train.py --dataroot /data/sustainbench --dataset ghana
python train.py --dataroot /data/sustainbench --dataset southsudan

--dataroot /data/sustainbench --dataset southsudan --epochs 500