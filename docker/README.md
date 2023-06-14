# Docker instructions
In this folder run 
```
docker build -t petmrac:v0 .
```
to build the docker image. To run the container run 
```
docker run -it --rm --gpus=all -p 5001:6006 -v "$PWD/..:/workspace" petmrac:v0
```
You may select different ports in the `-p` flag which will be used for a tensorboard instance. If you are still in the docker folder when this command is run it will bind the main repository folder to the docker container. 

To run a tensorboard at an external server at the selected port run 
```
tensorboard --logdir checkpoints --port 6006 --bind_all &
```
this wil run the tensorboard in the backgorund so we can run training/inference.