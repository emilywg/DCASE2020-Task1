# DCASE2020-Task1
Jupyter notebooks for [DCASE 2020 challenge Task 1]([http://dcase.community/challenge2020/task-acoustic-scene-classification)

## Training
Use `create_extra_labels.ipynb` to create new data labels for domain adaptation (Task 1a only).

Run `DCASE2020_Task1x_training.ipynb` to train the models.

The following block of script is used to save the model after training:
```
stamp = datetime.datetime.now().strftime('%y%m%d%H%M')
tag = stamp + '_' + WhichTask + '_' + MODE + '_'+ str(num_epochs)
savedir = os.path.join(os.getcwd(), tag)
print "Model path: %s" % savedir
try:
    os.makedirs(savedir)
except OSError:
    if not os.path.isdir(savedir):
        raise
```

Save checkpoints by specifying the weights of which epochs to be saved:
```
ckpt = ckpt(filepath=ckpt_path, ckpts=[70, 150])
```
The class `ckpt` is defined in `DCASE_training_functions.py`

## Test
Run `DCASE2020_Task1a_inference.ipynb` to test the model and to print out the confusion matrices
