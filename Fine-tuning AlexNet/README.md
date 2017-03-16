### Fine-tuning AlexNet
This implements a modification of the fine-tuning code from [this][1] tutorial. Please follow the
tutorial to obtain the pre-trained weights and the training data.

`finetune.py`     - script for fine tuning the model <br/>
`model.py`        - AlexNet model for DogsVsCats classification <br/>

The parameters can be changed during finetuning from the command line by setting the proper flags:

`conv`          - Number of convolutional layers of the original AlexNet model to use (default=5) <br/>
`fc`            - Number of neurons in the fully connected layers (default=4096) <br/>
`batch_size`    - Batch size to be used during training (default=128) <br/>
`learning_rate` - The learning rate to be used during training (default=0.001) <br/>
`dropout_rate`  - The keep probability for neurons to be used for the dropout layer (default=0.5) <br/>
`additional`   - (0/1) If set to 1, it uses additional layers in the model. Allowed only if conv=5. (default=0) <br/>

The script, finetune.py, can be run as:

```
python finetune.py --conv=4 --dropout_rate=0.3
```

[1]: https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
