The code included implements fine-tuning of the AlexNet for 2 separate classification tasks: 

finetune.py     - script for fine tuning the model
model.py        - AlexNet model for DogsVsCats classification
model_h_e.py    - AlexNet model for Cancer Recurrence Prediction task on H&E stained images with architectural changes

The parameters can be changed during finetuning from the command line by setting the proper flags:

--conv          - Number of convolutional layers of the original AlexNet model to use (default=5)
--fc            - Number of neurons in the fully connected layers (default=4096)
--batch_size    - Batch size to be used during training (default=128)
--learning_rate - The learning rate to be used during training (default=0.001)
--dropout_rate  - The keep probability for neurons to be used for the dropout layer (default=0.5)
--additional    - (0/1) If set to 1, it uses additional layers in the model. Allowed only if conv=5. (default=0)

Instead of passing the parameters from the command line, the lines in the 'main' function of 'finetune.py' can be
uncommented to do a complete grid search over a range of parameters. Due to computational restrictions, the model has
only been tested on the parameters as present in the file, the graphical results of which are present alongwith the code.
The trained models for reproducing the results are also available and can be provided if required. Also, noteworthy is that 
adding another additional shallow network to the model has increased the accuracy after training for 4 epochs.
The script, finetune.py, can be run as:

python finetune.py --conv=4 --dropout_rate=0.3

Since the pre-trained model has weights for image size 227x227 and the architecture reduces this size significantly (while 
increasing the number of feature maps, for eg. the 11x11 convolution along with a stride of 4 reduces the image size considerably),
the architecture had to be changed from the first layer onwards for an input size of 51x51, or else it would lead to a considerable 
loss of information. Hence, the pre-trained weights cannot be directly used in this case. This is why the analysis has been done on 
another dataset - Dogs Vs Cats (kaggle competition). However, these are the changes that I propose for the cancer recurrence prediction
task:

- filters_layer1 = [5, 5, 96] and using stride of 1 everywhere except the first pooling layer (This is done so that we don't lose relevant
  information by taking too large a stride / filter size).
- Using 1x1 convolutions to make the model deeper and reducing the number of parameters as well.
- Using a grid search to choose the best dropout_rate.
- Experimenting with the addition of momentum.
- Experimenting between padding='SAME' and padding='VALID'
- Using average pooling for the deeper layer as the input to that layer has a lower size and each pixel might contain some useful information.
- Choosing lesser number of neurons in the fully connected layer (default=1024). This is done to reduce overfitting since we'll have less data than what
  the original model was trained on.
  
The file 'model_h_e.py' implements some of the proposed changes.
  
