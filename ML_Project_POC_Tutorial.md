# Tutorial Project OutLine

- ML Project Scoping
    - Identify The Scope of the Project's Problem Space and it's qualities
    - Identify The Scope of the Project's Solution Space and it's qualities
- Project POC Prep
    - Installing libs
    - Remove Old Files
    - Import libs
- ML Steps:
    1. Download and load data
    2. Dataset Exploration with Dataset Summarries and Visualizations
    3. Select Preprocessing Steps
    4. Model Selection, Hyperparameter Tuning, Monitoring Training
    5. (Model data drift Testing)   
    6. (Model Serving and Monitoring)
- Development Steps:
    1. Hard code a bare bones proof-of-concept (POC) for an ML Project, within the identified project scope, in a single jupyter notebook file. 
    2. Define functions for repeatable lines of code, define variables for controling previously hardcoded settings,improve naming conventions
    3. Add variables to control previously hard coded settings
    4. move functions into seperate files to shorten and siplify the main file. 
    5. Add or update Documentation
    6. Add more Features to the project That are relavent to the identified project scope starting from step one if more improvements are needed
- Project Management Steps:
    - Setup a Project Dashboard for the github repo.
    - add meaningful views to the dashboard to help orginize the project.
    - setup auto email alerts for team members assigned to issues when updates are posted.
    - Add issues and set the priority, size, sprint, labels, tags, and more if needed.
    - Set and enforce standards for team members posting to the project dashboard.
    - Set Rules for the repo to protect the main branch from updates without reviews from more than one person
    - Manually assign project issues for team members to work on or allow self assigning issues for a more hands free approach to managing the team.
    - Regularly sceduled meetings with your team and with other managers from other departments.
    - Allways encourage team members to ask questions for clarity on the goal of a tasks and to ask for more directions if they get stuck.

# ML Project Scoping

## Identify The Scope of the Project's Problem Space and it's qualities:

- what tyoe of data are you working with?
    - Structured or unstructured?
    - labeled or unlabeled? (for supervised and unsupervised learning models and methods)
    - shift invariant features? examples include: sound (time), images(space), video(time and space)
    - Time Series data? examples include: text, events, sound, video 
- What task is your model going to be performing?
    - classification
    - linear regresion
    - segmentation
    - diffusion content generation
    - combinations of the above options
- how relavent is each of the dataset's features in determining the output task and how are the features related?
    - should you be worried about feature selection or dimentionality reduction?
    - have you considered feature engineering options?
    - is there any missing info in any of your examples?
- how many examples does your dataset have?
- Are there known class imbalence or data drift problems? do you need to explore your data or consult a domain expert to find out?
- Are there more data that you can aquire?
    - internal sources: What other data features could be useful to add but are either not currently being recorded or are being recorded somewhere else but not in the database you are using?
    - external sources: open source datasets, dataset venders, and web scaping.

## Identify the Scope of the Project's Solution Space and it's qualities:

- Identify and ML and deep ml models that are well suited for the Scope of the Problem Space and it's qualities.
- Identify checkpoints of deep models from the solution space that were previously trained for different tasks that could be used for transfer learning.
- Identify preprocessing options that are well suited for the scope of the Problem Space and Solution Space.

# Project Prep

## Base Requirements:

Follow tensorflow install guide. Some commands my be different depending on your OS. 

I used Windows so if you are using linux, update the commands in the "Remove Old Files" section for your OS. 

## Installing libs 

create envirnment... add this later...


```python
# add env creation here...
```

start env


```python
#C:\Users\gilbr\ML_Proj_Code\Proj_1_venv\Scripts\Activate.ps1
```


```python
#c:\Users\gilbr\ML_Proj_Code\Proj_1_venv\Scripts\python.exe -m pip install pandas
```

- You can update this later to use a requirements.txt file.
- easy to save and install with pip commands
- can hide in repo 


```python
#pip freeze > requirements.txt
#pip install -r "C:Users\rg\MLProjTutorial\requirements.txt"
```

## Remove Old Files

Clear out the old logs and saves with cmd commands


```python
rmdir /s /q C:\Users\gilbr\ML_Proj_Code\logs
```


```python
rmdir /Q /S C:\Users\gilbr\ML_Proj_Code\saves
```


```python
mkdir C:\Users\gilbr\ML_Proj_Code\saves
```

## Import libs


```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorboard.plugins.hparams import api as hp
import keras
from keras import layers
import numpy as np
import datetime
%load_ext tensorboard
```

    WARNING:tensorflow:From C:\Users\gilbr\ML_Proj_Code\Proj_1_venv\Lib\site-packages\keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.
    
    TensorFlow version: 2.15.0
    

# ML Steps

## Download and load data 

For this project we will use the MNIST Dataset from Tensorflow.

In other projects this step could also involve:
- Querrying data from a database with SQL and or reading data from a file or files in a directory.
- loading data into a dataframe


```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```

## Dataset Exploration with Dataset Summarries and Visualizations:

### Part 1: Print out basic dataset info


```python
print('Skipping: print("x_train:") print(x_train) print()')
#print("x_train:") 
#print(x_train) 
#print()
print()
print("Traning Image Shape:") 
print(x_train.shape) 
print()
print("Traning Image dtype:") 
print(x_train.dtype) 
print()
print("y_train:")
print(y_train)
print()
print("Traning Label Shape:")
print(y_train.shape)
print()
print("Traning Label dtype:")
print(y_train.dtype)
print()
print('Skipping: print("x_test:") print(x_test) print()')
#print("x_test:") 
#print(x_test) 
#print()
print()
print("Testing Image Shape:")
print(x_test.shape)
print()
print("Testing Image dtype:")
print(x_test.dtype)
print()
print("y_test:")
print(y_test)
print()
print("Testing Label Shape:")
print(y_test.shape)
print()
print("Testing Label dtype:")
print(y_test.dtype)
print()
```

    Skipping: print("x_train:") print(x_train) print()
    
    Traning Image Shape:
    (60000, 28, 28)
    
    Traning Image dtype:
    uint8
    
    y_train:
    [5 0 4 ... 5 6 8]
    
    Traning Label Shape:
    (60000,)
    
    Traning Label dtype:
    uint8
    
    Skipping: print("x_test:") print(x_test) print()
    
    Testing Image Shape:
    (10000, 28, 28)
    
    Testing Image dtype:
    uint8
    
    y_test:
    [7 2 1 ... 4 5 6]
    
    Testing Label Shape:
    (10000,)
    
    Testing Label dtype:
    uint8
    
    

### Part 2:



## Preprocessing




```python
x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0
#x_train, x_test = x_train / 255.0, x_test / 255.0
```

## Model Selection, Tuning and Moitoring:

### Create some example model defs (Intial Model Selection)


```python
def mlp_model(hidden):
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(hidden, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')])
def get_model():
    inputs = keras.Input(shape=(784,))
    dense = layers.Dense(64, activation="relu")
    x = dense(inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model
def get_flat_model():
    inputs = keras.Input(shape=(28, 28))
    flat = keras.layers.Flatten(input_shape=(28, 28))(inputs)
    x = keras.layers.Dense(64, activation="relu")(flat)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model
```

### Example 1: Training

Create, Compile, Train, and Evaluate a model


```python
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model = get_flat_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#model.fit(x=x_train, y=y_train, epochs=1, callbacks=[tensorboard_callback], validation_split=0.2)
```

    WARNING:tensorflow:From C:\Users\gilbr\ML_Proj_Code\Proj_1_venv\Lib\site-packages\keras\src\backend.py:1398: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.
    
    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_1 (InputLayer)        [(None, 28, 28)]          0         
                                                                     
     flatten (Flatten)           (None, 784)               0         
                                                                     
     dense (Dense)               (None, 64)                50240     
                                                                     
     dense_1 (Dense)             (None, 64)                4160      
                                                                     
     dense_2 (Dense)             (None, 10)                650       
                                                                     
    =================================================================
    Total params: 55050 (215.04 KB)
    Trainable params: 55050 (215.04 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    WARNING:tensorflow:From C:\Users\gilbr\ML_Proj_Code\Proj_1_venv\Lib\site-packages\keras\src\optimizers\__init__.py:309: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    

### Example 2: Tuning

Create, Compile, Train, and Evaluate a model for each hyperparameter configuration


```python
hidden_units=[256]#[64,128,256]

for x in hidden_units:
    model = mlp_model(x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #model.fit(x=x_train, y=y_train, epochs=1, callbacks=[tf.keras.callbacks.TensorBoard(log_dir=f"logs/fit{x}/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)], validation_split=0.2)
    model.summary()

    #probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    #probability_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #probability_model.evaluate(x_test, y_test, verbose=2)
    #probability_model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_1 (Flatten)         (None, 784)               0         
                                                                     
     dense_3 (Dense)             (None, 256)               200960    
                                                                     
     dropout (Dropout)           (None, 256)               0         
                                                                     
     dense_4 (Dense)             (None, 10)                2570      
                                                                     
    =================================================================
    Total params: 203530 (795.04 KB)
    Trainable params: 203530 (795.04 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    

### Example 3: Expand tuning and tuning visualizations

Create, Compile, Train, and Evaluate a model for each 'hp hyperparameter' Config


```python
# Creating Hparams
#HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([1, 2]))
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([4, 16, 64]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.0, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['sgd','adam'])) 

# Creating train test function
def train_test_model(hparams, run_dir):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    model.compile(optimizer=hparams[HP_OPTIMIZER],loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    callbacks = [tf.keras.callbacks.TensorBoard(run_dir), hp.KerasCallback(run_dir, hparams), ]# log metrics, log hparams
    model.fit(x_train, y_train, epochs=10, callbacks = callbacks, validation_split=0.2) # Run with 1 epoch to speed things up for demo purposes
    _, accuracy = model.evaluate(x_test, y_test)
    model.summary()
    return accuracy 

session_num = 0
for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for optimizer in HP_OPTIMIZER.domain.values:
            hparams = {HP_NUM_UNITS: num_units, HP_DROPOUT: dropout_rate, HP_OPTIMIZER: optimizer,}
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            train_test_model(hparams, 'logs/hparam_tuning/' + run_name)
            session_num += 1
```

    --- Starting trial: run-0
    {'num_units': 4, 'dropout': 0.0, 'optimizer': 'adam'}
    Epoch 1/10
    WARNING:tensorflow:From C:\Users\gilbr\ML_Proj_Code\Proj_1_venv\Lib\site-packages\keras\src\utils\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.
    
    WARNING:tensorflow:From C:\Users\gilbr\ML_Proj_Code\Proj_1_venv\Lib\site-packages\keras\src\engine\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.
    
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.9651 - accuracy: 0.6834 - val_loss: 0.6285 - val_accuracy: 0.8071
    Epoch 2/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.6088 - accuracy: 0.8185 - val_loss: 0.5372 - val_accuracy: 0.8397
    Epoch 3/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.5474 - accuracy: 0.8383 - val_loss: 0.5017 - val_accuracy: 0.8498
    Epoch 4/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.5158 - accuracy: 0.8504 - val_loss: 0.4855 - val_accuracy: 0.8553
    Epoch 5/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.4964 - accuracy: 0.8566 - val_loss: 0.4644 - val_accuracy: 0.8639
    Epoch 6/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.4835 - accuracy: 0.8610 - val_loss: 0.4618 - val_accuracy: 0.8647
    Epoch 7/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.4733 - accuracy: 0.8643 - val_loss: 0.4501 - val_accuracy: 0.8685
    Epoch 8/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.4658 - accuracy: 0.8666 - val_loss: 0.4431 - val_accuracy: 0.8704
    Epoch 9/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.4604 - accuracy: 0.8687 - val_loss: 0.4472 - val_accuracy: 0.8696
    Epoch 10/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.4559 - accuracy: 0.8699 - val_loss: 0.4364 - val_accuracy: 0.8732
    313/313 [==============================] - 1s 2ms/step - loss: 0.4629 - accuracy: 0.8692
    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_2 (Flatten)         (None, 784)               0         
                                                                     
     dense_5 (Dense)             (None, 4)                 3140      
                                                                     
     dropout_1 (Dropout)         (None, 4)                 0         
                                                                     
     dense_6 (Dense)             (None, 10)                50        
                                                                     
    =================================================================
    Total params: 3190 (12.46 KB)
    Trainable params: 3190 (12.46 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    --- Starting trial: run-1
    {'num_units': 4, 'dropout': 0.0, 'optimizer': 'sgd'}
    Epoch 1/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 1.3285 - accuracy: 0.5694 - val_loss: 0.8708 - val_accuracy: 0.7264
    Epoch 2/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.7912 - accuracy: 0.7465 - val_loss: 0.6807 - val_accuracy: 0.7778
    Epoch 3/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.6612 - accuracy: 0.7960 - val_loss: 0.5873 - val_accuracy: 0.8156
    Epoch 4/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.5929 - accuracy: 0.8259 - val_loss: 0.5459 - val_accuracy: 0.8341
    Epoch 5/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.5618 - accuracy: 0.8378 - val_loss: 0.5220 - val_accuracy: 0.8425
    Epoch 6/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.5438 - accuracy: 0.8439 - val_loss: 0.5066 - val_accuracy: 0.8501
    Epoch 7/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.5320 - accuracy: 0.8483 - val_loss: 0.4995 - val_accuracy: 0.8536
    Epoch 8/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.5227 - accuracy: 0.8515 - val_loss: 0.4903 - val_accuracy: 0.8544
    Epoch 9/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.5149 - accuracy: 0.8533 - val_loss: 0.4825 - val_accuracy: 0.8591
    Epoch 10/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.5090 - accuracy: 0.8555 - val_loss: 0.4797 - val_accuracy: 0.8591
    313/313 [==============================] - 1s 2ms/step - loss: 0.4935 - accuracy: 0.8594
    Model: "sequential_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_3 (Flatten)         (None, 784)               0         
                                                                     
     dense_7 (Dense)             (None, 4)                 3140      
                                                                     
     dropout_2 (Dropout)         (None, 4)                 0         
                                                                     
     dense_8 (Dense)             (None, 10)                50        
                                                                     
    =================================================================
    Total params: 3190 (12.46 KB)
    Trainable params: 3190 (12.46 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    --- Starting trial: run-2
    {'num_units': 4, 'dropout': 0.2, 'optimizer': 'adam'}
    Epoch 1/10
    1500/1500 [==============================] - 5s 3ms/step - loss: 1.4023 - accuracy: 0.4905 - val_loss: 0.8130 - val_accuracy: 0.8076
    Epoch 2/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 1.1602 - accuracy: 0.5840 - val_loss: 0.7322 - val_accuracy: 0.8254
    Epoch 3/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 1.1286 - accuracy: 0.6042 - val_loss: 0.6919 - val_accuracy: 0.8357
    Epoch 4/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 1.1107 - accuracy: 0.6113 - val_loss: 0.6984 - val_accuracy: 0.8332
    Epoch 5/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 1.1008 - accuracy: 0.6116 - val_loss: 0.6712 - val_accuracy: 0.8327
    Epoch 6/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 1.0871 - accuracy: 0.6174 - val_loss: 0.6740 - val_accuracy: 0.8348
    Epoch 7/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 1.0880 - accuracy: 0.6131 - val_loss: 0.6618 - val_accuracy: 0.8399
    Epoch 8/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 1.0856 - accuracy: 0.6151 - val_loss: 0.6504 - val_accuracy: 0.8390
    Epoch 9/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 1.0816 - accuracy: 0.6139 - val_loss: 0.6524 - val_accuracy: 0.8380
    Epoch 10/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 1.0748 - accuracy: 0.6184 - val_loss: 0.6497 - val_accuracy: 0.8421
    313/313 [==============================] - 1s 2ms/step - loss: 0.6677 - accuracy: 0.8338
    Model: "sequential_3"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_4 (Flatten)         (None, 784)               0         
                                                                     
     dense_9 (Dense)             (None, 4)                 3140      
                                                                     
     dropout_3 (Dropout)         (None, 4)                 0         
                                                                     
     dense_10 (Dense)            (None, 10)                50        
                                                                     
    =================================================================
    Total params: 3190 (12.46 KB)
    Trainable params: 3190 (12.46 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    --- Starting trial: run-3
    {'num_units': 4, 'dropout': 0.2, 'optimizer': 'sgd'}
    Epoch 1/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 1.8729 - accuracy: 0.3250 - val_loss: 1.4352 - val_accuracy: 0.6049
    Epoch 2/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 1.4735 - accuracy: 0.5019 - val_loss: 1.1321 - val_accuracy: 0.6959
    Epoch 3/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 1.3341 - accuracy: 0.5363 - val_loss: 1.0024 - val_accuracy: 0.7402
    Epoch 4/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 1.2670 - accuracy: 0.5651 - val_loss: 0.9171 - val_accuracy: 0.7704
    Epoch 5/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 1.2263 - accuracy: 0.5798 - val_loss: 0.8738 - val_accuracy: 0.7824
    Epoch 6/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 1.2041 - accuracy: 0.5864 - val_loss: 0.8371 - val_accuracy: 0.7901
    Epoch 7/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 1.1807 - accuracy: 0.5906 - val_loss: 0.8006 - val_accuracy: 0.7929
    Epoch 8/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 1.1688 - accuracy: 0.5916 - val_loss: 0.7765 - val_accuracy: 0.7969
    Epoch 9/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 1.1540 - accuracy: 0.5939 - val_loss: 0.7682 - val_accuracy: 0.8002
    Epoch 10/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 1.1490 - accuracy: 0.5953 - val_loss: 0.7604 - val_accuracy: 0.8024
    313/313 [==============================] - 1s 2ms/step - loss: 0.7580 - accuracy: 0.7970
    Model: "sequential_4"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_5 (Flatten)         (None, 784)               0         
                                                                     
     dense_11 (Dense)            (None, 4)                 3140      
                                                                     
     dropout_4 (Dropout)         (None, 4)                 0         
                                                                     
     dense_12 (Dense)            (None, 10)                50        
                                                                     
    =================================================================
    Total params: 3190 (12.46 KB)
    Trainable params: 3190 (12.46 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    --- Starting trial: run-4
    {'num_units': 16, 'dropout': 0.0, 'optimizer': 'adam'}
    Epoch 1/10
    1500/1500 [==============================] - 5s 3ms/step - loss: 0.4758 - accuracy: 0.8627 - val_loss: 0.2735 - val_accuracy: 0.9218
    Epoch 2/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.2676 - accuracy: 0.9246 - val_loss: 0.2423 - val_accuracy: 0.9317
    Epoch 3/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.2332 - accuracy: 0.9341 - val_loss: 0.2234 - val_accuracy: 0.9368
    Epoch 4/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.2118 - accuracy: 0.9399 - val_loss: 0.2088 - val_accuracy: 0.9410
    Epoch 5/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.1957 - accuracy: 0.9439 - val_loss: 0.1941 - val_accuracy: 0.9464
    Epoch 6/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.1822 - accuracy: 0.9477 - val_loss: 0.1977 - val_accuracy: 0.9448
    Epoch 7/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.1734 - accuracy: 0.9500 - val_loss: 0.1885 - val_accuracy: 0.9470
    Epoch 8/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.1641 - accuracy: 0.9522 - val_loss: 0.1799 - val_accuracy: 0.9490
    Epoch 9/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.1580 - accuracy: 0.9548 - val_loss: 0.1796 - val_accuracy: 0.9482
    Epoch 10/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.1502 - accuracy: 0.9570 - val_loss: 0.1738 - val_accuracy: 0.9506
    313/313 [==============================] - 1s 2ms/step - loss: 0.1718 - accuracy: 0.9503
    Model: "sequential_5"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_6 (Flatten)         (None, 784)               0         
                                                                     
     dense_13 (Dense)            (None, 16)                12560     
                                                                     
     dropout_5 (Dropout)         (None, 16)                0         
                                                                     
     dense_14 (Dense)            (None, 10)                170       
                                                                     
    =================================================================
    Total params: 12730 (49.73 KB)
    Trainable params: 12730 (49.73 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    --- Starting trial: run-5
    {'num_units': 16, 'dropout': 0.0, 'optimizer': 'sgd'}
    Epoch 1/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.8434 - accuracy: 0.7698 - val_loss: 0.4257 - val_accuracy: 0.8869
    Epoch 2/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.3978 - accuracy: 0.8915 - val_loss: 0.3334 - val_accuracy: 0.9036
    Epoch 3/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.3369 - accuracy: 0.9054 - val_loss: 0.2984 - val_accuracy: 0.9122
    Epoch 4/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.3084 - accuracy: 0.9130 - val_loss: 0.2792 - val_accuracy: 0.9182
    Epoch 5/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.2897 - accuracy: 0.9175 - val_loss: 0.2670 - val_accuracy: 0.9217
    Epoch 6/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.2763 - accuracy: 0.9219 - val_loss: 0.2561 - val_accuracy: 0.9254
    Epoch 7/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.2646 - accuracy: 0.9251 - val_loss: 0.2485 - val_accuracy: 0.9287
    Epoch 8/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.2550 - accuracy: 0.9278 - val_loss: 0.2407 - val_accuracy: 0.9321
    Epoch 9/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.2466 - accuracy: 0.9301 - val_loss: 0.2358 - val_accuracy: 0.9325
    Epoch 10/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.2390 - accuracy: 0.9319 - val_loss: 0.2314 - val_accuracy: 0.9342
    313/313 [==============================] - 1s 2ms/step - loss: 0.2334 - accuracy: 0.9322
    Model: "sequential_6"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_7 (Flatten)         (None, 784)               0         
                                                                     
     dense_15 (Dense)            (None, 16)                12560     
                                                                     
     dropout_6 (Dropout)         (None, 16)                0         
                                                                     
     dense_16 (Dense)            (None, 10)                170       
                                                                     
    =================================================================
    Total params: 12730 (49.73 KB)
    Trainable params: 12730 (49.73 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    --- Starting trial: run-6
    {'num_units': 16, 'dropout': 0.2, 'optimizer': 'adam'}
    Epoch 1/10
    1500/1500 [==============================] - 5s 3ms/step - loss: 0.7115 - accuracy: 0.7778 - val_loss: 0.3204 - val_accuracy: 0.9123
    Epoch 2/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.4773 - accuracy: 0.8495 - val_loss: 0.2736 - val_accuracy: 0.9223
    Epoch 3/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.4420 - accuracy: 0.8613 - val_loss: 0.2608 - val_accuracy: 0.9283
    Epoch 4/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.4256 - accuracy: 0.8663 - val_loss: 0.2541 - val_accuracy: 0.9284
    Epoch 5/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.4082 - accuracy: 0.8704 - val_loss: 0.2441 - val_accuracy: 0.9308
    Epoch 6/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.3968 - accuracy: 0.8749 - val_loss: 0.2414 - val_accuracy: 0.9302
    Epoch 7/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.3880 - accuracy: 0.8759 - val_loss: 0.2406 - val_accuracy: 0.9320
    Epoch 8/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3813 - accuracy: 0.8785 - val_loss: 0.2322 - val_accuracy: 0.9350
    Epoch 9/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3770 - accuracy: 0.8803 - val_loss: 0.2328 - val_accuracy: 0.9350
    Epoch 10/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3686 - accuracy: 0.8817 - val_loss: 0.2274 - val_accuracy: 0.9352
    313/313 [==============================] - 1s 2ms/step - loss: 0.2404 - accuracy: 0.9311
    Model: "sequential_7"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_8 (Flatten)         (None, 784)               0         
                                                                     
     dense_17 (Dense)            (None, 16)                12560     
                                                                     
     dropout_7 (Dropout)         (None, 16)                0         
                                                                     
     dense_18 (Dense)            (None, 10)                170       
                                                                     
    =================================================================
    Total params: 12730 (49.73 KB)
    Trainable params: 12730 (49.73 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    --- Starting trial: run-7
    {'num_units': 16, 'dropout': 0.2, 'optimizer': 'sgd'}
    Epoch 1/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 1.1360 - accuracy: 0.6440 - val_loss: 0.5317 - val_accuracy: 0.8731
    Epoch 2/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.6678 - accuracy: 0.7934 - val_loss: 0.4078 - val_accuracy: 0.8948
    Epoch 3/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.5781 - accuracy: 0.8223 - val_loss: 0.3605 - val_accuracy: 0.9052
    Epoch 4/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.5375 - accuracy: 0.8344 - val_loss: 0.3326 - val_accuracy: 0.9095
    Epoch 5/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.5069 - accuracy: 0.8430 - val_loss: 0.3151 - val_accuracy: 0.9136
    Epoch 6/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.4861 - accuracy: 0.8488 - val_loss: 0.3034 - val_accuracy: 0.9172
    Epoch 7/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.4680 - accuracy: 0.8542 - val_loss: 0.2925 - val_accuracy: 0.9196
    Epoch 8/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.4617 - accuracy: 0.8570 - val_loss: 0.2835 - val_accuracy: 0.9208
    Epoch 9/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.4472 - accuracy: 0.8618 - val_loss: 0.2779 - val_accuracy: 0.9227
    Epoch 10/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.4417 - accuracy: 0.8631 - val_loss: 0.2717 - val_accuracy: 0.9228
    313/313 [==============================] - 1s 2ms/step - loss: 0.2697 - accuracy: 0.9236
    Model: "sequential_8"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_9 (Flatten)         (None, 784)               0         
                                                                     
     dense_19 (Dense)            (None, 16)                12560     
                                                                     
     dropout_8 (Dropout)         (None, 16)                0         
                                                                     
     dense_20 (Dense)            (None, 10)                170       
                                                                     
    =================================================================
    Total params: 12730 (49.73 KB)
    Trainable params: 12730 (49.73 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    --- Starting trial: run-8
    {'num_units': 64, 'dropout': 0.0, 'optimizer': 'adam'}
    Epoch 1/10
    1500/1500 [==============================] - 6s 3ms/step - loss: 0.3372 - accuracy: 0.9056 - val_loss: 0.1854 - val_accuracy: 0.9463
    Epoch 2/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.1623 - accuracy: 0.9521 - val_loss: 0.1327 - val_accuracy: 0.9622
    Epoch 3/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.1174 - accuracy: 0.9661 - val_loss: 0.1205 - val_accuracy: 0.9651
    Epoch 4/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.0916 - accuracy: 0.9726 - val_loss: 0.1081 - val_accuracy: 0.9687
    Epoch 5/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.0750 - accuracy: 0.9777 - val_loss: 0.1120 - val_accuracy: 0.9660
    Epoch 6/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.0629 - accuracy: 0.9810 - val_loss: 0.1034 - val_accuracy: 0.9699
    Epoch 7/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.0531 - accuracy: 0.9836 - val_loss: 0.0976 - val_accuracy: 0.9714
    Epoch 8/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.0444 - accuracy: 0.9865 - val_loss: 0.1047 - val_accuracy: 0.9705
    Epoch 9/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.0386 - accuracy: 0.9882 - val_loss: 0.1054 - val_accuracy: 0.9697
    Epoch 10/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.0330 - accuracy: 0.9895 - val_loss: 0.1029 - val_accuracy: 0.9719
    313/313 [==============================] - 1s 2ms/step - loss: 0.0926 - accuracy: 0.9734
    Model: "sequential_9"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_10 (Flatten)        (None, 784)               0         
                                                                     
     dense_21 (Dense)            (None, 64)                50240     
                                                                     
     dropout_9 (Dropout)         (None, 64)                0         
                                                                     
     dense_22 (Dense)            (None, 10)                650       
                                                                     
    =================================================================
    Total params: 50890 (198.79 KB)
    Trainable params: 50890 (198.79 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    --- Starting trial: run-9
    {'num_units': 64, 'dropout': 0.0, 'optimizer': 'sgd'}
    Epoch 1/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.7432 - accuracy: 0.8107 - val_loss: 0.3920 - val_accuracy: 0.8955
    Epoch 2/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.3755 - accuracy: 0.8953 - val_loss: 0.3206 - val_accuracy: 0.9116
    Epoch 3/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3234 - accuracy: 0.9083 - val_loss: 0.2890 - val_accuracy: 0.9207
    Epoch 4/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.2937 - accuracy: 0.9168 - val_loss: 0.2679 - val_accuracy: 0.9261
    Epoch 5/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.2710 - accuracy: 0.9230 - val_loss: 0.2519 - val_accuracy: 0.9315
    Epoch 6/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.2525 - accuracy: 0.9282 - val_loss: 0.2378 - val_accuracy: 0.9335
    Epoch 7/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.2368 - accuracy: 0.9333 - val_loss: 0.2245 - val_accuracy: 0.9382
    Epoch 8/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.2234 - accuracy: 0.9370 - val_loss: 0.2159 - val_accuracy: 0.9403
    Epoch 9/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.2115 - accuracy: 0.9402 - val_loss: 0.2048 - val_accuracy: 0.9440
    Epoch 10/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.2009 - accuracy: 0.9429 - val_loss: 0.1966 - val_accuracy: 0.9459
    313/313 [==============================] - 1s 2ms/step - loss: 0.1955 - accuracy: 0.9434
    Model: "sequential_10"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_11 (Flatten)        (None, 784)               0         
                                                                     
     dense_23 (Dense)            (None, 64)                50240     
                                                                     
     dropout_10 (Dropout)        (None, 64)                0         
                                                                     
     dense_24 (Dense)            (None, 10)                650       
                                                                     
    =================================================================
    Total params: 50890 (198.79 KB)
    Trainable params: 50890 (198.79 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    --- Starting trial: run-10
    {'num_units': 64, 'dropout': 0.2, 'optimizer': 'adam'}
    Epoch 1/10
    1500/1500 [==============================] - 5s 3ms/step - loss: 0.3951 - accuracy: 0.8842 - val_loss: 0.1974 - val_accuracy: 0.9440
    Epoch 2/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.2118 - accuracy: 0.9374 - val_loss: 0.1469 - val_accuracy: 0.9568
    Epoch 3/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.1701 - accuracy: 0.9486 - val_loss: 0.1332 - val_accuracy: 0.9603
    Epoch 4/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.1449 - accuracy: 0.9563 - val_loss: 0.1122 - val_accuracy: 0.9672
    Epoch 5/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.1271 - accuracy: 0.9607 - val_loss: 0.1076 - val_accuracy: 0.9682
    Epoch 6/10
    1500/1500 [==============================] - 5s 3ms/step - loss: 0.1145 - accuracy: 0.9639 - val_loss: 0.1026 - val_accuracy: 0.9706
    Epoch 7/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.1037 - accuracy: 0.9674 - val_loss: 0.0987 - val_accuracy: 0.9727
    Epoch 8/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.0980 - accuracy: 0.9695 - val_loss: 0.0943 - val_accuracy: 0.9722
    Epoch 9/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.0936 - accuracy: 0.9706 - val_loss: 0.0983 - val_accuracy: 0.9718
    Epoch 10/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.0861 - accuracy: 0.9723 - val_loss: 0.0945 - val_accuracy: 0.9732
    313/313 [==============================] - 1s 2ms/step - loss: 0.0911 - accuracy: 0.9730
    Model: "sequential_11"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_12 (Flatten)        (None, 784)               0         
                                                                     
     dense_25 (Dense)            (None, 64)                50240     
                                                                     
     dropout_11 (Dropout)        (None, 64)                0         
                                                                     
     dense_26 (Dense)            (None, 10)                650       
                                                                     
    =================================================================
    Total params: 50890 (198.79 KB)
    Trainable params: 50890 (198.79 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    --- Starting trial: run-11
    {'num_units': 64, 'dropout': 0.2, 'optimizer': 'sgd'}
    Epoch 1/10
    1500/1500 [==============================] - 4s 2ms/step - loss: 0.8428 - accuracy: 0.7649 - val_loss: 0.3964 - val_accuracy: 0.8982
    Epoch 2/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.4460 - accuracy: 0.8739 - val_loss: 0.3164 - val_accuracy: 0.9130
    Epoch 3/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3804 - accuracy: 0.8908 - val_loss: 0.2797 - val_accuracy: 0.9232
    Epoch 4/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3380 - accuracy: 0.9033 - val_loss: 0.2554 - val_accuracy: 0.9289
    Epoch 5/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3140 - accuracy: 0.9104 - val_loss: 0.2375 - val_accuracy: 0.9337
    Epoch 6/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.2942 - accuracy: 0.9164 - val_loss: 0.2245 - val_accuracy: 0.9360
    Epoch 7/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.2769 - accuracy: 0.9204 - val_loss: 0.2123 - val_accuracy: 0.9417
    Epoch 8/10
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.2624 - accuracy: 0.9246 - val_loss: 0.2022 - val_accuracy: 0.9442
    Epoch 9/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.2510 - accuracy: 0.9289 - val_loss: 0.1935 - val_accuracy: 0.9458
    Epoch 10/10
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.2394 - accuracy: 0.9317 - val_loss: 0.1873 - val_accuracy: 0.9474
    313/313 [==============================] - 1s 2ms/step - loss: 0.1865 - accuracy: 0.9459
    Model: "sequential_12"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_13 (Flatten)        (None, 784)               0         
                                                                     
     dense_27 (Dense)            (None, 64)                50240     
                                                                     
     dropout_12 (Dropout)        (None, 64)                0         
                                                                     
     dense_28 (Dense)            (None, 10)                650       
                                                                     
    =================================================================
    Total params: 50890 (198.79 KB)
    Trainable params: 50890 (198.79 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    

### Monitoring with Tensorboard:


```python
%tensorboard --logdir logs --port 6009
```


    Reusing TensorBoard on port 6009 (pid 21552), started 7:23:25 ago. (Use '!kill 21552' to kill it.)




<iframe id="tensorboard-frame-24a9924930e130ec" width="100%" height="800" frameborder="0">
</iframe>
<script>
  (function() {
    const frame = document.getElementById("tensorboard-frame-24a9924930e130ec");
    const url = new URL("/", window.location);
    const port = 6009;
    if (port) {
      url.port = port;
    }
    frame.src = url;
  })();
</script>



### Initial Results



## A/B testing for detecting Data Drift

### Evaluate Strata

Examples of Stratas:
- Data sources
- label source
- Sex
- race
- age

Task:
- split val set into subsets based on strata meta info or use a dataset from a diferent strata from the original train and val set. 
- a/b testing via evaluations with the new datasets or subsets

## Model Serving and Monitoring


```python

```

# Development Steps:

## 1. Hard code a bare bones proof-of-concept (POC) for an ML Project, within the identified project scope, in a single jupyter notebook file. 

## 2. Define functions for repeatable lines of code, define variables for controling previously hardcoded settings,improve naming conventions

## 3. Add variables to control previously hard coded settings

## 4. move functions into seperate files to shorten and siplify the main file. 

## 5. Add or update Documentation

## 6. Add more Features to the project That are relavent to the identified project scope starting from step one if more improvements are needed

# Project Management Steps:

## Setup a Project Dashboard for the github repo.

## add meaningful views to the dashboard to help orginize the project.

## setup auto email alerts for team members assigned to issues when updates are posted.

## Add issues and set the priority, size, sprint, labels, tags, and more if needed.

## Set and enforce standards for team members posting to the project dashboard.

## Set Rules for the repo to protect the main branch from updates without reviews from more than one person

## Manually assign project issues for team members to work on or allow self assigning issues for a more hands free approach to managing the team.

## Regularly sceduled meetings with your team and with other managers from other departments.

- Management Meetings
    - discus customer concerns and requests
    - discus internal questions and concerns related to budgetting, time lines, security, legal, ect.
    - re-evaluate project issue priorities and sceduled sprints as a group.
- Team Meating
    - Summary of major updates
    - Point out examples of good work from the team 
    - Point out common mistakes that more than one team member is made without speciffing who made the mistake
- Member meetings
    - Discus progress and personal growth

## Allways encourage team members to ask questions for clarity on the goal of a tasks and to ask for more directions if they get stuck.

- Try to respond to the questions in a way that helps them build the intuition to find the answers to similar questions themselves next time.
- Try to identify the way they were confused by the task and help them understand how to frame the problem better in thier mind by asking the right question.
- Time spent fostering growth among jonior team members instead of just solving the problem for them promotes 

# Tutorial 2 ML Steps

## Download and load the cifar10 Dataset


```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
```

## Preprocess


```python
x_train, x_test = x_train.astype("float32") / 255.0, x_test.astype("float32") / 255.0
y_train, y_test = keras.utils.to_categorical(y_train, 10), keras.utils.to_categorical(y_test, 10)
```

## model def for mini resnet 


```python
def get_mini_resnet():
    inputs = keras.Input(shape=(32, 32, 3), name="img")
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    block_1_output = layers.MaxPooling2D(3)(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    block_2_output = layers.add([x, block_1_output])
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    block_3_output = layers.add([x, block_2_output])
    x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model
```

## compile and train a mini resnet model


```python
model = get_mini_resnet()
model.compile(optimizer=keras.optimizers.RMSprop(1e-3), loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=["acc"],)
model.fit(x_train, y_train, batch_size=64, epochs=1, validation_split=0.2)
```

    WARNING:tensorflow:From C:\Users\gilbr\ML_Proj_Code\Proj_1_venv\Lib\site-packages\keras\src\layers\pooling\max_pooling2d.py:161: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.
    
    Model: "model_1"
    __________________________________________________________________________________________________
     Layer (type)                Output Shape                 Param #   Connected to                  
    ==================================================================================================
     img (InputLayer)            [(None, 32, 32, 3)]          0         []                            
                                                                                                      
     conv2d (Conv2D)             (None, 30, 30, 32)           896       ['img[0][0]']                 
                                                                                                      
     conv2d_1 (Conv2D)           (None, 28, 28, 64)           18496     ['conv2d[0][0]']              
                                                                                                      
     max_pooling2d (MaxPooling2  (None, 9, 9, 64)             0         ['conv2d_1[0][0]']            
     D)                                                                                               
                                                                                                      
     conv2d_2 (Conv2D)           (None, 9, 9, 64)             36928     ['max_pooling2d[0][0]']       
                                                                                                      
     conv2d_3 (Conv2D)           (None, 9, 9, 64)             36928     ['conv2d_2[0][0]']            
                                                                                                      
     add (Add)                   (None, 9, 9, 64)             0         ['conv2d_3[0][0]',            
                                                                         'max_pooling2d[0][0]']       
                                                                                                      
     conv2d_4 (Conv2D)           (None, 9, 9, 64)             36928     ['add[0][0]']                 
                                                                                                      
     conv2d_5 (Conv2D)           (None, 9, 9, 64)             36928     ['conv2d_4[0][0]']            
                                                                                                      
     add_1 (Add)                 (None, 9, 9, 64)             0         ['conv2d_5[0][0]',            
                                                                         'add[0][0]']                 
                                                                                                      
     conv2d_6 (Conv2D)           (None, 7, 7, 64)             36928     ['add_1[0][0]']               
                                                                                                      
     global_average_pooling2d (  (None, 64)                   0         ['conv2d_6[0][0]']            
     GlobalAveragePooling2D)                                                                          
                                                                                                      
     dense_29 (Dense)            (None, 256)                  16640     ['global_average_pooling2d[0][
                                                                        0]']                          
                                                                                                      
     dropout_13 (Dropout)        (None, 256)                  0         ['dense_29[0][0]']            
                                                                                                      
     dense_30 (Dense)            (None, 10)                   2570      ['dropout_13[0][0]']          
                                                                                                      
    ==================================================================================================
    Total params: 223242 (872.04 KB)
    Trainable params: 223242 (872.04 KB)
    Non-trainable params: 0 (0.00 Byte)
    __________________________________________________________________________________________________
    625/625 [==============================] - 76s 119ms/step - loss: 1.9313 - acc: 0.2680 - val_loss: 1.7624 - val_acc: 0.3608
    




    <keras.src.callbacks.History at 0x1e4ae746590>



## Using the VGG19 Model and extract all layers (features exstraction)


```python
vgg19 = keras.applications.VGG19()
vgg19.summary()
features_list = [layer.output for layer in vgg19.layers]
feat_extraction_model = keras.Model(inputs=vgg19.input, outputs=features_list)
img = np.random.random((1, 224, 224, 3)).astype("float32")
extracted_features = feat_extraction_model(img)
```

    Model: "vgg19"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_2 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                     
     block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
                                                                     
     block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
                                                                     
     block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
                                                                     
     block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
                                                                     
     block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
                                                                     
     block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
                                                                     
     block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
                                                                     
     block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                     
     block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                     
     block3_conv4 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                     
     block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
                                                                     
     block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
                                                                     
     block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                     
     block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                     
     block4_conv4 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                     
     block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
                                                                     
     block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_conv4 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                     
     flatten (Flatten)           (None, 25088)             0         
                                                                     
     fc1 (Dense)                 (None, 4096)              102764544 
                                                                     
     fc2 (Dense)                 (None, 4096)              16781312  
                                                                     
     predictions (Dense)         (None, 1000)              4097000   
                                                                     
    =================================================================
    Total params: 143667240 (548.05 MB)
    Trainable params: 143667240 (548.05 MB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    

## Autoencoder: Encoder + Decoder


```python
encoder_input = keras.Input(shape=(28, 28, 1), name="img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)
encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

x = layers.Reshape((4, 4, 1))(encoder_output)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)
autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
autoencoder.summary()
```

    Model: "encoder"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     img (InputLayer)            [(None, 28, 28, 1)]       0         
                                                                     
     conv2d_7 (Conv2D)           (None, 26, 26, 16)        160       
                                                                     
     conv2d_8 (Conv2D)           (None, 24, 24, 32)        4640      
                                                                     
     max_pooling2d_1 (MaxPoolin  (None, 8, 8, 32)          0         
     g2D)                                                            
                                                                     
     conv2d_9 (Conv2D)           (None, 6, 6, 32)          9248      
                                                                     
     conv2d_10 (Conv2D)          (None, 4, 4, 16)          4624      
                                                                     
     global_max_pooling2d (Glob  (None, 16)                0         
     alMaxPooling2D)                                                 
                                                                     
    =================================================================
    Total params: 18672 (72.94 KB)
    Trainable params: 18672 (72.94 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    Model: "autoencoder"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     img (InputLayer)            [(None, 28, 28, 1)]       0         
                                                                     
     conv2d_7 (Conv2D)           (None, 26, 26, 16)        160       
                                                                     
     conv2d_8 (Conv2D)           (None, 24, 24, 32)        4640      
                                                                     
     max_pooling2d_1 (MaxPoolin  (None, 8, 8, 32)          0         
     g2D)                                                            
                                                                     
     conv2d_9 (Conv2D)           (None, 6, 6, 32)          9248      
                                                                     
     conv2d_10 (Conv2D)          (None, 4, 4, 16)          4624      
                                                                     
     global_max_pooling2d (Glob  (None, 16)                0         
     alMaxPooling2D)                                                 
                                                                     
     reshape (Reshape)           (None, 4, 4, 1)           0         
                                                                     
     conv2d_transpose (Conv2DTr  (None, 6, 6, 16)          160       
     anspose)                                                        
                                                                     
     conv2d_transpose_1 (Conv2D  (None, 8, 8, 32)          4640      
     Transpose)                                                      
                                                                     
     up_sampling2d (UpSampling2  (None, 24, 24, 32)        0         
     D)                                                              
                                                                     
     conv2d_transpose_2 (Conv2D  (None, 26, 26, 16)        4624      
     Transpose)                                                      
                                                                     
     conv2d_transpose_3 (Conv2D  (None, 28, 28, 1)         145       
     Transpose)                                                      
                                                                     
    =================================================================
    Total params: 28241 (110.32 KB)
    Trainable params: 28241 (110.32 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    


```python
encoder_input = keras.Input(shape=(28, 28, 1), name="original_img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)
encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

decoder_input = keras.Input(shape=(16,), name="encoded_img")
x = layers.Reshape((4, 4, 1))(decoder_input)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)
decoder = keras.Model(decoder_input, decoder_output, name="decoder")
decoder.summary()

autoencoder_input = keras.Input(shape=(28, 28, 1), name="img")
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = keras.Model(autoencoder_input, decoded_img, name="autoencoder")
autoencoder.summary()
```

    Model: "encoder"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     original_img (InputLayer)   [(None, 28, 28, 1)]       0         
                                                                     
     conv2d_11 (Conv2D)          (None, 26, 26, 16)        160       
                                                                     
     conv2d_12 (Conv2D)          (None, 24, 24, 32)        4640      
                                                                     
     max_pooling2d_2 (MaxPoolin  (None, 8, 8, 32)          0         
     g2D)                                                            
                                                                     
     conv2d_13 (Conv2D)          (None, 6, 6, 32)          9248      
                                                                     
     conv2d_14 (Conv2D)          (None, 4, 4, 16)          4624      
                                                                     
     global_max_pooling2d_1 (Gl  (None, 16)                0         
     obalMaxPooling2D)                                               
                                                                     
    =================================================================
    Total params: 18672 (72.94 KB)
    Trainable params: 18672 (72.94 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    Model: "decoder"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     encoded_img (InputLayer)    [(None, 16)]              0         
                                                                     
     reshape_1 (Reshape)         (None, 4, 4, 1)           0         
                                                                     
     conv2d_transpose_4 (Conv2D  (None, 6, 6, 16)          160       
     Transpose)                                                      
                                                                     
     conv2d_transpose_5 (Conv2D  (None, 8, 8, 32)          4640      
     Transpose)                                                      
                                                                     
     up_sampling2d_1 (UpSamplin  (None, 24, 24, 32)        0         
     g2D)                                                            
                                                                     
     conv2d_transpose_6 (Conv2D  (None, 26, 26, 16)        4624      
     Transpose)                                                      
                                                                     
     conv2d_transpose_7 (Conv2D  (None, 28, 28, 1)         145       
     Transpose)                                                      
                                                                     
    =================================================================
    Total params: 9569 (37.38 KB)
    Trainable params: 9569 (37.38 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    Model: "autoencoder"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     img (InputLayer)            [(None, 28, 28, 1)]       0         
                                                                     
     encoder (Functional)        (None, 16)                18672     
                                                                     
     decoder (Functional)        (None, 28, 28, 1)         9569      
                                                                     
    =================================================================
    Total params: 28241 (110.32 KB)
    Trainable params: 28241 (110.32 KB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    

## Multi Input and Multi Output


```python
num_tags = 12  # Number of unique issue tags
num_words = 10000  # Size of vocabulary obtained when preprocessing text data
num_departments = 4  # Number of departments for predictions

title_input = keras.Input(shape=(None,), name="title")  # Variable-length sequence of ints
body_input = keras.Input(shape=(None,), name="body")  # Variable-length sequence of ints
tags_input = keras.Input(shape=(num_tags,), name="tags")  # Binary vectors of size `num_tags`

# Embed each word in the title into a 64-dimensional vector
title_features = layers.Embedding(num_words, 64)(title_input)
# Embed each word in the text into a 64-dimensional vector
body_features = layers.Embedding(num_words, 64)(body_input)

# Reduce sequence of embedded words in the title into a single 128-dimensional vector
title_features = layers.LSTM(128)(title_features)
# Reduce sequence of embedded words in the body into a single 32-dimensional vector
body_features = layers.LSTM(32)(body_features)

# Merge all available features into a single large vector via concatenation
x = layers.concatenate([title_features, body_features, tags_input])

# Stick a logistic regression for priority prediction on top of the features
priority_pred = layers.Dense(1, name="priority")(x)
# Stick a department classifier on top of the features
department_pred = layers.Dense(num_departments, name="department")(x)

# Instantiate an end-to-end model predicting both priority and department
model = keras.Model(
    inputs=[title_input, body_input, tags_input],
    outputs=[priority_pred, department_pred],)

if True:
    model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
        loss=[keras.losses.BinaryCrossentropy(from_logits=True),keras.losses.CategoricalCrossentropy(from_logits=True),],
        loss_weights=[1.0, 0.2],)
else:
    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss={"priority": keras.losses.BinaryCrossentropy(from_logits=True),"department": keras.losses.CategoricalCrossentropy(from_logits=True),},
        loss_weights={"priority": 1.0, "department": 0.2},)
# Dummy input data
title_data = np.random.randint(num_words, size=(1280, 10))
body_data = np.random.randint(num_words, size=(1280, 100))
tags_data = np.random.randint(2, size=(1280, num_tags)).astype("float32")
# Dummy target data
priority_targets = np.random.random(size=(1280, 1))
dept_targets = np.random.randint(2, size=(1280, num_departments))

#model.fit({"title": title_data, "body": body_data, "tags": tags_data}, {"priority": priority_targets, "department": dept_targets}, epochs=2, batch_size=32,)
model.summary()
```

    Model: "model_3"
    __________________________________________________________________________________________________
     Layer (type)                Output Shape                 Param #   Connected to                  
    ==================================================================================================
     title (InputLayer)          [(None, None)]               0         []                            
                                                                                                      
     body (InputLayer)           [(None, None)]               0         []                            
                                                                                                      
     embedding (Embedding)       (None, None, 64)             640000    ['title[0][0]']               
                                                                                                      
     embedding_1 (Embedding)     (None, None, 64)             640000    ['body[0][0]']                
                                                                                                      
     lstm (LSTM)                 (None, 128)                  98816     ['embedding[0][0]']           
                                                                                                      
     lstm_1 (LSTM)               (None, 32)                   12416     ['embedding_1[0][0]']         
                                                                                                      
     tags (InputLayer)           [(None, 12)]                 0         []                            
                                                                                                      
     concatenate (Concatenate)   (None, 172)                  0         ['lstm[0][0]',                
                                                                         'lstm_1[0][0]',              
                                                                         'tags[0][0]']                
                                                                                                      
     priority (Dense)            (None, 1)                    173       ['concatenate[0][0]']         
                                                                                                      
     department (Dense)          (None, 4)                    692       ['concatenate[0][0]']         
                                                                                                      
    ==================================================================================================
    Total params: 1392097 (5.31 MB)
    Trainable params: 1392097 (5.31 MB)
    Non-trainable params: 0 (0.00 Byte)
    __________________________________________________________________________________________________
    

## Word Embedding


```python
# Embedding for 1000 unique words mapped to 128-dimensional vectors
shared_embedding = layers.Embedding(1000, 128)

# Variable-length sequences of integers
text_input_a = keras.Input(shape=(None,), dtype="int32")
text_input_b = keras.Input(shape=(None,), dtype="int32")

# Reuse the same layer to encode both inputs
encoded_input_a = shared_embedding(text_input_a)
encoded_input_b = shared_embedding(text_input_b)
```


```python

```


```python
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris,load_diabetes,load_digits,load_linnerud,load_wine,load_breast_cancer
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn import svm
```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
