import numpy as np
import pandas as pd 
import h5py
import skimage
import matplotlib.pyplot as plt
import sys
from pprint import pprint
import sklearn

from subprocess import check_output

################################
##### ---- Read File  ---- #####
################################

# File Format
f = h5py.File('./input/food_c101_n10099_r64x64x3.h5', 'r')
print("Loaded h5 file")
# print(list(f.keys()))
print("Number of images: ", len(f["category"]))
print("Number of classes: ",len(f["category_names"]))

num_classes = len(f["category_names"])

class_names = [name.decode() for name in f["category_names"]]
#pprint(f"Class names: {class_names}")


# fig=plt.figure(figsize=(20,20))
# n=25
# col=5
# for i in range(n):
#     ax=fig.add_subplot(n/col, col, i+1)
#     #ax.set_title(f["category_names"][i].decode())
#     ax.imshow(f["images"][i])
# plt.savefig("./sample_show_64x64")
# sys.exit(0)



############ ###############
##### ---- Keras ---- ######
############################

from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import model_from_json
from keras import optimizers
from os.path import isfile

epochs = 10

x = preprocess_input(np.array(f["images"]))
y = np.array([np.array(j, dtype=int) for j in f["category"]])

if isfile("model.json") and isfile("model.h5"):
    # load json and create model
    model_json_file = open('model.json', 'r')
    model_json = model_json_file.read()
    model_json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")
else:
    print("Starting training from scratch")
    model = ResNet50(include_top=False, pooling='avg', weights='imagenet',
                     input_shape=(64, 64, 3))

    # Only make last 30 layers trainable
    print("Total Layer size: ", len(model.layers))
    for layer in model.layers[:-10]:
        layer.trainable = False
    for layer in model.layers[-10:]:
        layer.trainable = True                 

    out = model.output

    # fully-connected non-linear layer
    out = Dense(1024, activation='relu')(out)

    # logistic layer
    predictions = Dense(num_classes, activation='softmax')(out)

    model = Model(inputs=model.input, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.97, epsilon=1e-7),
              metrics=['accuracy'])

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)

for i in range(epochs):
    model.fit(train_x, train_y, batch_size=128, epochs=1, shuffle=True)
    print(f"{model.metrics_names} in epoch {i}: {model.evaluate(test_x, test_y)}")

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
