import numpy as np
np.random.seed(42)
import pandas as pd

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from tensorflow.random import set_seed
set_seed(42)
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import OneHotEncoder


train_file = 'training files/emnist-balanced-train.csv'
train_df = pd.read_csv(train_file, header=None)

image_size = (28, 28)

def transform_images(images):
    images = images.reshape(images.shape[0], image_size[0], image_size[1]).astype(float)
    images /= 255
    images = np.fliplr(images)
    images = np.rot90(images, -1, axes=[1, 2])
    return images

X_train = train_df.drop(columns=[0]).values
X_train = transform_images(X_train)
y_train = train_df[0].values

ohe = OneHotEncoder(sparse_output=False)
y_train_ohe = ohe.fit_transform(y_train.reshape(y_train.shape[0], 1))

mapping_file = 'training files/mapping.txt'
mapping = {}
with open(mapping_file, 'r') as file:
    lines = file.read().strip().split('\n')
    for line in lines:
        key, value = map(int, line.split())
        mapping[key] = value

batch_size=128

model = Sequential([
    Input((image_size[0], image_size[1], 1), batch_size=batch_size),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(len(mapping), activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(
    X_train,
    y_train_ohe,
    epochs=50,
    batch_size=batch_size,
    validation_split=0.1,
    callbacks=[early_stopping]
)
model.save('model.keras')
