import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = 128

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(
    "dataset",
    target_size=(img_size, img_size),
    batch_size=16,
    class_mode="binary",
    subset="training"
)

val = datagen.flow_from_directory(
    "dataset",
    target_size=(img_size, img_size),
    batch_size=16,
    class_mode="binary",
    subset="validation"
) 

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size,img_size,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(train, validation_data=val, epochs=10)

model.save("deepfake_model.h5")
