import os

from keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

DATASET_PATH = 'dataset/'

num_classes = len(os.listdir(DATASET_PATH))

class_mode = "binary" if num_classes == 2 else "categorical"

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode=class_mode,
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode=class_mode,
    subset="validation"
)


model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid') if class_mode=="binary"
    else Dense(num_classes, activation='softmax')
])

loss_function = 'binary_crossentropy' if class_mode=="binary" else 'categorical_crossentropy'

model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

model.fit(train_data, epochs=10, validation_data=val_data)

test_loss, test_acc = model.evaluate(val_data)
print('\nTest accuracy:', test_acc)

model.save('image_classifier.h5')