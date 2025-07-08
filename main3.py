import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from PIL import Image
import matplotlib.pyplot as plt

# Путь к датасету (папка должна содержать подкаталоги для каждого класса, например 'cats' и 'dogs')
# Path to dataset folder (should contain subfolders for each class, e.g., 'cats' and 'dogs')
DATASET_PATH = 'dataset/'

# Размер изображений и размер батча для обучения
# Image size and batch size for training
IMG_SIZE = 128
BATCH_SIZE = 32

# Подсчет количества классов (ожидается 2: коты и собаки)
# Count number of classes (expected 2: cats and dogs)
num_classes = len(os.listdir(DATASET_PATH))

# Установка режима классов: binary для 2 классов, categorical — для большего числа
# Set class_mode depending on number of classes: 'binary' for 2 classes, 'categorical' otherwise
class_mode = 'binary' if num_classes == 2 else 'categorical'

# Создание генератора изображений с аугментацией и разделением на обучающую и валидационную выборки
# Create ImageDataGenerator with augmentation and validation split (20%)
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Нормализация пикселей к диапазону [0,1]
    # Normalize pixel values to [0,1]
    rotation_range=20,  # Случайный поворот до 20 градусов
    # Random rotation up to 20 degrees
    width_shift_range=0.2,  # Сдвиг по ширине
    # Horizontal shift
    height_shift_range=0.2,  # Сдвиг по высоте
    # Vertical shift
    shear_range=0.15,  # Сдвиг по сдвигу (shear)
    # Shear transformation
    zoom_range=0.15,  # Масштабирование
    # Zoom in/out
    horizontal_flip=True,  # Случайное отражение по горизонтали
    # Random horizontal flip
    fill_mode='nearest',  # Заполнение пустых пикселей после трансформаций
    # Fill mode for empty pixels after transforms
    validation_split=0.2  # 20% данных для валидации
    # 20% data reserved for validation
)

# Генератор обучающих данных
# Training data generator
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode=class_mode,
    subset='training',  # Использовать обучающую часть
    # Use training subset
    shuffle=True  # Перемешивать данные
    # Shuffle data
)

# Генератор валидационных данных
# Validation data generator
validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode=class_mode,
    subset='validation',  # Использовать валидационную часть
    # Use validation subset
    shuffle=False
)

# Загрузка предобученной модели MobileNetV2 без верхних слоев
# Load pretrained MobileNetV2 model without top layers
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'  # Веса, обученные на ImageNet
    # Weights pretrained on ImageNet
)
base_model.trainable = False  # Заморозить базовую модель на начальном этапе
# Freeze base model initially

# Создаем новую модель, добавляя к базовой моделью свои слои
# Build the full model by adding custom top layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Глобальный средний пуллинг (сжатие по пространственным измерениям)
    # Global average pooling to reduce feature maps
    layers.Dense(128, activation='relu'),  # Полносвязный слой с 128 нейронами и ReLU активацией
    # Fully connected layer with 128 neurons and ReLU activation
    layers.Dropout(0.5),  # Dropout для уменьшения переобучения
    # Dropout to reduce overfitting
    layers.Dense(1 if num_classes == 2 else num_classes,
                 activation='sigmoid' if num_classes == 2 else 'softmax') # Выходной слой
])

# Компиляция модели с оптимизатором Adam и соответствующей функцией потерь
# Compile model with Adam optimizer, appropriate loss function, and accuracy metric
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy',
    metrics=['accuracy']
)

# Колбэки для управления скоростью обучения и ранней остановки
# Callbacks to control learning rate and early stopping
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    # Уменьшение LR при отсутствии улучшения
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
    # Ранняя остановка
]

# Параметры обучения
# Training parameters
initial_epochs = 5  # Кол-во эпох для начального обучения (замороженная базовая модель)
fine_tune_epochs = 3  # Кол-во эпох для дообучения (размороженная базовая модель)

# Начальное обучение только верхних слоев
# Initial training on top layers only
history = model.fit(
    train_generator,
    epochs=initial_epochs,
    steps_per_epoch=100,  # Ограничение числа шагов на эпоху для ускорения
    # Limit steps per epoch to reduce training time
    validation_data=validation_generator,
    validation_steps=20,
    callbacks=callbacks
)

# Разморозка базовой модели для дообучения
# Unfreeze base model for fine-tuning
base_model.trainable = True

# Перекомпиляция с меньшим learning rate для дообучения
# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy',
    metrics=['accuracy']
)

# Дообучение модели целиком
# Continue training (fine-tuning)
history_fine = model.fit(
    train_generator,
    epochs=initial_epochs + fine_tune_epochs,
    initial_epoch=history.epoch[-1],
    steps_per_epoch=100,
    validation_data=validation_generator,
    validation_steps=20,
    callbacks=callbacks
)

# Сохранение модели
# Save the trained model to disk
model.save('image_classifier_improved.h5')


# Функция для пакетного предсказания и отображения нескольких изображений с результатами
# Function for batch prediction and visualization of multiple images
def predict_images_batch(image_paths):
    # Загрузка сохраненной модели
    # Load the saved model
    model = tf.keras.models.load_model('image_classifier_improved.h5')

    # Получение словаря: индекс класса -> название класса
    # Get mapping from class indices to class names
    class_names = train_generator.class_indices
    inv_class_names = {v: k for k, v in class_names.items()}

    plt.figure(figsize=(12, 12))

    for i, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            print(f'{image_path} does not exist')
            continue
        try:
            img = Image.open(image_path)
            img.verify()
        except (OSError, IOError):
            print(f'{image_path} could not be opened')
            continue

        # Подготовка изображения для модели
        # Prepare image for model input
        img = Image.open(image_path).resize((IMG_SIZE, IMG_SIZE))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = tf.expand_dims(img_array, 0)

        prediction = model.predict(img_array)

        # Получение предсказанного класса и вероятности
        # Get predicted class and confidence score
        if class_mode == 'binary':
            predicted_class = inv_class_names[int(prediction[0] > 0.5)]
            confidence = prediction[0][0]
        else:
            predicted_class = inv_class_names[tf.argmax(prediction[0])]
            confidence = tf.reduce_max(prediction[0])

        # Отобразить изображение с предсказанием и уверенностью
        # Display image with predicted label and confidence
        plt.subplot(2, 2, i + 1)
        plt.imshow(Image.open(image_path))
        plt.title(f'{predicted_class} ({confidence:.2f})')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Пример вызова функции для 4 изображений
# Example usage: predict and show 4 images
predict_images_batch([
    'image1.jpg',
    'image2.jpg',
    'image3.jpg',
    'image4.jpg'
])
