import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 1. Завантаження датасету
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Використаємо меншу кількість даних (наприклад 1 000)
idx = np.random.choice(len(x_train), 1000, replace=False)
x_train_small = x_train[idx]
y_train_small = y_train[idx]

# Масштабування та приведення до формату (96,96,3)
x_train_small = tf.image.resize(tf.expand_dims(x_train_small, -1), (96,96))
x_test_small = tf.image.resize(tf.expand_dims(x_test, -1), (96,96))
x_train_small = tf.image.grayscale_to_rgb(x_train_small) / 255.0
x_test_small = tf.image.grayscale_to_rgb(x_test_small) / 255.0

# 2. VGG16 без верхніх шарів
base_model = keras.applications.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(96,96,3)
)
base_model.trainable = False

# 3. Functional API (без pruning)
inputs = keras.Input(shape=(96,96,3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = keras.Model(inputs, outputs)

# 4. Компіляція
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. Навчання (Feature Extraction)
history = model.fit(
    x_train_small, y_train_small,
    epochs=15,
    batch_size=16,
    validation_split=0.1,
    verbose=2,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)]
)

# 6. Fine-Tuning: розморожуємо верхні шари
for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    x_train_small, y_train_small,
    epochs=10,
    batch_size=16,
    validation_split=0.1,
    verbose=2
)

# 7. Оцінка
test_loss, test_acc = model.evaluate(x_test_small, y_test, verbose=0)
print(f"Точність VGG16 (Fine-Tuning): {test_acc:.4f}")

# 8. Збереження моделі
model.save("vgg16_model.h5")

# 9. Quantization у TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("vgg16_quantized.tflite", "wb") as f:
    f.write(tflite_model)

print("Модель збережено як vgg16_model.h5 та vgg16_quantized.tflite")

# 10. Збереження історії навчання у npz
np.savez("vgg_history.npz",
         fe_accuracy=history.history['accuracy'],
         fe_val_accuracy=history.history['val_accuracy'],
         fe_loss=history.history['loss'],
         fe_val_loss=history.history['val_loss'],
         ft_accuracy=history_ft.history['accuracy'],
         ft_val_accuracy=history_ft.history['val_accuracy'],
         ft_loss=history_ft.history['loss'],
         ft_val_loss=history_ft.history['val_loss'])

print("Історію навчання збережено у vgg_history.npz")
