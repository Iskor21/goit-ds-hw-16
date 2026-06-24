import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
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
base_model.trainable = False   # Етап 1: Feature Extraction

# 3. Модель
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

# 4. Компіляція
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 5. EarlyStopping
callback = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
)

# 6. Навчання (Feature Extraction)
history_fe = model.fit(
    x_train_small, y_train_small,
    epochs=15,
    batch_size=16,
    validation_split=0.1,
    verbose=2,
    callbacks=[callback]
)

# 7. Fine-Tuning: розморожуємо верхні шари
for layer in base_model.layers[-4:]:  # останні 4 шари
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 8. Донавчання
history_ft = model.fit(
    x_train_small, y_train_small,
    epochs=10,
    batch_size=16,
    validation_split=0.1,
    verbose=2,
    callbacks=[callback]
)

# 9. Оцінка
test_loss, test_acc = model.evaluate(x_test_small, y_test, verbose=0)
print(f"Точність VGG16 (Feature Extraction + Fine-Tuning): {test_acc:.4f}")

# 10. Збереження моделі
model.save("vgg16_model.keras")

# 11. Збереження історії
combined_history = {
    "fe_accuracy": history_fe.history["accuracy"],
    "fe_val_accuracy": history_fe.history["val_accuracy"],
    "ft_accuracy": history_ft.history["accuracy"],
    "ft_val_accuracy": history_ft.history["val_accuracy"],
    "fe_loss": history_fe.history["loss"],
    "fe_val_loss": history_fe.history["val_loss"],
    "ft_loss": history_ft.history["loss"],
    "ft_val_loss": history_ft.history["val_loss"]
}

np.savez("vgg_history.npz", history=combined_history)