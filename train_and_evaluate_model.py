import tensorflow as tf
from tensorflow.keras import layers, models

# ===============================
# PATHS & CONSTANTS
# ===============================
DATASET_PATH = "My_Dataset/train"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

# ===============================
# LOAD DATASET
# ===============================
dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=True,
    seed=42
)

print("Class names:", dataset.class_names)

# ===============================
# SPLIT TRAIN / VALIDATION
# ===============================
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)

train_ds = dataset.take(train_size)
val_ds = dataset.skip(train_size)

# ===============================
# PERFORMANCE OPTIMIZATION
# ===============================
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ===============================
# DATA AUGMENTATION
# ===============================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ===============================
# MODEL
# ===============================
model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(224,224,3)),

    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# ===============================
# COMPILE
# ===============================
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ===============================
# TRAIN (HAPPENS ONCE)
# ===============================
print("\n--- Training Started ---\n")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ===============================
# EVALUATE (NO TRAINING HERE)
# ===============================
print("\n--- Evaluating Model ---\n")
val_loss, val_accuracy = model.evaluate(val_ds)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# ===============================
# SAVE MODEL
# ===============================
model.save("pothole_binary_model.h5")
print("\nModel saved as pothole_binary_model.h5")
