#use on colab previously
#pip install tensorflow scikit-learn matplotlib numpy

# Load the MNIST dataset
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
mnist = datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Combine train and test for splitting
all_images = train_images.tolist() + test_images.tolist()
all_labels = train_labels.tolist() + test_labels.tolist()

# Convert back to numpy arrays
import numpy as np
all_images = np.array(all_images)
all_labels = np.array(all_labels)


# Split the data
# First split into training and remaining data (validation + test)
from sklearn.model_selection import train_test_split
X_train, X_rem, y_train, y_rem = train_test_split(all_images, all_labels, train_size=0.7, random_state=39)

# Split the remaining data into validation and testing sets
# 20% validation and 10% testing from the original dataset
# This means the split ratio from y_rem should be 20/(20+10) = 2/3 for validation
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=1/3, random_state=39) # 0.1 / (0.2 + 0.1) = 1/3

# Print the shapes of the datasets
print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Validation data shape:", X_val.shape)
print("Validation labels shape:", y_val.shape)
print("Testing data shape:", X_test.shape)
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)
X_test_processed = X_test.reshape(-1, 28, 28, 1)

#Training of model
Limpehmodel = models.Sequential()
Limpehmodel.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
Limpehmodel.add(layers.MaxPooling2D((2, 2)))
Limpehmodel.add(layers.Conv2D(64, (3, 3), activation='relu'))
Limpehmodel.add(layers.Flatten())
Limpehmodel.add(layers.Dense(16, activation='relu'))
Limpehmodel.add(layers.Dense(10, activation='softmax'))
Limpehmodel.summary()

#Compile and train the mode
Limpehmodel.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = Limpehmodel.fit(X_train, y_train, epochs=5,
                    validation_data=(X_val, y_val))

Limpehmodel.save('Zach_model.h5') #use for competition

# Reshape X_test to include the channel dimension
X_test_reshaped = X_test.reshape(-1, 28, 28, 1)

test_loss, test_acc = Limpehmodel.evaluate(X_test_reshaped,  y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")

plt.figure(figsize=(12,7))
for i in range(25):
    plt.subplot(5,5,i+1)
    # Use the reshaped X_test for plotting
    plt.imshow(X_test[i], cmap='gray')
    # Predict on a single image slice
    prediction = Limpehmodel.predict(X_test_reshaped[i:i+1], verbose=0)
    predicted_digit = np.argmax(prediction, axis=1)[0] # Get the first element as it's a single prediction
    plt.title(f'Expected:{y_test[i]}\n Predicted digit:{predicted_digit}')
plt.tight_layout()
plt.show()
