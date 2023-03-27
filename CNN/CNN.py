import tensorflow as tf
import tensorflow
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
print(X_train.shape)
print(X_test.shape)

'''
reshaping y_test, y_train to 1D arrays
'''

y_train = y_train.reshape(-1, )
y_test = y_test.reshape(-1, )
# print(y_train[:4])

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def plot_sample(X, y, index):
    plt.figure(figsize=(15, 2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


# plot_sample(X_train, y_train, 0)

'''
Normalizing the training data 
'''
X_train = X_train / 255
X_test = X_test / 255

'''
Creating ANN model for later comparison
'''

# ann = models.Sequential([
#     layers.Flatten(input_shape=(32,32,3)),
#     layers.Dense(3000, activation='relu'),
#     layers.Dense(1000, activation='relu'),
#     layers.Dense(10, activation='sigmoid')
# ])
#
# ann.compile(
#     optimizer='SGD',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy'])
#
# ann.fit(X_train, y_train, epochs=10)
# plt.show()

'''
Got an accuracy of 56.13 percent and further creating a Classification report
'''

# y_pred = ann.predict(X_test)
# y_pred_classes = [np.argmax(element) for element in y_pred]
# print('Classification Report \n', classification_report(y_test, y_pred_classes))


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cnn.fit(X_train, y_train, epochs=10)

cnn.evaluate(X_test, y_test)

y_pred = cnn.predict(X_test)
y_classes = [np.argmax(element) for element in y_pred]

print(y_classes[:5])  # predicted value using X_test
print(y_test[:5])  # truth value for that X_test
