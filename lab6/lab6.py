import keras
from keras.datasets import mnist, cifar10, fashion_mnist
from keras.utils import to_categorical
from keras import models, layers
from keras.optimizers import SGD
from keras.models import load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import cv2
import numpy as np

def train_numbers(model_file):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype("float32") / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    network = Sequential()
    network.add(Dense(512, activation="relu", input_shape=(28*28,)))
    network.add(Dense(10, activation="softmax"))
    network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    network.fit(train_images, train_labels, epochs=10, batch_size=128)
    test_loss, test_acc = network.evaluate(test_images, test_labels)

    network.save(model_file)
    #network.save_weights("model_weights.h5")

def eval_numbers(model_file, test_file):
    model = load_model(model_file)

    # cv2.imshow(test_file, cv2.imread(test_file))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    tst = 255-cv2.imread(test_file, 0)
    tst = cv2.resize(tst, (28, 28))
    tst = tst.reshape((1, 28 * 28))
    tst = tst.astype("float32") / 255

    pred = list(model.predict(tst)[0])
    print(f"File {test_file} - predicted number {pred.index(max(pred))}")

model_file = "mnist.h5"
test = "2.png"
#train_numbers(model_file)
eval_numbers(model_file, test)

def train_cifar(model_file):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10, batch_size=64)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    model.save(model_file)

def eval_cifar(model_file, files):
    model = load_model(model_file)
    for test_file in files:
        # cv2.imshow(test_file, cv2.imread(test_file))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        tst = cv2.imread(test_file, 1)
        tst = cv2.resize(tst, (32, 32))
        tst = tst.reshape((1, 32, 32, 3))
        tst = tst.astype("float32") / 255

        pred = model.predict(tst)
        classes = ["airplanes", "cars", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks"]
        print(f"File {test_file} - predicted class {classes[np.argmax(pred[0])]}")

model_file = "cifar10.h5"
test = ["ship.jpg", "plane.jpeg", "car.jpg"]
#train_cifar(model_file)
eval_cifar(model_file, test)

def train_fashion(model_file):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=15, batch_size=32)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    model.save(model_file)

def eval_fashion(model_file, files):
    model = load_model(model_file)
    for test_file in files:
        # cv2.imshow(test_file, cv2.imread(test_file))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        tst = cv2.imread(test_file, 0)
        tst = cv2.resize(tst, (28, 28))
        tst = tst.reshape((1, 28, 28, 1))
        tst = tst.astype("float32") / 255

        pred = model.predict(tst)
        classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        print(f"File {test_file} - predicted class {classes[np.argmax(pred[0])]}")

model_file = "fashion.h5"
test = ["jeans.jpg", "coat.jpg", "hoodie.jpg"]
#train_fashion(model_file)
eval_fashion(model_file, test)
