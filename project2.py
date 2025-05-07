# test harness for evaluating models on the cifar10 dataset
import sys
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.optimizers import AdamW
from keras.regularizers import l2
from sklearn.model_selection import train_test_split


def load_dataset(val_size=1000):
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # One-hot encode
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    # Split training into train and validation
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=val_size, random_state=42)
    return trainX, trainY, valX, valY, testX, testY


# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


def define_model_no_vgg(weight_decay=1e-4, lr=1e-4):

    model = Sequential()
    model.add(Conv2D(64, (2, 2), strides=(2, 2), activation='relu',
                     kernel_initializer='he_uniform',kernel_regularizer=l2(weight_decay), input_shape=(32, 32, 3)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay)))
    model.add(Dense(10, activation='softmax'))
    opt = AdamW(learning_rate=lr, weight_decay=weight_decay)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def define_model_one_vgg(weight_decay=1e-4, lr=1e-4):
    model = Sequential()
    model.add(Conv2D(64, (2, 2), strides=(2, 2), activation='relu',
                     kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay), input_shape=(32, 32, 3)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay)))
    model.add(Dense(10, activation='softmax'))
    opt = AdamW(learning_rate=lr, weight_decay=weight_decay)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def define_model_three_vgg(drop_rate=0.2, weight_decay=1e-4, lr=1e-4):
    model = Sequential()

    model.add(Conv2D(64, (2, 2), strides=(1, 1), activation='relu',
                     kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay),  input_shape=(32, 32, 3)))
    if drop_rate > 0:
        model.add(Dropout(drop_rate))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if drop_rate > 0:
        model.add(Dropout(drop_rate))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if drop_rate > 0:
        model.add(Dropout(drop_rate))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay)))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay)))
    if drop_rate > 0:
        model.add(Dropout(drop_rate))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(weight_decay)))
    if drop_rate > 0:
        model.add(Dropout(drop_rate))
    model.add(Dense(10, activation='softmax'))

    opt = AdamW(learning_rate=lr, weight_decay=weight_decay)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def summarize_diagnostics(history):
    n_epochs = len(history.history['loss'])
    epochs = range(1, n_epochs + 1)

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, history.history['accuracy'], label='Train acc')
    plt.plot(epochs, history.history['val_accuracy'], label='Val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    plt.figure()
    plt.plot(epochs, history.history['loss'], label='Train loss')
    plt.plot(epochs, history.history['val_loss'], label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


# run the test harness for evaluating a model
def main():
    trainX, trainY, valX, valY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    valX, _ = prep_pixels(valX, valX)

    epochs = 100
    drop_rate = 0.2
    model_choice = 3

    if model_choice == 1:
        model = define_model_no_vgg()
    elif model_choice == 2:
        model = define_model_one_vgg()
    elif model_choice == 3:
        model = define_model_three_vgg(drop_rate=drop_rate)

    history = model.fit(trainX, trainY, epochs=epochs, batch_size=64,
                        validation_data=(valX, valY), verbose=1)

    _, acc = model.evaluate(testX, testY, verbose=0)
    print(f'> Final test accuracy: {acc * 100:.2f}%')
    summarize_diagnostics(history)


if __name__ == '__main__':
    main()


