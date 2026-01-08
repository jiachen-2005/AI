import tensorflow as tf
from tensorflow.keras import layers, models

(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1,28*28).astype('float32') /255.0
x_test=x_test.reshape(-1,28*28).astype('float32') /255.0

model = models.Sequential([layers.Dense(128,activation = 'relu', input_shape = (784,)),
                           #layers.Dropout(0.2),
                           #layers.Dense(64, activation = 'relu'),
                           #layers.Dropout(0.2),
                           layers.Dense(10, activation = 'softmax')])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs =10, batch_size=32, validation_split = 0.1)

test_loss, test_acc = model.evaluate(x_test,y_test)
print(test_loss,test_acc)
model.save("my first trained model.keras")