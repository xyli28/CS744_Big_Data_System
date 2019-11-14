import os 
import sys
import json
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

def build_and_compile_lenet_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(), 
                  metrics=['accuracy'])
    print(model.summary())
    return model

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

nWorker = sys.argv[1]
taskId  = sys.argv[2]


# single node
if nWorker == 1:
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': { 
            'worker' : [
                "10.10.1.1:2222"]},
        'task' : {'type' : 'worker', 'index' : taskId}
    })
# two nodes
elif nWorker == 2:
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': { 
            'worker' : [
                "10.10.1.1:2222",
                "10.10.1.2:2222"]},
        'task' : {'type' : 'worker', 'index' : taskId}
    })

# three nodes
else:
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': { 
            'worker' : [
                "10.10.1.1:2222",
                "10.10.1.2:2222",
                "10.10.1.3:2222"]},
        'task' : {'type' : 'worker', 'index' : taskId}
    })

#load dataset
datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = datasets['train'], datasets['test']

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(64)
eval_dataset = mnist_test.map(scale).batch(64)

with strategy.scope():
    model = build_and_compile_lenet_model()

model.fit(train_dataset,
                epochs=1)
eval_loss, eval_acc = model.evaluate(eval_dataset)
print('Loss: {}, Accuracy: {}'.format(eval_loss, eval_acc))
