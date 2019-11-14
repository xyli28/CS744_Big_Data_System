import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import datetime
import time

# define the command line flags that can be sent
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_string("job_name", "worker", "either worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

clusterSpec_single = tf.train.ClusterSpec({
    "worker" : [
        "10.10.1.1:2222"
    ]
})

clusterSpec_cluster = tf.train.ClusterSpec({
    "ps" : [
        "10.10.1.1:2222"
    ],
    "worker" : [
        "10.10.1.1:2223",
        "10.10.1.2:2222"
    ]
})

clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps" : [
        "10.10.1.1:2222"
    ],
    "worker" : [
        "10.10.1.1:2223",
        "10.10.1.2:2222",
        "10.10.1.3:2222",
    ]
})

clusterSpec = {
    "single": clusterSpec_single,
    "cluster": clusterSpec_cluster,
    "cluster2": clusterSpec_cluster2
}

clusterinfo = clusterSpec[FLAGS.deploy_mode]
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
   
    # Loading dataset.
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Set up parameters
    learningRate = 0.01
    nEpoch = 100
    batchSize = 100

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, 784]) # image shape 28*28=784
    y = tf.placeholder(tf.float32, [None, 10])  # 10 categories

    # Set up weights
    W = tf.Variable(tf.random.uniform([784,10]))
    b = tf.Variable(tf.random.uniform([10]))
      
    # Set up model
    prediction  = tf.nn.softmax(tf.matmul(x, W) + b)
    loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(prediction), reduction_indices=1))
    
    # use gradient descent to train model
    opt = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)
        
    # make prediction with model
    good_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    
    # get accuracy
    accuracy = tf.reduce_mean(tf.cast(good_pred, tf.float32))

    # Initialize variables (assign the values to W, b, ... with prescribed values)
    init = tf.global_variables_initializer()
    
    # train
    start_time = datetime.datetime.now()
    with tf.Session() as ss:
        # initialize
        ss.run(init)
    
        # train 
        for epoch in range(nEpoch):
            epochLoss = 0
            nBatch = int(mnist.train.num_examples/batchSize)
            for i in range(nBatch):
                batch_x, batch_y = mnist.train.next_batch(batchSize)
                # optimize 
                _, c = ss.run([opt, loss], feed_dict={x: batch_x, y: batch_y})
                # keep track of average loss
                epochLoss += c / nBatch
            print("epoch:", epoch, "Loss:", epochLoss, "Accuracy:", ss.run(accuracy, 
                feed_dict = {x: mnist.test.images, y: mnist.test.labels}))
    
    end_time = datetime.datetime.now()
    print("Total_time_taken:%s" %(str((end_time-start_time).total_seconds())))
    print("process done") 
