import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

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
    nEpoch = 1
    batchSize = 100
    nBatch = int(mnist.train.num_examples/batchSize)

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=clusterinfo)):

        # tf Graph Input
        x = tf.placeholder(tf.float32, [None, 784]) # image shape 28*28=784
        y = tf.placeholder(tf.float32, [None, 10])  # 10 categories

        # Set up weights
        W = tf.Variable(tf.random.uniform([784,10]))
        b = tf.Variable(tf.random.uniform([10]))
          
        # Set up model
        prediction  = tf.nn.softmax(tf.matmul(x, W) + b)
        loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(prediction), reduction_indices=1))
        global_step = tf.contrib.framework.get_or_create_global_step()
        
        # Create any optimizer to update the variables, say a simple SGD:
        opt1 = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
  
        # Wrap the optimizer with sync_replicas_optimizer
        opt = tf.train.SyncReplicasOptimizer(opt1, replicas_to_aggregate=3,
            total_num_replicas=3)

        train_opt = opt.minimize(loss, global_step=global_step)

        # make prediction with model
        good_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    
        # get accuracy
        accuracy = tf.reduce_mean(tf.cast(good_pred, tf.float32))

        ## You can create the hook which handles initialization and queues.
        sync_replicas_hook = opt.make_session_run_hook(FLAGS.task_index == 0)
   
    # Initialize variables (assign the values to W, b, ... with prescribed values)
    init = tf.global_variables_initializer()


    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0)) as mon_sess:
        # initialize
        mon_sess.run(init)
        # train 
        for i in range(nEpoch):
            # Run a training step asynchronously.
            # See `tf.train.SyncReplicasOptimizer` for additional details on how to
            # perform *synchronous* training.
            # mon_sess.run handles AbortedError in case of preempted PS.
            epochLoss = 0
            for i in range(nBatch):
                batch_x, batch_y = mnist.train.next_batch(batchSize)
                # optimize 
                #while not mon_sess.should_stop():
                _, c = mon_sess.run([train_opt, loss], feed_dict={x: batch_x, y: batch_y})
                # keep track of average loss
                epochLoss += c / nBatch

            print("loss", epochLoss, "Accuracy:", mon_sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
