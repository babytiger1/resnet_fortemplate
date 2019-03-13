from base.base_model import BaseModel
import tensorflow as tf


class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_A_net(self,net,layer_id, layer_num):
        self.ACTIVATION_FN = self.config.activation_fn
        self.DROPOUT = self.config.dropout
        self.BATCH_NORM = self.config.batch_norm
        with tf.variable_scope(
                'dnn_{}/hiddenlayer_{}'.format(layer_id, layer_num),
                values=(net,)) as hidden_layer_scope:
            net = tf.layers.dense(
                net,
                units=layer_num,
                activation=self.ACTIVATION_FN,
                kernel_initializer=tf.glorot_uniform_initializer(
                ),  # also called Xavier uniform initializer.
                name=hidden_layer_scope)

            if self.DROPOUT is not None :
                net = tf.layers.dropout(net, rate=self.DROPOUT, training=True)
            if self.BATCH_NORM and self.trainable==1:
                net = tf.layers.batch_normalization(net)
        return net



    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.trainable = tf.cond(self.is_training , lambda:1 ,lambda:0)
        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        net = self.x
        net_collections = []
        len = 0
        net_collections.append( net )


        self.y = tf.placeholder(tf.float32, shape=[None, 10])
        for layer_id,layer_num in enumerate(self.config.hidden_layers):
            net = self.build_A_net( net, layer_id , layer_num)
            net_collections.append(net)
            len = len+1
            if layer_id%2 ==0 and layer_id!=0:
                print(layer_id)
                net = tf.concat([net_collections[len-2], net_collections[len]], axis=1)
                net_collections.append(net)
                len= len+1


          #  add_layer_summary(net, hidden_layer_scope.name)

        pre_net =tf.layers.dense(net,10,'relu')
        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=pre_net))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                         global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(pre_net, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

