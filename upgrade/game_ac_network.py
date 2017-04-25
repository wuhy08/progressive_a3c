# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import tf_common as tfc
import constants
import pickle

FLAGS = tf.app.flags.FLAGS
IMG_SIZE = constants.img_size
HIST_FRM = constants.history_frames

class Network(object):
    def __init__(self, name="agent"):
        self.name = name
        with tf.device(constants.device):
            with tf.variable_scope(name):
                self.create_pnn()

    def debug(self, sess):
        fd = {self.s: np.ones((1, IMG_SIZE, IMG_SIZE, 4))}

        a = sess.run(self.pi, feed_dict=fd)
        b = sess.run(self.pi, feed_dict={self.s: np.zeros((1, IMG_SIZE, IMG_SIZE, 4))})

        exit()

    def create_pnn(self):
        self.s = tf.placeholder(
            "float", 
            [None, IMG_SIZE, IMG_SIZE, HIST_FRM],
            "state"
        )
        self.train_vars = []
        self.var_dict = {}
        self.all_vars = []
        self.col_hiddens = []

        for i in range(len(constants.tasks)):
            print(">>>>>>>>>>>>>>>>>>>>>>")
            p, v, col_vars, col_h = create_column(
                constants.tasks, i, self.s, self.col_hiddens)
            vvv = []#col_vars[:-4]
            if i == len(constants.tasks)-1:
                vvv = col_vars

            self.all_vars.extend(col_vars)

            for var in vvv:
                if var in tf.trainable_variables():
                    self.train_vars.append(var)

            for col_var in col_vars:
                n = col_var.name
                n = n[n.index('/'):]
                self.var_dict[n] = col_var

            self.col_hiddens.append(col_h)

            if i == len(constants.tasks)-1:
                print("setting policy and value tensors.")
                self.pi = p
                self.v = v
            print("<<<<<<<<<<<<<<<<<<<<<")

        self.columns = len(self.col_hiddens)
        self.layers = len(self.col_hiddens[0])

        print([v.name for v in self.train_vars])
        print("%i trainable weight variables." % len(self.train_vars))

    def run_policy_and_value(self, sess, s_t):
        pi_out, v_out = sess.run([self.pi, self.v], feed_dict={self.s: [s_t]})
        return (pi_out[0], v_out[0])

    def run_policy(self, sess, s_t):
        pi_out = sess.run(self.pi, feed_dict={self.s: [s_t]})
        return pi_out[0]

    def run_value(self, sess, s_t):
        v_out = sess.run(self.v, feed_dict={self.s: [s_t]})
        return v_out[0]

    def get_train_vars(self):
        return self.train_vars

    def evaluate_vars(self, sess):
        for v in self.train_vars:
            print(v.name)
            print(sess.run(v))
            print("="*20)

    def prepare_loss(self, entropy_beta):
        with tf.device(constants.device):
            # taken action (input for policy)
            self.a = tf.placeholder("float", [None, FLAGS.action_size])

            # temporary difference (R-V) (input for policy)
            self.td = tf.placeholder("float", [None])

            # avoid NaN with clipping when value in pi becomes zero
            log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))

            # policy entropy
            entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)

            # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
            policy_loss = - tf.reduce_sum(
                tf.reduce_sum(tf.multiply(log_pi, self.a), axis=1) * self.td + entropy * entropy_beta)

            # R (input for value)
            self.r = tf.placeholder("float", [None])

            # value loss (output)
            # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
            value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

            # gradienet of policy and value are summed up
            self.total_loss = policy_loss + value_loss

    def sync_from(self, src_netowrk, name=None):
        src_vars = src_netowrk.all_vars
        dst_vars = self.all_vars

        sync_ops = []

        with tf.device(constants.device):
            with tf.name_scope(values=[], name=name, default_name="GameACNetwork") as name:
                for (src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops, name=name)

    def save(self, sess, path):
        weights = {}

        for name, var in self.var_dict.items():
            weights[name] = sess.run(var)

        pickle.dump(weights, open(path, "wb"))

    def load(self, sess, path):
        weights = pickle.load(open(path, "rb"))

        print("CURRENT MODEL VARIABLES: " + str([v for v in self.var_dict.keys()]))
        print("LOADING WEIGHTS FOR: " + str(weights.keys()))

        for suffix, values in weights.items():
            #if "p_" not in suffix and "v_" not in suffix:
                #print(suffix)
            var_name = self.name + suffix
            sess.run(tf.assign(self.var_dict[suffix], values))
            print("loaded values for: %s" % var_name)
            #else:
            #    print("!!!skipping")



    def get_grads(self):  # to be implemented later
        # print(self.col_hiddens[k][i])
        grads = [[None for i in range(self.layers)] for k in range(self.columns)]
        for k in range(self.columns):
            for i in range(self.layers):
                norm = self.col_hiddens[k][i]#/tf.reduce_sum(self.col_hiddens[k][i])
                g = tf.gradients(tf.log(self.pi), norm)[0]

                grads[k][i] = g
        return grads

    def sample_fisher(self, sess, state, grads):
        dpdh = []

        for k in range(self.columns):
            print(k)
            col_dpdh = []
            for i in range(self.layers):
                print(i)
                dpdh_mat = np.power(
                    sess.run(
                        grads[k][i], 
                        feed_dict={self.s: [state]}
                    ), 
                    2.0
                )
                if len(dpdh_mat.shape) == 4:
                    dpdh_mat = np.sum(dpdh_mat, (0, 1, 2, 3))
                else:
                    # print(grad)
                    dpdh_mat = np.sum(dpdh_mat, (0, 1))
                # self.get_current_dpdh(sess, i, k, state)
                # print(dpdh_mat.shape)
                # print(self.col_hiddens[k][i])
                # print(dpdh_mat)

                col_dpdh.append(dpdh_mat)
            dpdh.append(col_dpdh)

        # fishers = []

        # for k in range(self.columns):
        #    lyrs = []
        #    for i in range(self.layers):
        #        f = np.dot(dpdh[k][i], dpdh[k][i].T)
        #        lyrs.append(f)
        #    fishers.append(lyrs)

        return dpdh  # fishers

def create_column(col_names, i_task, state, col_hiddens):
    print("creating column {}".format(i_task))
    arch = [
        [8, HIST_FRM, 16, 4],  # size, in, out, stride
        [4, 16, 32, 2],
        [256],
        -1
    ]
    train_vars = []
    lats = []
    c_lats = []
    # If current task is not task 0, build laterals
    if i_task > 0:
        with tf.variable_scope("laterals"):
            print("creating lateral connections to column {}".format(i_task))
            # From task 0 to current task
            for i_col in range(i_task):
                hiddens = col_hiddens[i_col]
                print("##{}".format(len(hiddens)))
                col_lats = []
                print("creating laterals {} -> {} ".format(i_col, i_task))
                with tf.variable_scope("{}_to_{}".format(col_names[i_col], col_names[i_task])):

                    # From the first layer to last layer
                    for i_layer in range(len(hiddens)):
                        layer_lats = []
                        print("###Layer {}".format(i_layer))
                        dest_h_shape = arch[i_layer + 1] # The shape of h in destination layer

                        with tf.variable_scope("layer{}to{}".format(i_layer, i_layer+1)):
                            orig_h = hiddens[i_layer]
                            #OLD: tf.stop_gradient(hiddens[i_layer]) #origin
                            print("layer {} -> {}".format(i_layer, i_layer + 1))
                            if dest_h_shape == -1: # to policy and value layer
                                with tf.variable_scope("policy"):
                                    lat_h_p, lat_vars_p = lateral_connection(
                                        orig_h, [FLAGS.action_size], i_task
                                    )
                                with tf.variable_scope("value"):
                                    lat_h_v, lat_vars_v = lateral_connection(
                                        orig_h, [1], i_task
                                    )
                                layer_lats.append(lat_h_p)
                                layer_lats.append(lat_h_v)
                                train_vars.extend(lat_vars_p)
                                train_vars.extend(lat_vars_v)
                            else:
                                lat_h, lat_vars = lateral_connection(
                                    orig_h, dest_h_shape, i_task, arch[i_layer + 1]
                                )

                                layer_lats.append(lat_h)
                                train_vars.extend(lat_vars)

                            col_hiddens[i_col][i_layer] = orig_h

                        col_lats.append(layer_lats)
                lats.append(col_lats)

        #print("columns: %i" % (len(lats) + 1))
        #print("hidden layers: %i" % (len(lats[0])))
        #print("hidden shapes: %s" % col_hiddens[0])

        #concatenate same-layer lateral connections
        for i in range(len(lats[0])):
            if arch[i+1] == -1:
                to_policy_list = [lats[k][i][0] for k in range(len(lats))]
                to_value_list = [lats[k][i][1] for k in range(len(lats))]
                to_policy = tf.reduce_sum(to_policy_list, 0)
                to_value = tf.reduce_sum(to_value_list, 0)

                c_lats.append([to_policy, to_value])

                print("summing ->policy and ->value layers")
                print(to_policy_list)
                print("=>")
                print(to_policy)
                print("&")
                print(to_value_list)
                print("=>")
                print(to_value)
            else:
                h_list = [lats[k][i][0] for k in range(len(lats))]

                if len(arch[i+1]) > 1:
                    c = tf.reduce_sum(h_list, 0)
                    c_lats.append(c)
                    print("summing convolutional layers")
                    print(h_list)
                    print("=>")
                    print(c)
                else:
                    c = tf.reduce_sum(h_list, 0)
                    c_lats.append(c)
                    print("summing fully connected layers")
                    print(h_list)
                    print("=>")
                    print(c)

            print("~~~")

    print("done summing layers")
    #print("c lats:")
    #print(c_lats)

    def add_lat(layer, i, act=tf.nn.relu):

        if i_task <= 0:
            if act is None:
                return layer[0], layer[1], layer[2]
            else:
                return act(layer[0]), layer[1], layer[2]
        elif len(i) == 1:
            print("adding {} and {}".format(layer[0], c_lats[i[0]]))
            return act(layer[0]+c_lats[i[0]]), layer[1], layer[2]
        else:
            if act is None:
                print("(value) adding {} and {}".format(layer[0], c_lats[i[0]][i[1]]))
                return layer[0] + c_lats[i[0]][i[1]], layer[1], layer[2]
            else:
                print("(policy) adding {} and {}".format(layer[0], c_lats[i[0]][i[1]]))
                return act(layer[0]+c_lats[i[0]][i[1]]), layer[1], layer[2]

    train = i_task == len(constants.tasks)-1
    print("column trainable: {}".format(train))

    with tf.variable_scope(col_names[i_task]):
        #resized = tf.image.resize_images(state, IMG_SIZE, IMG_SIZE)

        c1, w1, b1 = tfc.conv2d(
            "c1", 
            state, 
            arch[0][1], 
            arch[0][2], 
            size=arch[0][0], 
            stride=arch[0][3], 
            trainable=train
        )
        c2, w2, b2 = add_lat(
            tfc.conv2d(
                "c2", 
                c1, 
                arch[1][1], 
                arch[1][2], 
                size=arch[1][0], 
                stride=arch[1][3], 
                act=None, 
                trainable=train
            ), 
            [0]
        )

        c2_size = np.prod(c2.get_shape().as_list()[1:])
        c2_flat = tf.reshape(c2, [-1, c2_size])

        if i_task <= 0:
            h_fc1, w3, b3 = tfc.fc(
                "fc1", 
                c2_flat, 
                c2_size, 
                arch[2][0], 
                trainable=train
            )
        else:
            h_fc1, w3, b3 = tfc.fc(
                "fc1", 
                c2_flat, 
                c2_size, 
                arch[2][0], 
                act=None, 
                trainable=train
            )

            lat = c_lats[1]
            print("adding {} and {}".format(h_fc1, lat))
            lat_size = np.prod(lat.get_shape().as_list()[1:])
            lat_flat = tf.reshape(lat, [-1, lat_size])
            h_fc1 = tf.nn.relu(h_fc1 + lat_flat)

        pi, wp, bp = add_lat(
            tfc.fc(
                "p_fc", 
                h_fc1, 
                arch[2][0], 
                FLAGS.action_size, 
                act=None, 
                trainable=train
            ), 
            [2, 0], 
            tf.nn.softmax
        )
        v_, wv, bv = add_lat(
            tfc.fc(
                "v_fc", 
                h_fc1, 
                arch[2][0], 
                1, 
                act=None, 
                trainable=train
            ), 
            [2, 1], 
            None
        )

        v = tf.reshape(v_, [-1])

        train_vars.extend(
            [w1, b1, 
             w2, b2, 
             w3, b3, 
             wp, bp, 
             wv, bv]
        )

        col_vars = pi, v, train_vars, [c1, c2, h_fc1]

        print("policy: {}".format(pi))
        print("last fc: {}".format(h_fc1))
        print("wp: {}".format(wp.name))
        print("created column {}.".format(i_task))

        return col_vars

def lateral_connection(orig_hidden, dest_shape, i_task, current_op_shape=None):
    print("adapter origin: %s" % orig_hidden.name)
    train = i_task == len(constants.tasks)-1
    #print(i_task)
    #print(len(constants.tasks)-1)
    print("lateral trainable: %s" % train)
    nonlinear = True

    omit_b = True

    a = tf.get_variable(
        name="adapter", 
        shape=[1], 
        initializer=tf.constant_initializer(1), 
        trainable=train
    )
    ah = tf.multiply(a, orig_hidden)

    if nonlinear:
        if len(orig_hidden.get_shape().as_list()) == 4:
            maps_in = ah.get_shape().as_list()[3]
            nic = int(maps_in / (2.0 * (i_task)))
            lateral, w1, b1 = tfc.conv2d(
                "V", 
                ah, 
                maps_in, 
                nic, 
                size=1, 
                stride=1, 
                trainable=train
            )  # reduction (keep bias)

            print("1) conv 1x1: {}".format(w1.get_shape()))

            if len(dest_shape) > 1:   # conv layer to conv layer
                lateral, w2, _ = tfc.conv2d(
                    "U", 
                    lateral, 
                    nic, 
                    current_op_shape[2], 
                    size=current_op_shape[0],
                    stride=current_op_shape[3], 
                    act=None, 
                    omit_bias=omit_b, 
                    padding="SAME", 
                    trainable=train
                )
                print("2) conv 1x1: {}".format(w2.get_shape()))
                print("end result: {}".format(lateral.name))

                return lateral, [w1, b1, w2]

            else:  # conv layer to fc layer
                c_size = np.prod(lateral.get_shape().as_list()[1:])
                c_flat = tf.reshape(lateral, [-1, c_size])
                lateral, w2, _ = tfc.fc(
                    "U", 
                    c_flat, 
                    c_size, 
                    dest_shape[0], 
                    act=None, 
                    omit_bias=omit_b, 
                    trainable=train
                )
                print("2) flattened conv fc: {}".format(w2.get_shape()))
                print("end result: {}".format(lateral.name))

                return lateral, [w1, b1, w2]

        else:  # fc layer to fc layer
            n_in = ah.get_shape().as_list()[1]
            ni = int(n_in / (2.0 * (i_task)))
            lateral, w1, b1 = tfc.fc(
                "V", 
                ah, 
                n_in, 
                ni, 
                trainable=train
            )  # reduction (keep bias)
            print("1) fc: {}".format(w1.get_shape()))
            lateral, w2, _ = tfc.fc(
                "U", 
                lateral, 
                ni, 
                dest_shape[0], 
                act=None, 
                omit_bias=omit_b, 
                trainable=train
            ) # to be added to next hidden
            print("2) fc: {}".format(w2.get_shape()))
            print("end result: {}".format(lateral.name))

            return lateral, [w1, b1, w2]
    else:
        if len(orig_hidden.get_shape().as_list()) == 4:
            maps_in = ah.get_shape().as_list()[3]

            if len(dest_shape) > 1:   # conv layer to conv layer
                lateral, w2, _ = tfc.conv2d(
                    "U", 
                    ah, 
                    maps_in, 
                    current_op_shape[2], 
                    size=current_op_shape[0],
                    stride=current_op_shape[3], 
                    act=None, 
                    omit_bias=omit_b, 
                    padding="SAME", 
                    trainable=train
                )
                return lateral, [w2]

            else:  # conv layer to fc layer
                c_size = np.prod(ah.get_shape().as_list()[1:])
                c_flat = tf.reshape(ah, [-1, c_size])
                lateral, w2, _ = tfc.fc(
                    "U", 
                    c_flat, 
                    c_size, 
                    dest_shape[0], 
                    act=None, 
                    omit_bias=True, 
                    trainable=train
                )
                return lateral, [w2]

        else:  # fc layer to fc layer
            n_in = ah.get_shape().as_list()[1]
            lateral, w2, _ = tfc.fc(
                "U", 
                ah, 
                n_in, 
                dest_shape[0], 
                act=None, 
                omit_bias=True, 
                trainable=train
            ) # to be added to next hidden
            return lateral, [w2]
