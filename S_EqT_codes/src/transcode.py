input_template = keras.layers.Input(shape=[None,1,16])
input_search = keras.layers.Input(shape=[47,1,16])

def x_corr_map(inputs):
    input_template = inputs[0]
    input_search = inputs[1]

    input_template = tf.transpose(input_template, perm=[1,2,0,3])
    input_search = tf.transpose(input_search, perm=[1,2,0,3])

    Hz, Wz, B, C = tf.unstack(tf.shape(input_template))
    Hx, Wx, Bx, Cx = tf.unstack(tf.shape(input_search))

    input_template = tf.reshape(input_template, (Hz, Wz, B*C, 1))
    input_search = tf.reshape(input_search, (1, Hx, Wx, Bx*Cx))

    feature_corr_res = tf.nn.depthwise_conv2d(input_template, input_search, strides=[1,1,1,1], padding='SAME')
    feature_corr_res = tf.concat(tf.split(feature_corr_res, batch_size, axis=3), axis=0)

    return  feature_corr_res

def x_corr_layer():
    return Lambda(x_corr_map, output_shape=(47,1))