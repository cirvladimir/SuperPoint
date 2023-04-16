import tensorflow as tf


def vgg_block(inputs, filters, kernel_size, data_format,
              batch_normalization=True, kernel_reg=0., **params):
    x = tf.keras.layers.Conv2D(filters, kernel_size,
                               kernel_regularizer=tf.keras.regularizers.l2(kernel_reg),
                               data_format=data_format, **params)(inputs)
    if batch_normalization:
        x = tf.keras.layers.BatchNormalization(
            axis=1 if data_format == 'channels_first' else -1)(x)
    return x


def vgg_backbone(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'activation': tf.nn.relu, 'batch_normalization': True,
                   'kernel_reg': config.get('kernel_reg', 0.)}
    params_pool = {'padding': 'SAME', 'data_format': config['data_format']}

    x = vgg_block(inputs, 64, 3, **params_conv)
    x = vgg_block(x, 64, 3, **params_conv)
    x = tf.keras.layers.MaxPool2D((2, 2), **params_pool)(x)

    x = vgg_block(x, 64, 3, **params_conv)
    x = vgg_block(x, 64, 3, **params_conv)
    x = tf.keras.layers.MaxPool2D((2, 2), **params_pool)(x)

    x = vgg_block(x, 128, 3, **params_conv)
    x = vgg_block(x, 128, 3, **params_conv)
    x = tf.keras.layers.MaxPool2D((2, 2), **params_pool)(x)

    x = vgg_block(x, 128, 3, **params_conv)
    x = vgg_block(x, 128, 3, **params_conv)

    return x
