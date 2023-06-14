import tensorflow as tf

from .base_model import BaseModel, Mode
from .backbones.vgg import vgg_backbone
from .utils import detector_head, detector_loss, box_nms
from .homographies import homography_adaptation

from superpoint.utils.tools import dict_update


class Threshold(tf.keras.layers.Layer):
    def __init__(self, threshold, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.tf_threshold = tf.constant(
            [self.threshold], tf.float32, (1,))

    def call(self, inputs):
        return tf.cast(tf.greater_equal(inputs, self.tf_threshold), tf.int32)

    def get_config(self):
        config = super().get_config()
        config.update({
            "threshold": self.threshold
        })
        return config

@tf.function(experimental_compile=True)
def my_space_to_depth(x, grid_size, data_format):
    return tf.nn.space_to_depth(x, grid_size, data_format=data_format)


class MagicPoint(BaseModel):
    input_spec = {
        'image': {'shape': [None, None, None, 1], 'type': tf.float32}
    }
    required_config_keys = []
    default_config = {
        'data_format': 'channels_first',
        'kernel_reg': 0.,
        'grid_size': 8,
        'detection_threshold': 0.4,
        'homography_adaptation': {'num': 0},
        'nms': 0,
        'top_k': 0
    }

    def _model(self, mode, **config):
        config = dict_update(self.default_config,
                             config)
        config['training'] = (mode == Mode.TRAIN)
        self.config = config

        inputs = tf.keras.Input(shape=(None, None, 1))

        image = inputs

        def net(image):
            if config['data_format'] == 'channels_first':
                image = tf.transpose(image, [0, 3, 1, 2])
            features = vgg_backbone(image, **config)
            outputs = detector_head(features, **config)
            return outputs

        if (mode == Mode.PRED) and config['homography_adaptation']['num']:
            prob = homography_adaptation(image, net, config['homography_adaptation'])
        else:
            softmax, prob = net(image)

        # TODO: Make non-maximum supporession work (it's not used during training though)
        # if config['nms']:
        #     prob = box_nms(prob, config['nms'],
        #                    min_prob=config['detection_threshold'],
        #                    keep_top_k=config['top_k'])
        # print(prob)
        # print(type(prob))
        self.prob = softmax
        pred = Threshold(config['detection_threshold'])(prob)
        # pred = tf.cast(tf.greater_equal(prob, detection_threshold), tf.int32)

        model = tf.keras.Model(inputs=inputs, outputs=pred)

        # Compile the model with the custom loss function
        model.compile(optimizer='adam', loss=self._loss)

        return model

    def _loss(self, y_true, y_pred):
        # cfirst = self.config['data_format'] == 'channels_first'
        cfirst = False
        # TODO: This has state which will probably break with load().
        float_map = tf.cast(y_true[..., tf.newaxis], tf.float32)
        binned = my_space_to_depth(float_map, self.config['grid_size'],
                                   'NCHW' if cfirst else 'NHWC')
        shape = tf.concat([tf.shape(binned)[:3], [1]], axis=0)
        labels = tf.concat([binned, tf.ones(shape)], axis=3)
        argmax = tf.argmax(labels + tf.random.uniform(tf.shape(labels), 0, 0.1),
                       axis=3)
        return tf.keras.losses.sparse_categorical_crossentropy(argmax,
                                                            self.prob,
                                                            True,
                                                            axis=(1 if self.config['data_format'] == 'channels_first' else 3))
        # return tf.keras.losses.MeanSquaredError()(y_true, self.prob)
