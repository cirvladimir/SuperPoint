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
            prob = net(image)

        # TODO: Make non-maximum supporession work
        # if config['nms']:
        #     prob = box_nms(prob, config['nms'],
        #                    min_prob=config['detection_threshold'],
        #                    keep_top_k=config['top_k'])
        # print(prob)
        # print(type(prob))
        self.prob = prob
        pred = Threshold(config['detection_threshold'])(prob)
        # pred = tf.cast(tf.greater_equal(prob, detection_threshold), tf.int32)

        model = tf.keras.Model(inputs=inputs, outputs=pred)

        # Compile the model with the custom loss function
        model.compile(optimizer='adam', loss=self._loss)

        return model

    def _loss(self, y_true, y_pred):
        # TODO: This has state which will probably break with load().
        return tf.keras.losses.MeanSquaredError()(y_true, self.prob)
