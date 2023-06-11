import tensorflow as tf

from .base_model import BaseModel, Mode
from .backbones.vgg import vgg_backbone
from .utils import detector_head, detector_loss, box_nms
from .homographies import homography_adaptation

from superpoint.utils.tools import dict_update


class MagicPointModel(tf.keras.Model):
    def __init__(self):
        super(MagicPointModel, self).__init__()

    def call(self, x):
        pass


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

        inputs = tf.keras.Input(shape=(120, 160, 1))

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
        detection_threshold = tf.constant(
            [config['detection_threshold']], tf.float32, (1,))
        pred = tf.keras.layers.Lambda(lambda x: tf.cast(
            tf.greater_equal(x, detection_threshold), tf.int32), name="Afdsafs")(prob)
        # pred = tf.cast(tf.greater_equal(prob, detection_threshold), tf.int32)

        model = tf.keras.Model(inputs=inputs, outputs=pred)

        # Define a custom loss function that computes the loss based on the logits layer
        def custom_loss(y_true, y_pred):
            return tf.keras.losses.MeanSquaredError()(y_true, prob)
            # logits = model.layers[-2].output  # Get the logits layer
            # logits = prob
            # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            #     labels=y_true, logits=logits))
            # return loss

        # Compile the model with the custom loss function
        model.compile(optimizer='adam', loss=custom_loss)

        return model
