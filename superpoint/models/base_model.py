from abc import ABCMeta, abstractmethod
import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
from tqdm import tqdm
import os.path as osp
import itertools
import logging
import datetime

from superpoint.utils.tools import dict_update


class Mode:
    TRAIN = 'train'
    EVAL = 'eval'
    PRED = 'pred'


class BaseModel(metaclass=ABCMeta):
    """Base model class.

    Arguments:
        data: A dictionary of `tf.data.Dataset` objects, can include the keys
            `"training"`, `"validation"`, and `"test"`.
        n_gpus: An integer, the number of GPUs available.
        data_shape: A dictionary, where the keys are the input features of the prediction
            network and the values are the associated shapes. Only required if `data` is
            empty or `None`.
        config: A dictionary containing the configuration parameters.
            Entries `"batch_size"` and `"learning_rate"` are required.

    Models should inherit from this class and implement the following methods:
        `_model`, `_loss`, and `_metrics`.
    Additionally, the following static attributes should be defined:
        input_spec: A dictionary, where the keys are the input features (e.g. `"image"`)
            and the associated values are dictionaries containing `"shape"` (list of
            dimensions, e.g. `[N, H, W, C]` where `None` indicates an unconstrained
            dimension) and `"type"` (e.g. `tf.float32`).
        required_config_keys: A list containing the required configuration entries.
        default_config: A dictionary of potential default configuration values.
    """
    dataset_names = set(['training', 'validation', 'test'])
    required_baseconfig = ['batch_size', 'learning_rate']
    _default_config = {'eval_batch_size': 1, 'pred_batch_size': 1}

    @abstractmethod
    def _model(self, mode, **config) -> tf.keras.Model:
        """Implements the graph of the model.

        This method is called three times: for training, evaluation and prediction (see
        the `mode` argument) and can return different tensors depending on the mode.
        It is a good practice to support both NCHW (channels first) and NHWC (channels
        last) data formats using a dedicated configuration entry.

        Arguments:
            inputs: A dictionary of input features, where the keys are their names
                (e.g. `"image"`) and the values of type `tf.Tensor`. Same keys as in the
                datasets given during the object instantiation.
            mode: An attribute of the `Mode` class, either `Mode.TRAIN`, `Mode.EVAL` or
                `Mode.PRED`.
            config: A configuration dictionary, given during the object instantiantion.

        Returns:
            A dictionary of outputs, where the keys are their names (e.g. `"logits"`) and
            the values are the corresponding `tf.Tensor`.
        """
        raise NotImplementedError

    def __init__(self, mode, data={}, n_gpus=1, data_shape=None, **config):
        self.datasets = data
        self.data_shape = data_shape
        self.n_gpus = n_gpus
        self.name = self.__class__.__name__.lower()  # get child name
        self.trainable = getattr(self, 'trainable', True)

        # Update config
        self.config = dict_update(self._default_config,
                                  getattr(self, 'default_config', {}))
        self.config = dict_update(self.config, config)

        required = self.required_baseconfig + getattr(self, 'required_config_keys', [])
        for r in required:
            assert r in self.config, 'Required configuration entry: \'{}\''.format(r)
        assert set(self.datasets) <= self.dataset_names, \
            'Unknown dataset name: {}'.format(set(self.datasets) - self.dataset_names)

        self.model = self._model(mode=mode, **self.config)

    def train(self, iterations, validation_interval=100, output_dir=None, profile=False,
              save_interval=None, checkpoint_path=None, keep_checkpoints=1):
        assert self.trainable, 'Model is not trainable.'
        assert 'training' in self.datasets, 'Training dataset is required.'

        logging.info('Start training')

        # print(model.summary())
        # tf.keras.utils.plot_model(model, show_shapes=True)

        # def mapFunc(row):
        #     print("asdsa")
        #     print(row)
        #     return (row["image"], row["keypoint_map"])

        # model.fit(self.datasets["training"].map(mapFunc))

        log_dir = "/tmp/aaa/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)

        self.model.fit(self.datasets["training"].map(
            lambda data: (data["image"], data["keypoint_map"])).batch(100).take(3),
            callbacks=[tensorboard_callback])

        # Old tensorflow 1 code:
        # for i in range(iterations):
        #     loss, summaries, _ = self.sess.run(
        #         [self.loss, self.summaries, self.trainer],
        #         feed_dict={self.handle: self.dataset_handles['training']},
        #         options=options, run_metadata=run_metadata)

        #     if save_interval and checkpoint_path and (i + 1) % save_interval == 0:
        #         self.save(checkpoint_path)
        #     if 'validation' in self.datasets and i % validation_interval == 0:
        #         metrics = self.evaluate('validation', mute=True)
        #         logging.info(
        #             'Iter {:4d}: loss {:.4f}'.format(i, loss) +
        #             ''.join([', {} {:.4f}'.format(m, metrics[m]) for m in metrics]))

        #         if output_dir is not None:
        #             train_writer.add_summary(summaries, i)
        #             metrics_summaries = tf.Summary(value=[
        #                 tf.Summary.Value(tag=m, simple_value=v)
        #                 for m, v in metrics.items()])
        #             train_writer.add_summary(metrics_summaries, i)

        #             if profile and i != 0:
        #                 fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        #                 chrome_trace = fetched_timeline.generate_chrome_trace_format()
        #                 with open(osp.join(output_dir,
        #                                    'profile_{}.json'.format(i)), 'w') as f:
        #                     f.write(chrome_trace)
        logging.info('Training finished')

    def predict(self, data, keys='pred', batch=False):
        assert set(data.keys()) >= set(self.input_spec.keys())
        if isinstance(keys, str):
            if keys == '*':
                op = self.pred_out  # just gather all outputs
            else:
                op = self.pred_out[keys]
        else:
            op = {k: self.pred_out[k] for k in keys}
        if not batch:  # add batch dimension
            data = {d: [v] for d, v in data.items()}
        feed = {self.pred_in[i]: data[i] for i in self.input_spec}
        pred = self.sess.run(op, feed_dict=feed)
        if not batch:  # remove batch dimension
            if isinstance(pred, dict):
                pred = {p: v[0] for p, v in pred.items()}
            else:
                pred = pred[0]
        return pred

    def evaluate(self, dataset, max_iterations=None, mute=False):
        assert dataset in self.datasets
        self.sess.run(self.dataset_iterators[dataset].initializer)

        if not mute:
            logging.info('Starting evaluation of dataset \'{}\''.format(dataset))
            if max_iterations:
                pbar = tqdm(total=max_iterations, ascii=True)
        i = 0
        metrics = []
        while True:
            try:
                metrics.append(self.sess.run(self.metrics,
                               feed_dict={self.handle: self.dataset_handles[dataset]}))
            except tf.errors.OutOfRangeError:
                break
            if max_iterations:
                i += 1
                if not mute:
                    pbar.update(1)
                if i == max_iterations:
                    break
        if not mute:
            logging.info('Finished evaluation')
            if max_iterations:
                pbar.close()

        # List of dicts to dict of lists
        metrics = dict(zip(metrics[0], zip(*[m.values() for m in metrics])))
        metrics = {m: np.nanmean(metrics[m], axis=0) for m in metrics}
        return metrics

    def load(self, path):
        logging.info('Saving model')
        # self.model.save(path)
        self.model.load_weights(path / "weights")

    def save(self, path):
        logging.info('Saving model')
        # self.model.save(path)
        self.model.save_weights(path / "weights")
