import logging
import yaml
import os
import argparse
import numpy as np
from contextlib import contextmanager
from json import dumps as pprint
from pathlib import Path
import cv2

from superpoint.datasets import get_dataset
from superpoint.models import get_model
from superpoint.utils.stdout_capturing import capture_outputs
from superpoint.settings import EXPER_PATH

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
import tensorflow as tf  # noqa: E402

from models.base_model import Mode


def train(config, n_iter, output_dir, pretrained_dir=None):

    set_seed(config.get('seed', int.from_bytes(os.urandom(4), byteorder='big')))

    dataset = get_dataset(config['data']['name'])(**config['data'])
    print(dataset.get_training_set().element_spec)
    model = get_model(config['model']['name'])(Mode.TRAIN,
        dataset.get_tf_datasets(), **config['model'])

    if pretrained_dir is not None:
        model.load(pretrained_dir)
    try:
        model.train(n_iter, output_dir=str(output_dir),
                    validation_interval=config.get('validation_interval', 100),
                    save_interval=config.get('save_interval', None),
                    checkpoint_path=None,
                    keep_checkpoints=config.get('keep_checkpoints', 1))
    except KeyboardInterrupt:
        logging.info('Got Keyboard Interrupt, saving model and closing.')

    model.save(output_dir / "saved_model")


def evaluate(config, output_dir, n_iter=None):
    set_seed(config.get('seed', int.from_bytes(os.urandom(4), byteorder='big')))

    dataset = get_dataset(config['data']['name'])(**config['data'])
    model = get_model(config['model']['name'])(
        dataset.get_tf_datasets(), **config['model'])

    model.load(output_dir)
    results = model.evaluate(config.get('eval_set', 'test'), max_iterations=n_iter)
    return results


def predict(config, output_dir, input_image):
    model = get_model(config['model']['name'])(Mode.PRED, **config['model'])

    model.load(output_dir / "saved_model")

    image = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
    model_input = image.reshape(1, 120, 160, 1)
    # pred = model.model(image)
    pred = model.model.predict(model_input, steps=1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.imwrite("input.png", rgb_image)
    rgb_image[:,:,2] = np.maximum(rgb_image[:,:,2], pred[0] * 255)
    cv2.imwrite("prediction.png", rgb_image)
    # tf.io.write_file("prediction.png", tf.io.encode_png(pred))
    print("fasdfsadf")


def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)


def _cli_train(config, output_dir, args):
    assert 'train_iter' in config

    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    if args.pretrained_model is not None:
        pretrained_dir = os.path.join(EXPER_PATH, args.pretrained_model)
        if not os.path.exists(pretrained_dir):
            raise ValueError("Missing pretrained model: " + pretrained_dir)
    else:
        pretrained_dir = None

    train(config, config['train_iter'], Path(output_dir), pretrained_dir)

    if args.eval:
        _cli_eval(config, output_dir, args)


def _cli_eval(config, output_dir, args):
    # Load model config from previous experiment
    with open(os.path.join(output_dir, 'config.yml'), 'r') as f:
        model_config = yaml.load(f)['model']
    model_config.update(config.get('model', {}))
    config['model'] = model_config

    results = evaluate(config, output_dir, n_iter=config.get('eval_iter'))

    # Print and export results
    logging.info('Evaluation results: \n{}'.format(
        pprint(results, indent=2, default=str)))
    with open(os.path.join(output_dir, 'eval.txt'), 'a') as f:
        f.write('Evaluation for {} dataset:\n'.format(config['data']['name']))
        for r, v in results.items():
            f.write('\t{}:\n\t\t{}\n'.format(r, v))
        f.write('\n')


def _cli_pred(config, output_dir, args):
    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    predict(config, Path(output_dir), args.input_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Training command
    p_train = subparsers.add_parser('train')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.add_argument('--eval', action='store_true')
    p_train.add_argument('--pretrained_model', type=str, default=None)
    p_train.set_defaults(func=_cli_train)

    # Evaluation command
    p_evaluate = subparsers.add_parser('evaluate')
    p_evaluate.add_argument('config', type=str)
    p_evaluate.add_argument('exper_name', type=str)
    p_evaluate.set_defaults(func=_cli_eval)

    # Inference command
    p_predict = subparsers.add_parser('predict')
    p_predict.add_argument('config', type=str)
    p_predict.add_argument('exper_name', type=str)
    p_predict.add_argument('input_image', type=str)
    p_predict.set_defaults(func=_cli_pred)

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with capture_outputs(os.path.join(output_dir, 'log')):
        logging.info('Running command {}'.format(args.command.upper()))
        args.func(config, output_dir, args)
