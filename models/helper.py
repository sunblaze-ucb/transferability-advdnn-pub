import sys
import os.path as osp
import numpy as np
import tensorflow as tf
import os

# Add the kaffe module to the import path
sys.path.append(osp.realpath(osp.join(osp.dirname(__file__), './')))

from inception.inception import inception_model
from googlenet import GoogleNet
from vgg import VGG16
from alexnet import AlexNet
from caffenet import CaffeNet
from nin import NiN
from resnet import ResNet50, ResNet101, ResNet152


def auto_str(cls):
    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items())
        )
    cls.__str__ = __str__
    return cls


@auto_str
class DataSpec(object):
    '''Input data specifications for an ImageNet model.'''

    def __init__(self,
                 batch_size,
                 scale_size,
                 crop_size,
                 isotropic,
                 channels=3,
                 rescale=[0.0, 255.0],
                 mean=np.array([104., 117., 124.]),
                 bgr=True):
        # The recommended batch size for this model
        self.batch_size = batch_size
        # The image should be scaled to this size first during preprocessing
        self.scale_size = scale_size
        # Whether the model expects the rescaling to be isotropic
        self.isotropic = isotropic
        # A square crop of this dimension is expected by this model
        self.crop_size = crop_size
        # The number of channels in the input image expected by this model
        self.channels = channels
        # The mean to be subtracted from each image. By default, the per-channel ImageNet mean.
        # The values below are ordered BGR, as many Caffe models are trained in this order.
        # Some of the earlier models (like AlexNet) used a spatial three-channeled mean.
        # However, using just the per-channel mean values instead doesn't
        # affect things too much.
        self.mean = mean
        # Whether this model expects images to be in BGR order
        self.expects_bgr = bgr
        self.rescale = rescale


def alexnet_spec(batch_size=500):
    """Parameters used by AlexNet and its variants."""
    return DataSpec(
        batch_size=batch_size,
        scale_size=256,
        crop_size=227,
        isotropic=False)


def ensemble_spec():
    return DataSpec(
        batch_size=1,
        scale_size=224,
        crop_size=224,
        isotropic=False)


def inception_spec(batch_size=25, crop_size=299, bgr=False):
    """Parameters used by Inception and its variants."""
    return DataSpec(batch_size=batch_size,
                    scale_size=crop_size,
                    crop_size=crop_size,
                    isotropic=False,
                    bgr=bgr,
                    rescale=[-1.0,
                             1.0],
                    mean=np.array([0.,
                                   0.,
                                   0.]))


def std_spec(batch_size, isotropic=True):
    """Parameters commonly used by "post-AlexNet" architectures."""
    return DataSpec(
        batch_size=batch_size,
        scale_size=256,
        crop_size=224,
        isotropic=False)


# Collection of sample auto-generated models
str2Model = {
    "AlexNet": AlexNet,
    "CaffeNet": CaffeNet,
    "GoogleNet": GoogleNet,
    "NiN": NiN,
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    "VGG16": VGG16}
MODELS = (
    AlexNet,
    CaffeNet,
    GoogleNet,
    NiN,
    ResNet50,
    ResNet101,
    ResNet152,
    VGG16)

# The corresponding data specifications for the sample models
# These specifications are based on how the models were trained.
# The recommended batch size is based on a Titan X (12GB).
MODEL_DATA_SPECS = {
    "AlexNet": alexnet_spec(),
    "CaffeNet": alexnet_spec(),
    "GoogleNet": std_spec(batch_size=25, isotropic=False),
    "ResNet50": std_spec(batch_size=25),
    "ResNet101": std_spec(batch_size=25),
    "ResNet152": std_spec(batch_size=25),
    "NiN": std_spec(batch_size=500),
    "VGG16": std_spec(batch_size=25),
    "Inception": inception_spec(batch_size=25, crop_size=299),
    "Inception2": inception_spec(batch_size=25, crop_size=224, bgr=False),
    "ensemble": ensemble_spec()
}

MODEL_PATHES = {
    "AlexNet": "models/AlexNet.npy",
    "CaffeNet": "models/CaffeNet.npy",
    "GoogleNet": "models/GoogleNet.npy",
    "ResNet50": "models/ResNet50.npy",
    "ResNet101": "models/ResNet101.npy",
    "ResNet152": "models/ResNet152.npy",
    "VGG16": "models/VGG16.npy"
}

CKPT_PATHES = {
    "GoogleNet": "checkpoints/GoogleNet/GoogleNet.ckpt",
    "ResNet50": "checkpoints/ResNet50/ResNet50.ckpt",
    "ResNet101": "checkpoints/ResNet101/ResNet101.ckpt",
    "ResNet152": "checkpoints/ResNet152/ResNet152.ckpt",
    "VGG16": "checkpoints/VGG16/VGG16.ckpt",
    "Inception": "checkpoints/Inception/Inception.ckpt",
    "Inception2": "checkpoints/Inception2/Inception.ckpt",
}


def get_models():
    """Returns a tuple of sample models."""
    return MODELS


def get_model_path(model_name):
    return MODEL_PATHES[model_name]


def get_data_spec(model_name):
    """Returns the data specifications for the given network."""
    return MODEL_DATA_SPECS[model_name]


def get_model(sess, input_node, model_name, device=None):
    print 'Getting model def', model_name
    start_variable_set = set(tf.all_variables())
    if model_name == 'Inception':
        end_node = get_inception(input_node)
    elif model_name == 'Inception2':
        # swap_rgb = tf.reverse(input_node, [False, False, True])
        rescaled_input_node = tf.image.resize_bilinear(input_node, [299, 299])
        end_node = get_inception(rescaled_input_node)
    else:
        all_models = MODELS
        net_class = [
            model for model in all_models if model.__name__ == model_name][0]
        net = net_class({'data': input_node})
        end_node = net.get_output()
    end_variable_set = set(tf.all_variables())
    variable_set = end_variable_set - start_variable_set
    print 'Loading prarameters'
    saver = tf.train.Saver(variable_set)
    ckpt_dir = CKPT_PATHES[model_name]
    print 'Checkpoint dir', ckpt_dir
    saver.restore(sess, ckpt_dir)
    print 'Loaded prarameters'
    return end_node, variable_set


def get_model2(sess, input_node, model_name, device=None):
    print 'Getting model def', model_name
    start_variable_set = set(tf.all_variables())
    if model_name == 'Inception':
        end_node, end_node2 = get_inception2(input_node)
    elif model_name == 'Inception2':
        rescaled_input_node = tf.image.resize_bilinear(input_node, [299, 299])
        end_node, end_node2 = get_inception2(rescaled_input_node)
    else:
        all_models = MODELS
        net_class = [
            model for model in all_models if model.__name__ == model_name][0]
        net = net_class({'data': input_node})
        end_node = net.get_output()
        if model_name == 'VGG16':
            end_node2 = net.layers['fc8']
        elif model_name == 'GoogleNet':
            end_node2 = net.layers['loss3_classifier']
        else:
            end_node2 = net.layers['fc1000']
    end_variable_set = set(tf.all_variables())
    variable_set = end_variable_set - start_variable_set
    print 'Loading prarameters'
    saver = tf.train.Saver(variable_set)
    ckpt_dir = CKPT_PATHES[model_name]
    print 'Checkpoint dir', ckpt_dir
    saver.restore(sess, ckpt_dir)
    print 'Loaded prarameters'
    return end_node, end_node2


def get_inception(images):
    return inception_model.inference(images=images, num_classes=1000 + 1)


def get_inception2(images):
    return inception_model.inference2(images=images, num_classes=1000 + 1)


def inception_load_parameters(sess, var_list=None):
    saver = tf.train.Saver(var_list)
    ckpt_dir = 'models/inception/checkpoint'
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print ckpt.model_checkpoint_path
        saver.restore(sess, ckpt.model_checkpoint_path)

        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/imagenet_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Succesfully loaded model from %s at step=%s.' %
              (ckpt.model_checkpoint_path, global_step))
    else:
        print('No checkpoint file found')
        return


def save_all_as_checkponits():
    # Now this function only save ckpt file for GoogleNet,
    # TODO: Save ckpt for all 9 networks
    for model_name in [
        "AlexNet",
        "CaffeNet",
        "GoogleNet",
        "NiN",
        "ResNet50",
            "VGG16"]:
        if not tf.train.checkpoint_exists(CKPT_PATHES[model_name]):
            print "Checkpoint for " + model_name + " has not been created yet, creating checkpoint..."
            spec = get_data_spec(model_name)
            input_node = tf.placeholder(
                tf.float32,
                shape=(
                    None,
                    spec.crop_size,
                    spec.crop_size,
                    spec.channels))
            net = str2Model[model_name]({'data': input_node})
            with tf.Session() as sesh:
                net.load(get_model_path(model_name), sesh)
                saver = tf.train.Saver()
                save_path = saver.save(sesh, CKPT_PATHES[model_name])
                print(
                    model_name +
                    " Model checkpoint saved in file: %s" %
                    save_path)
