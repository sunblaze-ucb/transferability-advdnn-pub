"""Utility functions and classes for handling image datasets."""

import os.path as osp
import os
import numpy as np
import tensorflow as tf


def process_image(img, scale, isotropic, crop, mean, rescale, need_rescale):
    """Crops, scales, and normalizes the given image.
    scale : The image wil be first scaled to this size.
            If isotropic is true, the smaller side is rescaled to this,
            preserving the aspect ratio.
    crop  : After scaling, a central crop of this size is taken.
    mean  : Subtracted from the image
    rescale: Rescale pixel value to [x, y]
    """
    if need_rescale:
        # Rescale
        if isotropic:
            img_shape = tf.to_float(tf.shape(img)[:2])
            min_length = tf.minimum(img_shape[0], img_shape[1])
            new_shape = tf.to_int32((scale / min_length) * img_shape)
        else:
            new_shape = tf.pack([scale, scale])

        img = tf.image.resize_images(img, (new_shape[0], new_shape[1]))
        # Center crop
        # Use the slice workaround until crop_to_bounding_box supports deferred tensor shapes
        # See: https://github.com/tensorflow/tensorflow/issues/521
        offset = (new_shape - crop) / 2
        img = tf.slice(img, begin=tf.pack(
            [offset[0], offset[1], 0]), size=tf.pack([crop, crop, -1]))
    else:
        img = tf.image.resize_images(img, crop, crop)
    # Mean subtraction
    img = tf.to_float(img)
    [l, r] = rescale
    img = img / 255.0 * (r - l) + l
    img = img - mean
    return img


class ImageProducer (object):
    """
    Loads and processes batches of images in parallel.
    """

    def __init__(
            self,
            image_paths,
            need_rescale,
            data_spec,
            num_concurrent=1,
            batch_size=None,
            labels=None):
        # The data specifications describe how to process the image
        self.data_spec = data_spec
        # A list of full image paths
        self.image_paths = image_paths
        # Need to rescale images
        self.need_rescale = need_rescale
        # An optional list of labels corresponding to each image path
        self.labels = labels
        # A boolean flag per image indicating whether its a JPEG or PNG
        self.extension_mask = self.create_extension_mask(self.image_paths)

        # Load images and save as cache
        self.setup(batch_size=batch_size)

    def setup(self, batch_size):
        # Validate the batch size
        num_images = len(self.image_paths)
        self.batch_size = min(num_images, batch_size or self.data_spec.batch_size)
        if num_images % batch_size != 0:
            raise ValueError(
                'The total number of images ({}) must be divisible by the batch size ({}).'.format(
                    num_images, batch_size))
        self.num_batches = num_images / batch_size

        self.img_cache = {}

        for i in range(num_images):
            idx = i
            is_jpeg = self.extension_mask[i]
            image_path = self.image_paths[i]
            # Load the image
            img = self.load_image(image_path, is_jpeg)
            # Process the image
            processed_img = process_image(img=img,
                                          scale=self.data_spec.scale_size,
                                          isotropic=self.data_spec.isotropic,
                                          crop=self.data_spec.crop_size,
                                          mean=self.data_spec.mean,
                                          rescale=self.data_spec.rescale,
                                          need_rescale=self.need_rescale)
            self.img_cache[i] = processed_img



    def get(batch_idx, self):
        '''
        Get a single batch of images along with their indices. If a set of labels were provided,
        the corresponding labels are returned instead of the indices.
        '''

        indices = [batch_idx * self.batch_size + idx for idx in range(self.batch_size)]
        images = [self.img_cache[idx] for idx in indices]
        labels = [self.labels[idx] for idx in indices]
        names = [osp.basename(osp.normpath(self.image_paths[idx]))
                 for idx in indices]
        return (indices, labels, names, images)

    def batches(self):
        '''Yield a batch until no more images are left.'''
        for batch_idx in xrange(self.num_batches):
            yield self.get(idx)

    def load_image(self, image_path, is_jpeg):
        # Read the file
        file_data = tf.read_file(image_path)
        # Decode the image data
        img = tf.cond(
            is_jpeg, lambda: tf.image.decode_jpeg(
                file_data, channels=self.data_spec.channels), lambda: tf.image.decode_png(
                file_data, channels=self.data_spec.channels))
        if self.data_spec.expects_bgr:
            # Convert from RGB channel ordering to BGR
            # This matches, for instance, how OpenCV orders the channels.
            img = tf.reverse(img, [False, False, True])
        return img

    @staticmethod
    def create_extension_mask(paths):

        def is_jpeg(path):
            extension = osp.splitext(path)[-1].lower()
            if extension in ('.jpg', '.jpeg'):
                return True
            if extension != '.png':
                raise ValueError(
                    'Unsupported image format: {}'.format(extension))
            return False

        return [is_jpeg(p) for p in paths]

    @staticmethod
    def is_image(image_name):
        extension = osp.splitext(image_name)[-1].lower()
        if extension in ('.jpg', '.jpeg', '.png'):
            return True
        return False

    def __len__(self):
        return len(self.image_paths)


class ImageNetProducer(ImageProducer):

    @staticmethod
    def get_truth_labels(file_list):
        val_file_path = 'data/ilsvrc12/val.txt'
        label_finder = {}
        with open(val_file_path) as val_file:
            for line in val_file:
                (key, val) = line.split()
                label_finder[key[:23]] = int(val)

        def get_truth_label(file_name):
            file_name = file_name[:23]
            if file_name in label_finder:
                return label_finder[file_name]
            else:
                return -1

        return [get_truth_label(image_file_name)
                for image_file_name in file_list]

    @staticmethod
    def get_human_label(label_id):
        human_file_path = 'data/ilsvrc12/imagenet-classes.txt'
        descriptions = [line.strip() for line in open(human_file_path)]
        return descriptions[label_id]

    def __init__(
            self,
            file_list,
            data_path,
            num_images,
            data_spec,
            need_rescale=True,
            batch_size=None):
        # Read in the ground truth labels for the validation set
        # The get_ilsvrc_aux.sh in Caffe's data/ilsvrc12 folder can fetch a
        # copy of val.txt
        if file_list is None:
            file_list = [image_name for image_name in os.listdir(
                data_path) if ImageNetProducer.is_image(image_name)]
        else:
            file_list = [line.rstrip('\n') for line in open(file_list)]

        if len(file_list) > num_images:
            file_list = file_list[:num_images]

        image_paths = [osp.join(data_path, p) for p in file_list]
        # The corresponding ground truth labels
        labels = ImageNetProducer.get_truth_labels(file_list)
        # Initialize base
        super(
            ImageNetProducer,
            self).__init__(
            image_paths=image_paths,
            need_rescale=need_rescale,
            data_spec=data_spec,
            labels=labels,
            batch_size=batch_size)
