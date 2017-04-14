import argparse
import skimage
import pickle
import random
import os
import sys
import models
import dataset
import numpy as np
import tensorflow as tf
import scipy.misc
from numpy import linalg as LA

def dot(a, b):
    a = a.flatten()
    b = b.flatten()
    return np.inner(a, b)

def cos(a, b):
    a = a.flatten()
    b = b.flatten()
    return np.inner(a, b) / LA.norm(a) / LA.norm(b)

def calc_gradients(model_name, image_producer, args):
    """Compute the gradients for the given network and images."""
    spec = models.get_data_spec(model_name)
    sesh = tf.Session()
    input_node = tf.placeholder(tf.float32,
                                shape=(None, spec.crop_size, spec.crop_size, spec.channels))
    input_label = tf.placeholder(tf.int32, (None))
    probs, variable_set = models.get_model(sesh, input_node, model_name)
    true_label_prob = tf.reduce_mean(tf.reduce_sum(probs * tf.one_hot(input_label, 1000), [1]))
    loss = tf.log(1 - true_label_prob + 1e-6)
    var_grad = tf.gradients(loss, input_node)

    coordinator = tf.train.Coordinator()
    # Start the image processing workers
    threads = image_producer.start(session=sesh, coordinator=coordinator)
    image_producer.startover(sesh)

    record_all = np.zeros(shape=(len(image_producer), args.num_tries, args.num_tries))
    record_all_original = np.zeros(shape=(len(image_producer), args.num_tries))
    record_angle = []
    record_angle_with_original = []

    # Interactive with mini-batches
    for (indices, labels, names, images) in image_producer.batches(sesh):
        gradient_record = np.zeros(shape=(args.num_tries, spec.crop_size, spec.crop_size, spec.channels), dtype=float)
        print 'Calculate original dir'
        original_dir = sesh.run(var_grad, feed_dict={input_node: images, input_label: labels})[0][0]
        original_dir /= LA.norm(original_dir.flatten())
        print 'Done'

        for iterx in range(args.num_tries):
            print '!ITER', iterx
            new_images = np.zeros(shape=(1, spec.crop_size, spec.crop_size, spec.channels))
            for i in range(new_images.shape[0]):
                noise = np.random.randn(spec.crop_size, spec.crop_size, spec.channels)
                noise -= dot(noise, original_dir) * original_dir
                noise = noise / np.sqrt(np.mean(np.square(noise))) * args.std_dev / 255.0 * (spec.rescale[1] - spec.rescale[0])
                new_images[i] = images[i] + noise
                # print new_images[i] - images[i]
                print 'RMSD', names[i], np.sqrt(np.mean(np.square(new_images[i] - images[i]))), dot(noise, original_dir)
            print 'OK!'
            gradients = sesh.run(var_grad, feed_dict={input_node: new_images, input_label: labels})
            gradient_record[iterx] = gradients[0][0]

        angles = []
        angle_with_original = []
        for it1 in range(args.num_tries):
            tmp = cos(gradient_record[it1], original_dir)
            angle_with_original.append(tmp)
            record_all_original[indices[0]][it1] = tmp
            for it2 in range(it1 + 1, args.num_tries):
                tmp = cos(gradient_record[it1], gradient_record[it2])
                angles.append(tmp)
                record_all[indices[0]][it1][it2] = tmp

        mean_angle = np.mean(angles)
        print mean_angle
        print np.mean(angle_with_original)
        record_angle.append(mean_angle)
        record_angle_with_original.append(np.mean(angle_with_original))

    # Stop the worker threads
    image_producer.close_queue(sesh)
    coordinator.request_stop()
    coordinator.join(threads, stop_grace_period_secs=2)

    print np.mean(record_angle)
    print np.mean(record_all_original)
    dict_rec = {}
    dict_rec['data'] = record_all
    dict_rec['with_ori'] = record_all_original
    dict_rec['mean_data'] = np.mean(record_angle)
    dict_rec['mean_ori'] = np.mean(record_all_original)
    dict_rec['var'] = np.var(record_angle)
    with open(os.path.join(args.output_dir, '_'.join([model_name, str(args.std_dev)]) + '.pickle'), 'wb') as handle:
        pickle.dump(dict_rec, handle)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate model on some dataset.')
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='Directory of dataset.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Directory of output file.')
    parser.add_argument('--model', type=str, required=True,
                        choices=['Inception', 'ResNet50', 'ResNet101', 'ResNet152', 'VGG16', 'GoogleNet'],
                        help='Models to be evaluated.')
    parser.add_argument('--num_images', type=int, default=sys.maxint,
                        help='Max number of images to be evaluated.')
    parser.add_argument('--file_list', type=str, default=None,
                        help='Evaluate a specific list of file in dataset.')
    parser.add_argument('--num_tries', type=int, default=5, help='Number of time to add noise.')
    parser.add_argument('--std_dev', type=float, default=1, help='Std dev of noise, out of 255')

    args = parser.parse_args()
    print args

    data_spec = models.get_data_spec(model_name=args.model)
    image_producer = dataset.ImageNetProducer(file_list=args.file_list, data_path=args.input_dir,
                                              num_images=args.num_images, data_spec=data_spec,
                                              batch_size=1)

    calc_gradients(args.model, image_producer, args)


if __name__ == '__main__':
    main()
