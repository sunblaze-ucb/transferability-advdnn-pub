import argparse
import os
import sys
import shutil
import models
import dataset
import numpy as np
import tensorflow as tf
import scipy.misc


def calc_gradients(
        sesh,
        image_producer,
        input_node,
        probs,
        data_spec,
        use_sign,
        targets):
    grad_ys_node = tf.placeholder(tf.int32, shape=(None))
    one_hot_grad = tf.one_hot(grad_ys_node, 1000)
    grad_multi = tf.mul(one_hot_grad, probs)
    total_prob = tf.reduce_sum(grad_multi, [1])
    if targets is not None:
        loss = tf.reduce_sum(tf.log(total_prob + 1e-6))
    else:
        loss = tf.reduce_sum(tf.log(1 - total_prob + 1e-6))
    var_grad = tf.gradients(loss, input_node)

    total = len(image_producer)
    gradient_record = np.zeros(
        shape=(
            total,
            data_spec.crop_size,
            data_spec.crop_size,
            data_spec.channels),
        dtype=float)
    true_label_record = np.zeros(total, dtype=int)

    image_producer.startover()
    for (indices, labels, names, images) in image_producer.batches(sesh):
        for i in range(len(indices)):
            true_label_record[indices[i]] = labels[i]
        if targets is not None:
            labels = [targets[e] for e in names]
        val_total_prob, val_prob = sesh.run(
            (total_prob, probs), feed_dict={
                input_node: images, grad_ys_node: labels})
        gradients = sesh.run(
            var_grad,
            feed_dict={
                input_node: images,
                grad_ys_node: labels})
        if use_sign:
            gradients = np.sign(gradients)

        for i in range(len(indices)):
            l2 = np.sqrt(np.mean(np.square(gradients[0][i].flatten())))
            gradients[0][i] /= l2
            gradient_record[indices[i]] = gradients[0][i]

    return true_label_record, gradient_record


def save_file(sesh, image_producer, tmp_dir, noise, data_spec):
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    total = len(image_producer)
    diff = np.zeros(
        shape=(
            total,
            data_spec.crop_size,
            data_spec.crop_size,
            data_spec.channels))

    image_producer.startover()
    for (indices, label, names, images) in image_producer.batches(sesh):
        for index in range(len(indices)):
            attack_img = np.clip(
                images[index] +
                noise[
                    indices[index]] +
                data_spec.mean,
                data_spec.rescale[0],
                data_spec.rescale[1])
            diff[indices[index]] = attack_img - data_spec.mean - images[index]
            if data_spec.expects_bgr:
                for i in range(data_spec.crop_size):
                    for j in range(data_spec.crop_size):
                        b, g, r = attack_img[i][j]
                        attack_img[i][j] = [r, g, b]
            im = scipy.misc.toimage(
                arr=attack_img,
                cmin=data_spec.rescale[0],
                cmax=data_spec.rescale[1])
            new_name, ext = os.path.splitext(names[index])
            new_name += '.png'
            im.save(os.path.join(tmp_dir, new_name))
            print 'Saved', os.path.join(tmp_dir, new_name)
    return np.sqrt(np.mean(np.square(diff)))


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Use Fast Gradient or Fast Gradient Sign method \
                        to generate adversarial examples.')
    parser.add_argument('-i', '--input_dir', type=str, required=True,
        help='Directory of dataset.')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
        help='Directory of output log file.')
    parser.add_argument('--model', type=str, required=True,
        choices=['GoogleNet'],
        help='Models to be evaluated.')
    parser.add_argument('--file_list', type=str, required=True,
        help='Evaluate a specific list of file in dataset.')
    parser.add_argument('--num_iter', type=int, default=100)
    parser.add_argument('--sign', dest='use_sign', action='store_true')
    parser.add_argument('--target', type=str, default=None,
                        help='Target list of file in dataset.')
    parser.add_argument('--noise_file', type=str, default=None)
    parser.add_argument('-n', '--not_crop', dest='need_rescale',
                        action='store_false')
    parser.set_defaults(num_images=sys.maxsize)
    parser.set_defaults(use_sign=False)
    parser.set_defaults(need_rescale=True)

    args = parser.parse_args()

    targets = None
    if args.target is not None:
        targets = {}
        with open(args.target, 'r') as f:
            for line in f:
                key, value = line.strip().split()
                targets[key] = int(value)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    data_spec = models.get_data_spec(model_name=args.model)

    sesh = tf.Session()
    if args.noise_file is None:
        input_node = tf.placeholder(
            tf.float32,
            shape=(
                None,
                data_spec.crop_size,
                data_spec.crop_size,
                data_spec.channels))
        probs_output, variable_list = models.get_model(
            sesh, input_node, args.model)

    image_producer = dataset.ImageNetProducer(
        file_list=args.file_list,
        data_path=args.input_dir,
        num_images=args.num_images,
        need_rescale=args.need_rescale,
        data_spec=data_spec,
        batch_size=1)

    print 'Start compute gradients'
    if args.noise_file is None:
        true_label, gradients = calc_gradients(
            sesh, image_producer, input_node, probs_output, data_spec, args.use_sign, targets)
    else:
        gradients = np.load(args.noise_file)
        if args.use_sign:
            gradients = np.sign(gradients)
    print 'End compute gradients'
    gradients /= np.sqrt(np.mean(np.square(gradients)))
    print 'RMSE of gradients', np.sqrt(np.mean(np.square(gradients)))

    for magnitude in range(1, args.num_iter + 1):
        distance = save_file(
            sesh, image_producer, os.path.join(
                args.output_dir, str(magnitude)), gradients * magnitude / 255.0 * (
                data_spec.rescale[1] - data_spec.rescale[0]), data_spec)

if __name__ == '__main__':
    main()
