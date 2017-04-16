import argparse
import os
import sys
import models
import dataset
import numpy as np
import tensorflow as tf
import scipy.misc


def load_model(name, device):
    """ Creates and returns an instance of the model given its class name.
    The created model has a single placeholder node for feeding images.
    """
    # Find the model class from its name
    all_models = models.get_models()
    lut = {model.__name__: model for model in all_models}
    net_class = lut[name]

    # Create a placeholder for the input image
    spec = models.get_data_spec(name)
    data_node = tf.placeholder(
        tf.float32,
        shape=(
            None,
            spec.crop_size,
            spec.crop_size,
            spec.channels))

    # Construct and return the model
    return net_class({'data': data_node}, device=device)


def get_model(sesh, model_name, device):
    if model_name == 'Inception':
        net = None
        spec = models.get_data_spec(model_name)
        input_node = tf.placeholder(
            tf.float32,
            shape=(
                None,
                spec.crop_size,
                spec.crop_size,
                spec.channels))
        probs = models.get_inception(input_node)
    else:
        net = load_model(model_name, device)
        input_node = net.inputs['data']
        probs = net.get_output()

    print 'Loading parameters...', model_name
    if model_name == "Inception":
        models.inception_load_parameters(sesh)
    else:
        net.load(data_path=models.get_model_path(model_name), session=sesh)
    print 'Parameters loaded'
    return input_node, probs


def calc_gradients(sesh, image_producer, input_node, probs, data_spec):
    grad_ys_node = tf.placeholder(tf.float32, shape=(None, 1000))
    loss = tf.reduce_sum(-tf.log(tf.mul(grad_ys_node, probs)))
    var_grad = tf.gradients(loss, input_node)

    total = len(image_producer)
    gradient_record = np.zeros(
        shape=(
            total,
            data_spec.crop_size,
            data_spec.crop_size,
            data_spec.channels),
        dtype=float)

    coordinator = tf.train.Coordinator()
    threads = image_producer.start(session=sesh, coordinator=coordinator)
    image_producer.startover(session=sesh)
    for (indices, label, name, images) in image_producer.batches(sesh):
        one_hot = np.zeros(shape=(len(labels), 1000), dtype=float)
        one_hot[np.arange(len(labels)), labels] = 1
        new_images = images
        for i in range(len(indices)):
            new_images[i] = np.clip(
                new_images[i] + data_spec.mean,
                data_spec.rescale[0],
                data_spec.rescale[1])

        gradients = sesh.run(
            var_grad,
            feed_dict={
                input_node: new_images,
                grad_ys_node: one_hot})
        for i in range(len(indices)):
            gradients[0][i] /= np.sqrt(np.reduce_mean(
                np.square(gradients[0][i].flatten())))
            gradient_record[indices[i]] = gradients[0][i]
    image_producer.close_queue(session=sesh)
    coordinator.request_stop()
    coordinator.join(threads, stop_grace_period_secs=2)

    return gradient_record


def save_file(sesh, image_producer, tmp_dir, noise):
    coordinator = tf.train.Coordinator()
    threads = image_producer.start(session=sesh, coordinator=coordinator)
    image_producer.startover(session=sesh)

    for (indices, label, name, images) in image_producer.batches(sesh):
        for index in range(len(indices)):
            attack_img = np.clip(images[index] + noise[indices[index]] +
                                 data_spec.mean, data_spec.rescale[0], data_spec.rescale[1])
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

    image_producer.close_queue(session=sesh)
    coordinator.request_stop()
    coordinator.join(threads, stop_grace_period_secs=2)


def evaluate(sesh, image_producer, input_node, probs, data_spec):
    coordinator = tf.train.Coordinator()
    threads = image_producer.start(session=sesh, coordinator=coordinator)
    image_producer.startover(sesh)

    correct_top1 = 0
    correct_top5 = 0
    count = 0
    total = len(image_producer)
    label_record = np.zeros(total, dtype=int)
    # Interactive with mini-batches
    for (indices, labels, names, images) in image_producer.batches(sesh):

        probs_output = sesh.run(probs, feed_dict={input_node: images})
        for index in range(len(probs_output)):
            top1 = np.argsort(probs_output[index])[-1:]
            top5 = np.argsort(probs_output[index])[-5:]
            label_record[indices[index]] = top1
            label = labels[index]
            print indices[index], names[index], label, top1[0]
            if label in top1:
                correct_top1 += 1
            if label in top5:
                correct_top5 += 1
            count += 1
        cur_top1_accuracy = float(correct_top1) * 100 / count
        cur_top5_accuracy = float(correct_top5) * 100 / count
        print('{:>6}/{:<6} {:>6.2f}% {:>6.2f}%'.format(count,
                                                       total, cur_top1_accuracy, cur_top5_accuracy))

    # Stop the worker threads
    image_producer.close_queue(sesh)
    coordinator.request_stop()
    coordinator.join(threads, stop_grace_period_secs=2)

    print('Top 1 Accuracy: {}'.format(float(correct_top1) / total))
    print('Top 5 Accuracy: {}'.format(float(correct_top5) / total))

    return label_record


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Evaluate model on some dataset.')
    parser.add_argument(
        '-i',
        '--input_dir',
        type=str,
        required=True,
        help='Directory of dataset.')
    parser.add_argument(
        '-t',
        '--tmp_dir',
        type=str,
        required=True,
        help='Directory of dataset.')
    parser.add_argument(
        '--model1',
        type=str,
        required=True,
        choices=[
            'Inception',
            'ResNet50',
            'ResNet101',
            'ResNet152',
            'VGG16',
            'AlexNet',
            'GoogleNet'],
        help='Models to be evaluated.')
    parser.add_argument(
        '--model2',
        type=str,
        required=True,
        choices=[
            'Inception',
            'ResNet50',
            'ResNet101',
            'ResNet152',
            'VGG16',
            'AlexNet',
            'GoogleNet'],
        help='Models to be evaluated.')
    parser.add_argument(
        '--file_list',
        type=str,
        required=True,
        help='Evaluate a specific list of file in dataset.')
    parser.add_argument('--device1', type=str, default=None)
    parser.add_argument('--device2', type=str, default=None)

    args = parser.parse_args()

    data_spec1 = models.get_data_spec(model_name=args.model1)
    image_producer_o = dataset.ImageNetProducer(
        file_list=args.file_list,
        data_path=args.input_dir,
        num_images=args.num_images,
        need_rescale=True,
        data_spec=data_spec1)
    image_producer1 = dataset.ImageNetProducer(
        file_list=args.file_list,
        data_path=args.tmp_dir,
        num_images=args.num_images,
        need_rescale=False,
        data_spec=data_spec1)

    data_spec2 = models.get_data_spec(model_name=args.model2)
    image_producer2 = dataset.ImageNetProducer(
        file_list=args.file_list,
        data_path=args.tmp_dir,
        num_images=args.num_images,
        need_rescale=False,
        data_spec=data_spec2)

    need_swap = False
    if args.model2 == 'Inception':
        need_swap = True
        args.model1, args.model2 = args.model2, args.model1
        args.device1, args.device2 = args.device2, args.device1

    sesh = tf.Session()
    input_node1, probs_output1 = get_model(sesh, args.model1, args.device1)
    with tf.variable_scope('model2'):
        input_node2, probs_output2 = get_model(sesh, args.model2, args.device2)

    if need_swap:
        args.model1, args.model2 = args.model2, args.model1
        args.device1, args.device2 = args.device2, args.device1
        input_node1, input_node2 = input_node2, input_node1
        probs_output1, probs_output2 = probs_output2, probs_output1

    gradients = calc_gradients(
        sesh,
        image_producer_o,
        input_node1,
        probs_output1,
        data_spec1)

    true_label = image_producer1.get_truth_labels(file_list)
    total = len(image_producer1)
    upper_bound = 10
    while True do:
        save_file(
            sesh,
            image_producer_o,
            args.tmp_dir,
            gradients *
            upper_bound)
        predict1 = evaluate(
            sesh,
            image_producer1,
            input_node1,
            probs_output1,
            data_spec1)
        predict2 = evaluate(
            sesh,
            image_producer2,
            input_node2,
            probs_output2,
            data_spec2)
        ok = True
        for i in range(total):
            if true_label[i] == predict1[i] or true_label[i] == predict2[i]:
                ok = False
        if ok:
            break
        upper_bound *= 10

    left_bound = np.zeros(total, dtype=float)
    right_bound = np.zeros(total, dtype=float)
    for i in range(total):
        right_bound[i] = upper_bound

    for cur_iter in range(15):
        middle = (left_bound + right_bound) * 0.5
        save_file(sesh, image_producer_o, args.tmp_dir, gradients * middle)
        predict1 = evaluate(
            sesh,
            image_producer1,
            input_node1,
            probs_output1,
            data_spec1)
        predict2 = evaluate(
            sesh,
            image_producer2,
            input_node2,
            probs_output2,
            data_spec2)

        for i in range(total):
            if true_label[i] == predict1[i] or true_label[i] == predict2[i]:
                left_bound[i] = middle[i]
            else:
                right_bound[i] = middle[i]
    print 'Mean distance', np.mean(right_bound)


if __name__ == '__main__':
    main()
