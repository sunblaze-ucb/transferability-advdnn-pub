import argparse
import os
import sys
import models
import dataset
import numpy as np
import tensorflow as tf
from multiprocessing import Queue


def load_model(name, input_node):
    """ Creates and returns an instance of the model given its class name.
    The created model has a single placeholder node for feeding images.
    """
    # Find the model class from its name
    all_models = models.get_models()
    net_class = [model for model in all_models if model.__name__ == name][0]

    # Construct and return the model
    return net_class({'data': input_node})


def convert_noise(noise, size0, size1):
    if noise.shape[1] == size0 and nosie.shape[2] == size1:
        return noise


def calc_gradients(
        model_name,
        image_producer,
        output_file_dir=None,
        max_iter=1,
        noise_file=None,
        learning_rate=0.001 * 255,
        use_round=False,
        weight_loss2=1,
        data_spec=None,
        batch_size=1):
    """Compute the gradients for the given network and images."""

    spec = models.get_data_spec(model_name)

    modifier = tf.Variable(
        np.zeros(
            (batch_size,
             spec.crop_size,
             spec.crop_size,
             spec.channels),
            dtype=np.float32))
    input_image = tf.placeholder(
        tf.float32, (None, spec.crop_size, spec.crop_size, spec.channels))
    input_label = tf.placeholder(tf.int32, (None))

    true_image = tf.minimum(tf.maximum(modifier +
                                       input_image, -
                                       spec.mean +
                                       spec.rescale[0]), -
                            spec.mean +
                            spec.rescale[1])
    # true_image = modifier + input_image

    # l2dist = tf.sqrt(tf.reduce_mean(tf.square(true_image - input_image), [1, 2, 3]))
    diff = true_image - input_image
    loss2 = tf.reduce_mean(tf.square(true_image - input_image))

    if model_name == 'Inception':
        net = None
        # Get the output of network (class probabilities)
        probs = models.get_inception(true_image)
    else:
        net = load_model(model_name, true_image)
        # Get the output of the network (class probabilities)
        probs = net.get_output()

    temp_set = set(tf.global_variables())

    weight_loss1 = 1
    true_label_prob = tf.reduce_mean(
        tf.reduce_sum(
            probs *
            tf.one_hot(
                input_label,
                1000),
            [1]))
    loss1 = -tf.log(1 - true_label_prob)
    # In paper it's mentioned set lambda to 0, i.e. not restricting d(x, x*)
    # to let Adam optimizer explore the surrounding space
    loss = weight_loss1 * loss1  # + weight_loss2 * loss2 * 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss, var_list=[modifier])
    init = tf.variables_initializer([modifier])

    print 'Weight of loss1:', weight_loss1
    print 'Weight of loss2:', weight_loss2

    # The total number of images
    total = len(image_producer)

    # Adding noise
    noise = np.zeros(
        shape=(
            total,
            spec.crop_size,
            spec.crop_size,
            spec.channels),
        dtype=float)
    if noise_file is not None:
        noise = convert_noise(
            np.load(noise_file),
            spec.crop_size,
            spec.crop_size)

    gradient_record = np.zeros(
        shape=(
            total,
            spec.crop_size,
            spec.crop_size,
            spec.channels),
        dtype=float)
    rec_iters = []
    rec_names = []
    rec_dist = []

    with tf.Session() as sesh:

        # initiallize all uninitialized varibales
        varibale_list = (set(tf.global_variables()) -
                         temp_set).union(set([modifier]))
        sesh.run(tf.variables_initializer(varibale_list))

        # Load the parameters to model
        print 'Loading parameters...', model_name
        if model_name == "Inception":
            models.inception_load_parameters(
                sesh, set(tf.all_variables()) - varibale_list)
        else:
            net.load(data_path=models.get_model_path(model_name), session=sesh)
        print 'Parameters loaded'
        coordinator = tf.train.Coordinator()

        # Start the image processing workers
        threads = image_producer.start(session=sesh, coordinator=coordinator)
        image_producer.startover(sesh)

        # Interactive with mini-batches
        for (indices, labels, names, images) in image_producer.batches(sesh):
            sesh.run(tf.variables_initializer(varibale_list))
            feed_dict = {input_image: images, input_label: labels}
            var_loss, true_prob, var_loss1, var_loss2 = sesh.run(
                (loss, true_label_prob, loss1, loss2), feed_dict=feed_dict)
            print 'Start!', var_loss, true_prob, var_loss1, var_loss2
            min_loss = var_loss
            last_min = -1

            # record numer of iteration
            tot_iter = 0
            for cur_iter in range(max_iter):
                tot_iter += 1

                # print sesh.run(modifier)
                sesh.run(train, feed_dict=feed_dict)
                var_loss, true_prob, var_loss1, var_loss2 = sesh.run(
                    (loss, true_label_prob, loss1, loss2), feed_dict=feed_dict)
                print cur_iter + 1, var_loss, true_prob, var_loss1, var_loss2, np.sqrt(var_loss)

                break_condition = False
                if var_loss < min_loss * 0.99:
                    min_loss = var_loss
                    last_min = cur_iter
                # if last_min + 10 < cur_iter:
                if true_prob < 1e-4:
                    break_condition = True

                if cur_iter + 1 == max_iter or break_condition:
                    var_diff, var_probs = sesh.run(
                        (diff, probs), feed_dict=feed_dict)
                    var_diff = np.sqrt(np.mean(np.square(var_diff), (1, 2, 3)))
                    correct_top_1 = 0
                    for i in range(len(indices)):
                        top1 = var_probs[i].argmax()
                        print i, names[i], var_diff[i], labels[i], var_probs[i][labels[i]], top1, var_probs[i][top1]
                        if labels[i] == top1:
                            correct_top_1 += 1
                        rec_iters.append(tot_iter)
                        rec_names.append(names[i])
                        rec_dist.append(var_diff[i])

                    break

            noise_diff = sesh.run(diff, feed_dict=feed_dict)
            if use_round:
                noise_diff = np.sign(noise_diff) * \
                    np.ceil(np.absolute(noise_diff))
            for i in range(len(indices)):
                gradient_record[indices[i]] = noise_diff[i]

        # Close queue
        image_producer.close_queue(session=sesh)
        # Stop the worker threads
        coordinator.request_stop()
        coordinator.join(threads, stop_grace_period_secs=2)

    if output_file_dir is not None:
        np.save(os.path.join(output_file_dir, model_name), gradient_record)
    with open(os.path.join(output_file_dir, model_name + '_log.txt'), 'w') as f:
        f.write('Average numer of iterations: %.2f\n' % np.mean(rec_iters))
        f.write('Average L2 distance %.2f\n' % np.mean(rec_dist))
        for i in range(len(rec_names)):
            f.write('%s %d %.2f\n' % (rec_names[i], rec_iters[i], rec_dist[i]))


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
        '-o',
        '--output_dir',
        type=str,
        default=None,
        help='Directory of output noise file.')
    parser.add_argument(
        '--model',
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
        '--num_images',
        type=int,
        default=sys.maxsize,
        help='Max number of images to be evaluated.')
    parser.add_argument('--file_list', type=str, default=None,
                        help='Evaluate a specific list of file in dataset.')
    parser.add_argument(
        '--num_iter',
        type=int,
        default=1000,
        help='Number of iterations to generate attack.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001 * 255,
        help='Learning rate of each iteration.')
    parser.add_argument(
        '--use_round',
        dest='use_round',
        action='store_true',
        help='Round to integer.')
    parser.add_argument(
        '--weight_loss2',
        type=float,
        default=1,
        help='Weight of distance penalty.')
    parser.add_argument(
        '--noise_file',
        type=str,
        default=None,
        help='Directory of added noise.')
    parser.set_defaults(use_round=False)

    args = parser.parse_args()

    data_spec = models.get_data_spec(model_name=args.model)
    args.learning_rate *= (data_spec.rescale[1] - data_spec.rescale[0]) / 255.0

    print args
    image_producer = dataset.ImageNetProducer(
        file_list=args.file_list,
        data_path=args.input_dir,
        num_images=args.num_images,
        data_spec=data_spec,
        batch_size=1)

    calc_gradients(
        args.model,
        image_producer,
        args.output_dir,
        args.num_iter,
        args.noise_file,
        args.learning_rate,
        args.use_round,
        args.weight_loss2,
        data_spec,
        batch_size=1)


if __name__ == '__main__':
    main()
