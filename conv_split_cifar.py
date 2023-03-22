"""
Training script for split CIFAR 100 experiment.
"""
from __future__ import print_function
import collections
import argparse
import os
import sys
import math
import time

import datetime
import numpy as np
import tensorflow as tf
from copy import deepcopy
from six.moves import cPickle as pickle
from tqdm import tqdm

from utils.data_utils import construct_split_cifar
from utils.utils import get_sample_weights, sample_from_dataset, update_episodic_memory, concatenate_datasets, samples_for_each_class, sample_from_dataset_icarl, compute_fgt, load_task_specific_data
from utils.utils import average_acc_stats_across_runs, average_fgt_stats_across_runs, update_reservior, der_update_reservior, average_ltr_across_runs
from utils.vis_utils import plot_acc_multiple_runs, plot_histogram, snapshot_experiment_meta_data, snapshot_experiment_eval, snapshot_task_labels
from model import Model
import os
from scipy.spatial import distance
from utils.sv_knn_buffer import SVKNNBuffer
# 指定使用0,1,2三块卡
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

###############################################################
################ Some definitions #############################
### These will be edited by the command line options ##########
###############################################################

## Training Options
distance_metric = 'euclidean'
NUM_RUNS = 5           # Number of experiments to average over
TRAIN_ITERS = 2000      # Number of training iterations per task
BATCH_SIZE = 16
LEARNING_RATE = 0.1
RANDOM_SEED = 1234
VALID_OPTIMS = ['SGD', 'MOMENTUM', 'ADAM']
OPTIM = 'SGD'
OPT_MOMENTUM = 0.9
OPT_POWER = 0.9
VALID_ARCHS = ['CNN', 'RESNET-S', 'RESNET-B', 'VGG']
ARCH = 'RESNET-S'

## Model options
MODELS = ['VAN', 'PI', 'EWC', 'MAS', 'RWALK', 'M-EWC', 'S-GEM', 'A-GEM', 'FTR_EXT', 'PNN', 'ER', 'MEGA', 'MEGAD', 'CDD'] #List of valid models
IMP_METHOD = 'MEGA'
SYNAP_STGTH = 75000
FISHER_EMA_DECAY = 0.9          # Exponential moving average decay factor for Fisher computation (online Fisher)
FISHER_UPDATE_AFTER = 50        # Number of training iterations for which the F_{\theta}^t is computed (see Eq. 10 in RWalk paper)

SAMPLES_PER_CLASS = 13
K_cache = 20
IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 3
TOTAL_CLASSES = 100          # Total number of classes in the dataset
VISUALIZE_IMPORTANCE_MEASURE = False
MEASURE_CONVERGENCE_AFTER = 0.9
EPS_MEM_BATCH_SIZE = 256
DEBUG_EPISODIC_MEMORY = False
K_FOR_CROSS_VAL = 3
TIME_MY_METHOD = False
COUNT_VIOLATONS = False
MEASURE_PERF_ON_EPS_MEMORY = False

## Logging, saving and testing options
LOG_DIR = './split_cifar_results'
RESNET18_CIFAR10_CHECKPOINT = './resnet-18-pretrained-cifar10/model.ckpt-19999'
## Evaluation options

## Task split
NUM_TASKS = 20
MULTI_TASK = False


# Define function to load/ store training weights. We will use ImageNet initialization later on
def save(saver, sess, logdir, step):
   '''Save weights.

   Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
   '''
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)

   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
        saver: TensorFlow Saver object.
        sess: TensorFlow session.
        ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Script for split cifar experiment.")
    parser.add_argument("--cross-validate-mode", action="store_true",
            help="If option is chosen then snapshoting after each batch is disabled")
    parser.add_argument("--online-cross-val", action="store_true",
            help="If option is chosen then enable the online cross validation of the learning rate")
    parser.add_argument("--train-single-epoch", action="store_true",
            help="If option is chosen then train for single epoch")
    parser.add_argument("--eval-single-head", action="store_true",
            help="If option is chosen then evaluate on a single head setting.")
    parser.add_argument("--arch", type=str, default=ARCH,
                        help="Network Architecture for the experiment.\
                                \n \nSupported values: %s"%(VALID_ARCHS))
    parser.add_argument("--num-runs", type=int, default=NUM_RUNS,
                       help="Total runs/ experiments over which accuracy is averaged.")
    parser.add_argument("--train-iters", type=int, default=TRAIN_ITERS,
                       help="Number of training iterations for each task.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help="Mini-batch size for each task.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                       help="Random Seed.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                       help="Starting Learning rate for each task.")
    parser.add_argument("--optim", type=str, default=OPTIM,
                        help="Optimizer for the experiment. \
                                \n \nSupported values: %s"%(VALID_OPTIMS))
    parser.add_argument("--imp-method", type=str, default=IMP_METHOD,
                       help="Model to be used for LLL. \
                        \n \nSupported values: %s"%(MODELS))
    parser.add_argument("--synap-stgth", type=float, default=SYNAP_STGTH,
                       help="Synaptic strength for the regularization.")
    parser.add_argument("--fisher-ema-decay", type=float, default=FISHER_EMA_DECAY,
                       help="Exponential moving average decay for Fisher calculation at each step.")
    parser.add_argument("--fisher-update-after", type=int, default=FISHER_UPDATE_AFTER,
                       help="Number of training iterations after which the Fisher will be updated.")
    parser.add_argument("--mem-size", type=int, default=SAMPLES_PER_CLASS,
                       help="Total size of episodic memory.")
    parser.add_argument("--eps-mem-batch", type=int, default=EPS_MEM_BATCH_SIZE,
                       help="Number of samples per class from previous tasks.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                       help="Directory where the plots and model accuracies will be stored.")
    parser.add_argument("--m1-value", type=float, default=0.4)
    parser.add_argument("--m2-value", type=float, default=0.01)
    parser.add_argument("--s-value", type=int, default=24)
    return parser.parse_args()

MFR_matrix = np.zeros([17, 17, 5, 160])
TSNE_feature = []
TSNE_label = []
MFR_list = []

def train_task_sequence(model, sess, datasets, args):
    """
    Train and evaluate LLL system such that we only see a example once
    Args:
    Returns:
        dict    A dictionary containing mean and stds for the experiment
    """
    # List to store accuracy for each run
    global task_pro, store_images, store_labels, store_features, num_ctd, avg_centroids, dist_cls, confident, centroids, total_num, temp_images, temp_labels, temp_features, examples_seen_so_far, sv_knn_buffer, tasks_ed, norm_episodic_features
    runs = []
    task_labels_dataset = []

    if model.imp_method == 'A-GEM' or model.imp_method == 'ER' or model.imp_method == 'DER' or model.imp_method == 'MEGA' or model.imp_method == 'MEGAD' or model.imp_method == 'CDD' or model.imp_method == 'SCR':
        use_episodic_memory = True
    else:
        use_episodic_memory = False

    batch_size = args.batch_size
    # Loop over number of runs to average over
    for runid in range(args.num_runs):
        print('\t\tRun %d:'%(runid))
        # Initialize the random seeds
        np.random.seed(args.random_seed+runid)

        # Get the task labels from the total number of tasks and full label space
        task_labels = []
        classes_per_task = TOTAL_CLASSES// NUM_TASKS
        total_classes = classes_per_task * model.num_tasks
        if args.online_cross_val:
            label_array = np.arange(total_classes)
        else:
            class_label_offset = K_FOR_CROSS_VAL * classes_per_task
            label_array = np.arange(class_label_offset, total_classes+class_label_offset)

        np.random.shuffle(label_array)
        for tt in range(model.num_tasks):
            tt_offset = tt*classes_per_task
            task_labels.append(list(label_array[tt_offset:tt_offset+classes_per_task]))
            print('Task: {}, Labels:{}'.format(tt, task_labels[tt]))

        # Store the task labels
        task_labels_dataset.append(task_labels)

        # Set episodic memory size
        episodic_mem_size = args.mem_size * total_classes

        # Initialize all the variables in the model
        sess.run(tf.global_variables_initializer())

        # Run the init ops
        model.init_updates(sess)

        # List to store accuracies for a run
        evals = []

        # List to store the classes that we have so far - used at test time
        test_labels = []
        if model.imp_method == 'ER':
            sv_knn_buffer = SVKNNBuffer(args, episodic_mem_size, [32, 32, 3], 100, args.eps_mem_batch)

        if use_episodic_memory:
            # Reserve a space for episodic memory
            episodic_images = np.zeros([episodic_mem_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
            episodic_labels = np.zeros([episodic_mem_size, TOTAL_CLASSES])
            episodic_features = np.zeros([episodic_mem_size, 160])
            norm_episodic_features = np.zeros([episodic_mem_size, 160])
            episodic_filled_counter = 0
            nd_logit_mask = np.zeros([model.num_tasks, TOTAL_CLASSES])
            count_cls = np.zeros(TOTAL_CLASSES, dtype=np.int32)
            episodic_filled_counter = 0
            examples_seen_so_far = 0
            task_pro = np.zeros([17, 5, 13])
            num_ctd = np.zeros([17, 5, 1])
            store_images = np.zeros([17, 5, 13, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
            store_labels = np.zeros([17, 5, 13, TOTAL_CLASSES])
            store_features = np.zeros([17, 5, 13, 160])
            avg_centroids = []
            dist_cls = []
            confident = np.zeros(episodic_mem_size, dtype=np.int32)
            centroids = []
            total_num = []
            temp_images = []
            temp_labels = []
            temp_features = []
            tasks_ed = []
            for i in range(TOTAL_CLASSES):
                centroids.append([])
                total_num.append([])
                temp_images.append([])
                temp_labels.append([])
                temp_features.append([])

        # Mask for softmax 
        logit_mask = np.zeros(TOTAL_CLASSES)

        if COUNT_VIOLATONS:
            violation_count = np.zeros(model.num_tasks)
            vc = 0

        # Training loop for all the tasks
        for task in range(len(task_labels)):
            print('\t%s\tTask %d:'%(model.imp_method,task))
            # If not the first task then restore weights from previous task
            if(task > 0 and model.imp_method != 'PNN'):
                model.restore(sess)

            if model.imp_method == 'PNN':
                pnn_train_phase = np.array(np.zeros(model.num_tasks), dtype=np.bool)
                pnn_train_phase[task] = True
                pnn_logit_mask = np.zeros([model.num_tasks, TOTAL_CLASSES])

            # If not in the cross validation mode then concatenate the train and validation sets
            task_tr_images, task_tr_labels = load_task_specific_data(datasets[0]['train'], task_labels[task])
            task_val_images, task_val_labels = load_task_specific_data(datasets[0]['validation'], task_labels[task])
            task_train_images, task_train_labels = concatenate_datasets(task_tr_images, task_tr_labels, task_val_images, task_val_labels)

            # If multi_task is set then train using all the datasets of all the tasks
            if MULTI_TASK:
                if task == 0:
                    for t_ in range(1, len(task_labels)):
                        task_tr_images, task_tr_labels = load_task_specific_data(datasets[0]['train'], task_labels[t_])
                        task_train_images = np.concatenate((task_train_images, task_tr_images), axis=0)
                        task_train_labels = np.concatenate((task_train_labels, task_tr_labels), axis=0)

                else:
                    # Skip training for this task
                    continue


            print('Received {} images, {} labels at task {}'.format(task_train_images.shape[0], task_train_labels.shape[0], task))
            print('Unique labels in the task: {}'.format(np.unique(np.nonzero(task_train_labels)[1])))
            # Test for the tasks that we've seen so far
            test_labels += task_labels[task]

            # Assign equal weights to all the examples
            task_sample_weights = np.ones([task_train_labels.shape[0]], dtype=np.float32)

            num_train_examples = task_train_images.shape[0]

            logit_mask[:] = 0
            # Train a task observing sequence of data
            if args.train_single_epoch:
                # Ceiling operation
                num_iters = (num_train_examples + batch_size - 1) // batch_size
                if args.cross_validate_mode:
                    logit_mask[task_labels[task]] = 1.0
            else:
                num_iters = args.train_iters
                # Set the mask only once before starting the training for the task
                logit_mask[task_labels[task]] = 1.0

            if MULTI_TASK:
                logit_mask[:] = 1.0

            # Randomly suffle the training examples
            perm = np.arange(num_train_examples)
            np.random.shuffle(perm)
            train_x = task_train_images[perm]
            train_y = task_train_labels[perm]
            task_sample_weights = task_sample_weights[perm]

            # Array to store accuracies when training for task T
            ftask = []

            # Number of iterations after which convergence is checked
            convergence_iters = int(num_iters * MEASURE_CONVERGENCE_AFTER)

            # Training loop for task T
            for iters in range(num_iters):
                # if iters == 20:
                #     start = datetime.datetime.now()
                last_cur_loss = 0.
                last_ref_loss = 0.

                if args.train_single_epoch:
                    offset = iters * batch_size
                    if (offset+batch_size <= num_train_examples):
                        residual = batch_size
                    else:
                        residual = num_train_examples - offset

                    if model.imp_method == 'PNN':
                        feed_dict = {model.x: train_x[offset:offset+residual], model.y_[task]: train_y[offset:offset+residual],
                                model.training_iters: num_iters, model.train_step: iters, model.keep_prob: 0.5}
                        train_phase_dict = {m_t: i_t for (m_t, i_t) in zip(model.train_phase, pnn_train_phase)}
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, pnn_logit_mask)}
                        feed_dict.update(train_phase_dict)
                        feed_dict.update(logit_mask_dict)
                    else:
                        feed_dict = {model.x: train_x[offset:offset+residual], model.y_: train_y[offset:offset+residual],
                                model.sample_weights: task_sample_weights[offset:offset+residual],
                                model.training_iters: num_iters, model.train_step: iters, model.keep_prob: 0.5,
                                model.train_phase: True}
                else:
                    offset = (iters * batch_size) % (num_train_examples - batch_size)
                    if model.imp_method == 'PNN':
                        feed_dict = {model.x: train_x[offset:offset+batch_size], model.y_[task]: train_y[offset:offset+batch_size],
                                model.training_iters: num_iters, model.train_step: iters, model.keep_prob: 0.5}
                        train_phase_dict = {m_t: i_t for (m_t, i_t) in zip(model.train_phase, pnn_train_phase)}
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, pnn_logit_mask)}
                        feed_dict.update(train_phase_dict)
                        feed_dict.update(logit_mask_dict)
                    else:
                        feed_dict = {model.x: train_x[offset:offset+batch_size], model.y_: train_y[offset:offset+batch_size],
                                model.sample_weights: task_sample_weights[offset:offset+batch_size],
                                model.training_iters: num_iters, model.train_step: iters, model.keep_prob: 0.5,
                                model.train_phase: True}

                if model.imp_method == 'VAN':
                    feed_dict[model.output_mask] = logit_mask
                    _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'PNN':
                    _, loss = sess.run([model.train[task], model.unweighted_entropy[task]], feed_dict=feed_dict)

                elif model.imp_method == 'FTR_EXT':
                    feed_dict[model.output_mask] = logit_mask
                    if task == 0:
                        _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)
                    else:
                        _, loss = sess.run([model.train_classifier, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'EWC' or model.imp_method == 'M-EWC':
                    feed_dict[model.output_mask] = logit_mask
                    # If first iteration of the first task then set the initial value of the running fisher
                    if task == 0 and iters == 0:
                        sess.run([model.set_initial_running_fisher], feed_dict=feed_dict)
                    # Update fisher after every few iterations
                    if (iters + 1) % model.fisher_update_after == 0:
                        sess.run(model.set_running_fisher)
                        sess.run(model.reset_tmp_fisher)

                    if (iters >= convergence_iters) and (model.imp_method == 'M-EWC'):
                        _, _, _, _, loss = sess.run([model.weights_old_ops_grouped, model.set_tmp_fisher, model.train, model.update_small_omega,
                                              model.reg_loss], feed_dict=feed_dict)
                    else:
                        _, _, loss = sess.run([model.set_tmp_fisher, model.train, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'PI':
                    feed_dict[model.output_mask] = logit_mask
                    _, _, _, loss = sess.run([model.weights_old_ops_grouped, model.train, model.update_small_omega,
                                              model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'MAS':
                    feed_dict[model.output_mask] = logit_mask
                    _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)

                elif model.imp_method == 'A-GEM':
                    if task == 0:
                        nd_logit_mask[:] = 0
                        nd_logit_mask[task][task_labels[task]] = 1.0
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, nd_logit_mask)}
                        feed_dict.update(logit_mask_dict)
                        feed_dict[model.mem_batch_size] = batch_size
                        # Normal application of gradients
                        _, loss = sess.run([model.train_first_task, model.agem_loss], feed_dict=feed_dict)
                    else:
                        ## Compute and store the reference gradients on the previous tasks
                        # Set the mask for all the previous tasks so far
                        nd_logit_mask[:] = 0
                        for tt in range(task):
                            nd_logit_mask[tt][task_labels[tt]] = 1.0

                        if episodic_filled_counter <= args.eps_mem_batch:
                            mem_sample_mask = np.arange(episodic_filled_counter)
                        else:
                            # Sample a random subset from episodic memory buffer
                            mem_sample_mask = np.random.choice(episodic_filled_counter, args.eps_mem_batch, replace=False) # Sample without replacement so that we don't sample an example more than once

                        # Store the reference gradient
                        ref_feed_dict = {model.x: episodic_images[mem_sample_mask], model.y_: episodic_labels[mem_sample_mask],
                                model.keep_prob: 1.0, model.train_phase: True}
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, nd_logit_mask)}
                        ref_feed_dict.update(logit_mask_dict)
                        ref_feed_dict[model.mem_batch_size] = float(len(mem_sample_mask))
                        sess.run([model.store_ref_grads], feed_dict=ref_feed_dict)

                        # Compute the gradient for current task and project if need be
                        nd_logit_mask[:] = 0
                        nd_logit_mask[task][task_labels[task]] = 1.0
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, nd_logit_mask)}
                        feed_dict.update(logit_mask_dict)
                        feed_dict[model.mem_batch_size] = batch_size
                        if COUNT_VIOLATONS:
                            vc, _, loss = sess.run([model.violation_count, model.train_subseq_tasks, model.agem_loss], feed_dict=feed_dict)
                        else:
                            _, loss = sess.run([model.train_subseq_tasks, model.agem_loss], feed_dict=feed_dict)

                    # Put the batch in the ring buffer
                    for er_x, er_y_ in zip(train_x[offset:offset+residual], train_y[offset:offset+residual]):
                        cls = np.unique(np.nonzero(er_y_))[-1]
                        # Write the example at the location pointed by count_cls[cls]
                        cls_to_index_map = np.where(np.array(task_labels[task]) == cls)[0][0]
                        with_in_task_offset = args.mem_size  * cls_to_index_map
                        mem_index = count_cls[cls] + with_in_task_offset + episodic_filled_counter
                        episodic_images[mem_index] = er_x
                        episodic_labels[mem_index] = er_y_
                        count_cls[cls] = (count_cls[cls] + 1) % args.mem_size

                elif model.imp_method == 'MEGA':
                    if task == 0:
                        nd_logit_mask[:] = 0
                        nd_logit_mask[task][task_labels[task]] = 1.0
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, nd_logit_mask)}
                        feed_dict.update(logit_mask_dict)
                        feed_dict[model.mem_batch_size] = batch_size
                        # Normal application of gradients
                        _, loss = sess.run([model.train_first_task, model.agem_loss], feed_dict=feed_dict)
                    else:
                        ## Compute and store the reference gradients on the previous tasks
                        # Set the mask for all the previous tasks so far
                        nd_logit_mask[:] = 0
                        for tt in range(task):
                            nd_logit_mask[tt][task_labels[tt]] = 1.0

                        if episodic_filled_counter <= args.eps_mem_batch:
                            mem_sample_mask = np.arange(episodic_filled_counter)
                        else:
                            # Sample a random subset from episodic memory buffer
                            mem_sample_mask = np.random.choice(episodic_filled_counter, args.eps_mem_batch, replace=False) # Sample without replacement so that we don't sample an example more than once
                        # Store the reference gradient
                        ref_feed_dict = {model.x: episodic_images[mem_sample_mask], model.y_: episodic_labels[mem_sample_mask],
                                model.keep_prob: 1.0, model.train_phase: True}
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, nd_logit_mask)}
                        ref_feed_dict.update(logit_mask_dict)
                        ref_feed_dict[model.mem_batch_size] = float(len(mem_sample_mask))
                        sess.run([model.store_ref_grads, model.store_ref_loss], feed_dict=ref_feed_dict)

                        # Compute the gradient for current task and project if need be
                        nd_logit_mask[:] = 0
                        nd_logit_mask[task][task_labels[task]] = 1.0
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, nd_logit_mask)}
                        feed_dict.update(logit_mask_dict)
                        feed_dict[model.mem_batch_size] = batch_size

                        # Training
                        _, loss, theta = sess.run([model.train_subseq_tasks, model.agem_loss, model.theta], feed_dict=feed_dict)

                    # Put the batch in the ring buffer
                    for er_x, er_y_ in zip(train_x[offset:offset+residual], train_y[offset:offset+residual]):
                        cls = np.unique(np.nonzero(er_y_))[-1]
                        # Write the example at the location pointed by count_cls[cls]
                        cls_to_index_map = np.where(np.array(task_labels[task]) == cls)[0][0]
                        with_in_task_offset = args.mem_size  * cls_to_index_map
                        mem_index = count_cls[cls] + with_in_task_offset + episodic_filled_counter
                        episodic_images[mem_index] = er_x
                        episodic_labels[mem_index] = er_y_
                        count_cls[cls] = (count_cls[cls] + 1) % args.mem_size


                elif model.imp_method == 'RACL':
                    if task == 0:
                        nd_logit_mask[:] = 0
                        nd_logit_mask[task][task_labels[task]] = 1.0
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, nd_logit_mask)}
                        feed_dict.update(logit_mask_dict)
                        feed_dict[model.mem_batch_size] = batch_size
                        # Normal application of gradients
                        _, loss = sess.run([model.train_first_task, model.agem_loss], feed_dict=feed_dict)
                    else:
                        ## Compute and store the reference gradients on the previous tasks
                        # Set the mask for all the previous tasks so far
                        nd_logit_mask[:] = 0
                        for tt in range(task):
                            nd_logit_mask[tt][task_labels[tt]] = 1.0

                        if episodic_filled_counter <= args.eps_mem_batch:
                            mem_sample_mask = np.arange(episodic_filled_counter)
                        else:
                            # Sample a random subset from episodic memory buffer
                            mem_sample_mask = np.random.choice(episodic_filled_counter, args.eps_mem_batch, replace=False) # Sample without replacement so that we don't sample an example more than once
                        # Store the reference gradient
                        ref_feed_dict = {model.x: episodic_images[mem_sample_mask], model.y_: episodic_labels[mem_sample_mask],
                                model.keep_prob: 1.0, model.train_phase: True}
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, nd_logit_mask)}
                        ref_feed_dict.update(logit_mask_dict)
                        ref_feed_dict[model.mem_batch_size] = float(len(mem_sample_mask))
                        _, _, ref_loss = sess.run([model.store_ref_grads, model.store_ref_loss, model.agem_loss], feed_dict=ref_feed_dict)

                        # Compute the gradient for current task and project if need be
                        nd_logit_mask[:] = 0
                        nd_logit_mask[task][task_labels[task]] = 1.0
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, nd_logit_mask)}
                        feed_dict.update(logit_mask_dict)
                        feed_dict[model.mem_batch_size] = batch_size

                        # Training
                        if iters == 0:
                            feed_dict[model.last_cur_loss] = sess.run(model.agem_loss, feed_dict=feed_dict)
                            feed_dict[model.last_ref_loss] = ref_loss
                        else:
                            feed_dict[model.last_cur_loss] = last_cur_loss
                            feed_dict[model.last_ref_loss] = last_ref_loss
                        _, loss, ab = sess.run([model.train_subseq_tasks, model.agem_loss, model.ab], feed_dict=feed_dict)
                        # print(ab)
                        # print(last_cur_loss)
                        # print(ref_loss)
                        last_cur_loss = loss
                        last_ref_loss = ref_loss

                    # Put the batch in the ring buffer
                    for er_x, er_y_ in zip(train_x[offset:offset+residual], train_y[offset:offset+residual]):
                        cls = np.unique(np.nonzero(er_y_))[-1]
                        # Write the example at the location pointed by count_cls[cls]
                        cls_to_index_map = np.where(np.array(task_labels[task]) == cls)[0][0]
                        with_in_task_offset = args.mem_size  * cls_to_index_map
                        mem_index = count_cls[cls] + with_in_task_offset + episodic_filled_counter
                        episodic_images[mem_index] = er_x
                        episodic_labels[mem_index] = er_y_
                        count_cls[cls] = (count_cls[cls] + 1) % args.mem_size

                elif model.imp_method == 'MEGAD':
                    if task == 0:
                        nd_logit_mask[:] = 0
                        nd_logit_mask[task][task_labels[task]] = 1.0
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, nd_logit_mask)}
                        feed_dict.update(logit_mask_dict)
                        feed_dict[model.mem_batch_size] = batch_size
                        # Normal application of gradients
                        _, loss = sess.run([model.train_first_task, model.agem_loss], feed_dict=feed_dict)
                    else:
                        ## Compute and store the reference gradients on the previous tasks
                        # Set the mask for all the previous tasks so far
                        nd_logit_mask[:] = 0
                        for tt in range(task):
                            nd_logit_mask[tt][task_labels[tt]] = 1.0

                        if episodic_filled_counter <= args.eps_mem_batch:
                            mem_sample_mask = np.arange(episodic_filled_counter)
                        else:
                            # Sample a random subset from episodic memory buffer
                            mem_sample_mask = np.random.choice(episodic_filled_counter, args.eps_mem_batch,
                                                               replace=False)  # Sample without replacement so that we don't sample an example more than once
                        # Store the reference gradient
                        ref_feed_dict = {model.x: episodic_images[mem_sample_mask],
                                         model.y_: episodic_labels[mem_sample_mask],
                                         model.keep_prob: 1.0, model.train_phase: True}
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, nd_logit_mask)}
                        ref_feed_dict.update(logit_mask_dict)
                        ref_feed_dict[model.mem_batch_size] = float(len(mem_sample_mask))
                        sess.run([model.store_ref_grads, model.store_ref_loss], feed_dict=ref_feed_dict)

                        # Compute the gradient for current task and project if need be
                        nd_logit_mask[:] = 0
                        nd_logit_mask[task][task_labels[task]] = 1.0
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, nd_logit_mask)}
                        feed_dict.update(logit_mask_dict)
                        feed_dict[model.mem_batch_size] = batch_size

                        # Training
                        _, loss = sess.run([model.train_subseq_tasks, model.agem_loss], feed_dict=feed_dict)

                    # Put the batch in the ring buffer
                    for er_x, er_y_ in zip(train_x[offset:offset + residual], train_y[offset:offset + residual]):
                        cls = np.unique(np.nonzero(er_y_))[-1]
                        # Write the example at the location pointed by count_cls[cls]
                        cls_to_index_map = np.where(np.array(task_labels[task]) == cls)[0][0]
                        with_in_task_offset = args.mem_size * cls_to_index_map
                        mem_index = count_cls[cls] + with_in_task_offset + episodic_filled_counter
                        episodic_images[mem_index] = er_x
                        episodic_labels[mem_index] = er_y_
                        count_cls[cls] = (count_cls[cls] + 1) % args.mem_size

                elif model.imp_method == 'CDD':
                    if task == 0:
                        nd_logit_mask[:] = 0
                        nd_logit_mask[task][task_labels[task]] = 1.0
                        # print(nd_logit_mask[0], np.sum(nd_logit_mask))
                        # exit()
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, nd_logit_mask)}
                        feed_dict.update(logit_mask_dict)
                        feed_dict[model.mem_batch_size] = batch_size
                        _, loss = sess.run([model.train_first_task, model.agem_loss], feed_dict=feed_dict)

                    else:
                        ## Compute and store the reference gradients on the previous tasks
                        # Set the mask for all the previous tasks so far
                        nd_logit_mask[:] = 0
                        for tt in range(task):
                            nd_logit_mask[tt][task_labels[tt]] = 1.0

                        if episodic_filled_counter <= args.eps_mem_batch:
                            mem_sample_mask = np.arange(episodic_filled_counter)
                        else:
                            # Sample a random subset from episodic memory buffer
                            mem_sample_mask = np.random.choice(episodic_filled_counter, args.eps_mem_batch, replace=False)  # Sample without replacement so that we don't sample an example more than once
                        # Store the reference gradient
                        ref_feed_dict = {model.x: episodic_images[mem_sample_mask],
                                         model.y_: episodic_labels[mem_sample_mask],
                                         model.org_feat: episodic_features[mem_sample_mask],
                                         model.keep_prob: 1.0, model.train_phase: True, model.flag1: 1}
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, nd_logit_mask)}
                        ref_feed_dict.update(logit_mask_dict)
                        ref_feed_dict[model.mem_batch_size] = float(len(mem_sample_mask))
                        ref_loss, ref_grads, kl_grads = sess.run(
                            [model.agem_loss, model.ref_grads_save, model.kl_grads_save], feed_dict=ref_feed_dict)
                        # ref_loss, ref_grads = sess.run([model.agem_loss, model.ref_grads_save], feed_dict=ref_feed_dict)
                        stored_grads = ref_grads
                        stored_loss = ref_loss
                        stored_kl_grads = kl_grads
                        # Compute the gradient for current task and project if need be
                        nd_logit_mask[:] = 0
                        nd_logit_mask[task][task_labels[task]] = 1.0
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, nd_logit_mask)}
                        feed_dict.update(logit_mask_dict)
                        feed_dict[model.mem_batch_size] = batch_size
                        feed_dict[model.store_grads] = stored_grads
                        feed_dict[model.store_kl_grads] = stored_kl_grads
                        feed_dict[model.store_loss] = [stored_loss]
                        feed_dict[model.flag1] = 0
                        # Training
                        _, loss = sess.run([model.train_subseq_tasks, model.agem_loss], feed_dict=feed_dict)
                    # Put the batch in the ring buffer
                    for er_x, er_y_ in zip(train_x[offset:offset + residual], train_y[offset:offset + residual]):
                        feature_c = sess.run(model.features,
                                             feed_dict={
                                                 model.x: np.array([er_x]),
                                                 model.y_: np.array([er_y_]),
                                                 model.flag1: 0,
                                                 model.keep_prob: 1.0,
                                                 model.train_phase: False})
                        cls = np.unique(np.nonzero(er_y_))[-1]
                        cls_to_index_map = np.where(np.array(task_labels[task]) == cls)[0][0]
                        with_in_task_offset = args.mem_size * cls_to_index_map
                        mem_index = count_cls[cls] + with_in_task_offset + episodic_filled_counter
                        episodic_images[mem_index] = er_x
                        episodic_labels[mem_index] = er_y_
                        episodic_features[mem_index] = feature_c
                        count_cls[cls] = (count_cls[cls] + 1) % args.mem_size
                        distances = []
                        indices = []
                        if len(centroids[cls]) == 0:
                            centroids[cls].append(feature_c)
                            total_num[cls].append(1)
                            temp_images[cls].append([])
                            temp_labels[cls].append([])
                            temp_features[cls].append([])
                            temp_images[cls][0].append(er_x)
                            temp_labels[cls][0].append(er_y_)
                            temp_features[cls][0].append(feature_c)
                        n = len(centroids[cls])
                        for j in range(0, len(centroids[cls])):
                            # d = der_find_distance(centroid_features[io], centroids[jo], tsd_vector, distance_metric)
                            d = find_distance(feature_c, centroids[cls][j], distance_metric)
                            if d < 6:
                                distances.append(d)
                                indices.append(j)
                        if len(distances) == 0:
                            centroids[cls].append(feature_c)
                            total_num[cls].append(1)
                            temp_images[cls].append([])
                            temp_labels[cls].append([])
                            temp_features[cls].append([])
                            temp_images[cls][n].append(er_x)
                            temp_labels[cls][n].append(er_y_)
                            temp_features[cls][n].append(feature_c)
                        else:
                            min_d = np.argmin(distances)
                            centroids[cls][indices[min_d]] = np.add(np.multiply(len(temp_labels[cls][indices[min_d]]), centroids[cls][indices[min_d]]), feature_c)
                            total_num[cls][indices[min_d]] += 1
                            centroids[cls][indices[min_d]] = np.divide(centroids[cls][indices[min_d]],
                                                                       (len(temp_labels[cls][indices[min_d]]) + 1))
                            if len(temp_labels[cls][indices[min_d]]) < K_cache:
                                temp_images[cls][indices[min_d]].append(er_x)
                                temp_labels[cls][indices[min_d]].append(er_y_)
                                temp_features[cls][indices[min_d]].append(feature_c)
                            else:
                                dist = []
                                cur_feat_temp, cur_centroids = sess.run([model.features, model.avg_features],
                                                                        feed_dict={
                                                                         model.x: np.array(temp_images[cls][indices[min_d]]),
                                                                         model.flag1: 0,
                                                                         model.keep_prob: 1.0,
                                                                         model.train_phase: False})
                                centroids[cls][indices[min_d]] = cur_centroids
                                for i in range(len(temp_labels[cls][indices[min_d]])):
                                    temp_features[cls][indices[min_d]][i] = cur_feat_temp[i]
                                    dd = find_distance(temp_features[cls][indices[min_d]][i],
                                                       centroids[cls][indices[min_d]], distance_metric)
                                    dist.append(dd)
                                index_max = np.argmax(dist)
                                temp_images[cls][indices[min_d]].pop(index_max)
                                temp_labels[cls][indices[min_d]].pop(index_max)
                                temp_features[cls][indices[min_d]].pop(index_max)
                                temp_images[cls][indices[min_d]].append(er_x)
                                temp_labels[cls][indices[min_d]].append(er_y_)
                                temp_features[cls][indices[min_d]].append(feature_c)

                elif model.imp_method == 'RWALK':
                    feed_dict[model.output_mask] = logit_mask
                    # If first iteration of the first task then set the initial value of the running fisher
                    if task == 0 and iters == 0:
                        sess.run([model.set_initial_running_fisher], feed_dict=feed_dict)
                        # Store the current value of the weights
                        sess.run(model.weights_delta_old_grouped)
                    # Update fisher and importance score after every few iterations
                    if (iters + 1) % model.fisher_update_after == 0:
                        # Update the importance score using distance in riemannian manifold   
                        sess.run(model.update_big_omega_riemann)
                        # Now that the score is updated, compute the new value for running Fisher
                        sess.run(model.set_running_fisher)
                        # Store the current value of the weights
                        sess.run(model.weights_delta_old_grouped)
                        # Reset the delta_L
                        sess.run([model.reset_small_omega])

                    _, _, _, _, loss = sess.run([model.set_tmp_fisher, model.weights_old_ops_grouped,
                        model.train, model.update_small_omega, model.reg_loss], feed_dict=feed_dict)

                # if (math.isnan(loss)):
                #     print('ERROR: NaNs NaNs NaNs!!!')
                #     sys.exit(0)
            if model.imp_method == 'CDD':
                centroid_zero = [100000 for x in range(160)]
                centroids_a = []
                task_index = SAMPLES_PER_CLASS * 5
                for c, cls in enumerate(task_labels[task]):
                    num = c * SAMPLES_PER_CLASS + task_index * task
                    sum_total = 0
                    pro_centroids = []
                    for i in range(len(centroids[cls])):
                        sum_total += total_num[cls][i]
                    for j in range(len(centroids[cls])):
                        pro_centroids.append(total_num[int(cls)][j] / sum_total)
                    # print('pro_centroids',pro_centroids)
                    for i in range(SAMPLES_PER_CLASS):
                        r = np.random.rand()
                        temp = 0
                        dist = []
                        for j in range(len(centroids[cls])):
                            temp += pro_centroids[j]
                            if temp > r:
                                for m in range(len(temp_labels[cls][j])):
                                    dd = find_distance(temp_features[cls][j][m], centroids[cls][j], distance_metric)
                                    dist.append(dd)
                                min_dd = np.argmin(dist)
                                episodic_images[num + i] = temp_images[cls][j][min_dd]
                                episodic_labels[num + i] = temp_labels[cls][j][min_dd]
                                cur_feature = sess.run(model.features,
                                                     feed_dict={
                                                         model.x: np.array([temp_images[cls][j][min_dd]]),
                                                         model.flag1: 0,
                                                         model.keep_prob: 1.0,
                                                         model.train_phase: False})
                                episodic_features[num + i] = cur_feature
                                norm_episodic_features[num + i] = cur_feature / np.linalg.norm(cur_feature)
                                temp_features[cls][j][min_dd] = centroid_zero
                                confident[num + i] = j
                                break
                    a_cls = np.unique(confident[num:num+SAMPLES_PER_CLASS])
                    for i in range(len(a_cls)):
                        centroids_a.append(np.mean(norm_episodic_features[num:num+SAMPLES_PER_CLASS][confident[num:num+SAMPLES_PER_CLASS] == a_cls[i]], axis=0))
                task_dist = np.matmul(np.array(centroids_a), np.array(centroids_a).T)
                tasks_ed.append(task_dist)

            print('\t\t\t\tTraining for Task%d done!'%(task))

            if use_episodic_memory:
                episodic_filled_counter += args.mem_size * classes_per_task

            if model.imp_method == 'A-GEM':
                if COUNT_VIOLATONS:
                    violation_count[task] = vc
                    print('Task {}: Violation Count: {}'.format(task, violation_count))
                    sess.run(model.reset_violation_count, feed_dict=feed_dict)

            # Compute the inter-task updates, Fisher/ importance scores etc
            # Don't calculate the task updates for the last task
            if (task < (len(task_labels) - 1)) or MEASURE_PERF_ON_EPS_MEMORY:
                model.task_updates(sess, task, task_train_images, task_labels[task]) # TODO: For MAS, should the gradients be for current task or all the previous tasks
                print('\t\t\t\tTask updates after Task%d done!'%(task))

            if VISUALIZE_IMPORTANCE_MEASURE:
                if runid == 0:
                    for i in range(len(model.fisher_diagonal_at_minima)):
                        if i == 0:
                            flatten_fisher = np.array(model.fisher_diagonal_at_minima[i].eval()).flatten()
                        else:
                            flatten_fisher = np.concatenate((flatten_fisher,
                                np.array(model.fisher_diagonal_at_minima[i].eval()).flatten()))

                    #flatten_fisher [flatten_fisher > 0.1] = 0.1
                    if args.train_single_epoch:
                        plot_histogram(flatten_fisher, 100, '/private/home/arslanch/Dropbox/LLL_experiments/Single_Epoch/importance_vis/single_epoch/m_ewc/hist_fisher_task%s.png'%(task))
                    else:
                        plot_histogram(flatten_fisher, 100, '/private/home/arslanch/Dropbox/LLL_experiments/Single_Epoch/importance_vis/single_epoch/m_ewc/hist_fisher_task%s.png'%(task))

            if False or args.train_single_epoch and not args.cross_validate_mode:
                fbatch, flogits, fy = test_task_sequence(model, sess, datasets[0]['test'], task_labels, task)
                print('flogits[0]:-->',flogits[0].shape)

                print('Task: {}, {}, Acc: {}'.format(task, np.mean(fbatch), fbatch))
                ftask.append(fbatch)
                ftask = np.array(ftask)

                if model.imp_method == 'PNN':
                    pnn_train_phase[:] = False
                    pnn_train_phase[task] = True
                    pnn_logit_mask[:] = 0
                    pnn_logit_mask[task][task_labels[task]] = 1.0
            else:
                if MEASURE_PERF_ON_EPS_MEMORY:
                    eps_mem = {
                            'images': episodic_images,
                            'labels': episodic_labels,
                            }
                    # Measure perf on episodic memory
                    ftask = test_task_sequence(model, sess, eps_mem, task_labels, task, classes_per_task=classes_per_task)
                else:
                    # List to store accuracy for all the tasks for the current trained model
                    # ftask = test_task_sequence(model, sess, datasets[0]['test'], task_labels, task)
                    fbatch, flogits, fy = test_task_sequence(model, sess, datasets[0]['test'], task_labels, task)
                    # print('Task: {}, {}, Acc: {}'.format(task, np.mean(ftask), ftask))
                    print('Task: {},{},Acc: {}'.format(task, np.mean(fbatch), fbatch))
            # Store the accuracies computed at task T in a list
            evals.append(ftask)
            # Reset the optimizer
            model.reset_optimizer(sess)

            #-> End for loop task

        runs.append(np.array(evals))
        # End for loop runid

    runs_mean = [np.mean(run[-1]) for run in runs]
    print(runs_mean)
    print((max(runs_mean)+min(runs_mean))/2, '+-', (max(runs_mean)-min(runs_mean))/2, np.mean(runs_mean))
    # runs_mean = [np.mean(run[-1, -1]) for run in runs]
    # print(runs_mean)
    # print((max(runs_mean) + min(runs_mean)) / 2, '±', (max(runs_mean) - min(runs_mean)) / 2, np.mean(runs_mean))
    runs = np.array(runs)

    return runs, task_labels_dataset

def test_task_sequence(model, sess, test_data, test_tasks, task, classes_per_task=0):
    """
    Snapshot the current performance
    """
    # global pruned_logits
    if TIME_MY_METHOD:
        # Only compute the training time
        return np.zeros(model.num_tasks)

    final_acc = np.zeros(model.num_tasks)

    list_logit = []
    list_y = []
    if model.imp_method == 'PNN' or model.imp_method == 'A-GEM' or model.imp_method == 'ER' or model.imp_method == 'MEGA' or model.imp_method == 'MEGAD' or model.imp_method == 'CDD':
        logit_mask = np.zeros([model.num_tasks, TOTAL_CLASSES])
    else:
        logit_mask = np.zeros(TOTAL_CLASSES)

    if MEASURE_PERF_ON_EPS_MEMORY:
        for tt, labels in enumerate(test_tasks):
            # Multi-head evaluation setting
            logit_mask[:] = 0
            logit_mask[labels] = 1.0
            mem_offset = tt*SAMPLES_PER_CLASS*classes_per_task
            feed_dict = {model.x: test_data['images'][mem_offset:mem_offset+SAMPLES_PER_CLASS*classes_per_task],
                    model.y_: test_data['labels'][mem_offset:mem_offset+SAMPLES_PER_CLASS*classes_per_task], model.keep_prob: 1.0, model.train_phase: False, model.output_mask: logit_mask}
            acc = model.accuracy.eval(feed_dict = feed_dict)
            final_acc[tt] = acc
        print('aaaaaaaaaaaaaaaa')
        return final_acc

    for tt, labels in enumerate(test_tasks):
        # list_logit = None
        # list_y = None

        if not MULTI_TASK:
            if tt > task:
                # print(tt)
                # print(task)
                # print(final_acc)
                # print('aaaaaaaaaaaaaaaab')
                return final_acc, list_logit, list_y

        task_test_images, task_test_labels = load_task_specific_data(test_data, labels)
        if model.imp_method == 'PNN':
            pnn_train_phase = np.array(np.zeros(model.num_tasks), dtype=np.bool)
            logit_mask[:] = 0
            logit_mask[tt][labels] = 1.0
            feed_dict = {model.x: task_test_images,
                    model.y_[tt]: task_test_labels, model.keep_prob: 1.0}
            train_phase_dict = {m_t: i_t for (m_t, i_t) in zip(model.train_phase, pnn_train_phase)}
            logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, logit_mask)}
            feed_dict.update(train_phase_dict)
            feed_dict.update(logit_mask_dict)
            acc = model.accuracy[tt].eval(feed_dict = feed_dict)

        elif model.imp_method == 'A-GEM' or model.imp_method == 'ER' or model.imp_method == 'MEGA' or model.imp_method == 'MEGAD' or model.imp_method == 'CDD':
            logit_mask[:] = 0
            logit_mask[tt][labels] = 1.0
            feed_dict = {model.x: task_test_images,
                    model.y_: task_test_labels, model.keep_prob: 1.0, model.train_phase: False}
            logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, logit_mask)}
            feed_dict.update(logit_mask_dict)
            acc = model.accuracy[tt].eval(feed_dict = feed_dict)
            #pruned_logits:
            pruned_logits = model.features.eval(feed_dict=feed_dict)
            #pruned_logits = model.net.eval(feed_dict=feed_dict)
            #pruned_logits = model.features2.eval(feed_dict=feed_dict)  # tsne

        else:
            logit_mask[:] = 0
            logit_mask[labels] = 1.0
            feed_dict = {model.x: task_test_images,
                    model.y_: task_test_labels, model.keep_prob: 1.0, model.train_phase: False, model.output_mask: logit_mask}
            acc = model.accuracy.eval(feed_dict = feed_dict)

        # list_logit = pruned_logits  # mfr
        # list_y = task_test_labels
        # # save f to matrix mfr    id-task
        # yp = np.argmax(list_y, axis=-1)[:]
        # yp_dict = dict(collections.Counter(yp))
        # key_list = list(yp_dict.keys())
        # for w, p in enumerate(yp):
        #     MFR_matrix[task][tt][key_list.index(p)] += list_logit[w]
        # for p in range(5):
        #     MFR_matrix[task][tt][p] /= yp_dict[key_list[p]]

        list_logit.append(pruned_logits)  # tsne         maybe mfr also okay
        list_y.append(task_test_labels)

        final_acc[tt] = acc

    # print(np.array(list_logit).shape)  # (task,500,160)
    # print(np.mean(list_acc), list_acc)
    # exit()
    return final_acc, list_logit, list_y


def main():
    """
    Create the model and start the training
    """

    # Get the CL arguments
    args = get_arguments()

    # Check if the network architecture is valid
    if args.arch not in VALID_ARCHS:
        raise ValueError("Network architecture %s is not supported!"%(args.arch))

    # Check if the method to compute importance is valid
    if args.imp_method not in MODELS:
        raise ValueError("Importance measure %s is undefined!"%(args.imp_method))

    # Check if the optimizer is valid
    if args.optim not in VALID_OPTIMS:
        raise ValueError("Optimizer %s is undefined!"%(args.optim))

    # Create log directories to store the results
    if not os.path.exists(args.log_dir):
        print('Log directory %s created!'%(args.log_dir))
        os.makedirs(args.log_dir)

    # Generate the experiment key and store the meta data in a file
    exper_meta_data = {'ARCH': args.arch,
            'DATASET': 'SPLIT_CIFAR',
            'NUM_RUNS': args.num_runs,
            'TRAIN_SINGLE_EPOCH': args.train_single_epoch,
            'IMP_METHOD': args.imp_method,
            'SYNAP_STGTH': args.synap_stgth,
            'FISHER_EMA_DECAY': args.fisher_ema_decay,
            'FISHER_UPDATE_AFTER': args.fisher_update_after,
            'OPTIM': args.optim,
            'LR': args.learning_rate,
            'BATCH_SIZE': args.batch_size,
            'MEM_SIZE': args.mem_size,
            'M1_VALUE': args.m1_value,
            'M2_VALUE': args.m2_value,
            'S_VALUE': args.s_value}
    experiment_id = "SPLIT_CIFAR_HERDING_%s_%r_%s_%s_%s_%s_%s_%s_%s_%s-"%(args.arch, args.train_single_epoch, args.imp_method,
            str(args.synap_stgth).replace('.', '_'), str(args.learning_rate).replace('.', '_'),
            str(args.batch_size), str(args.mem_size), str(args.m1_value), str(args.m2_value), str(args.s_value)) + datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    snapshot_experiment_meta_data(args.log_dir, experiment_id, exper_meta_data)

    # Get the task labels from the total number of tasks and full label space
    if args.online_cross_val:
        num_tasks = K_FOR_CROSS_VAL
    else:
        num_tasks = NUM_TASKS - K_FOR_CROSS_VAL

    # Load the split cifar dataset
    data_labs = [np.arange(TOTAL_CLASSES)]
    datasets = construct_split_cifar(data_labs)

    # Variables to store the accuracies and standard deviations of the experiment
    acc_mean = dict()
    acc_std = dict()

    # Reset the default graph
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():

        # Set the random seed
        tf.set_random_seed(args.random_seed)

        # Define Input and Output of the model
        x = tf.placeholder(tf.float32, shape=[None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])

        # x = tf.placeholder(tf.float32, shape=[None, INPUT_FEATURE_SIZE])
        if args.imp_method == 'PNN':
            y_ = []
            for i in range(num_tasks):
                y_.append(tf.placeholder(tf.float32, shape=[None, TOTAL_CLASSES]))
        else:
            y_ = tf.placeholder(tf.float32, shape=[None, TOTAL_CLASSES])

        # Define the optimizer
        if args.optim == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

        elif args.optim == 'SGD':
            opt = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)

        elif args.optim == 'MOMENTUM':
            base_lr = tf.constant(args.learning_rate)
            learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - train_step / training_iters), OPT_POWER))
            opt = tf.train.MomentumOptimizer(args.learning_rate, OPT_MOMENTUM)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",x)
        # Create the Model/ contruct the graph
        model = Model(x, y_, num_tasks, opt, args.imp_method, args.synap_stgth, args.fisher_update_after,
                args.fisher_ema_decay, args.m1_value, args.m2_value, args.s_value, network_arch=args.arch)


        # Set up tf session and initialize variables.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config, graph=graph) as sess:
            runs, task_labels_dataset = train_task_sequence(model, sess, datasets, args)
            # Close the session

        # with open('time.txt', 'a') as f:
        #     f.write('cifar_m1_{}_m2_{}_s_{}_method_{}time:{}'.format(args.m1_value, args.m2_value,
        #                                                            args.s_value, args.imp_method, time_spent))
        #     f.write('\n')

    # Store all the results in one dictionary to process later
    exper_acc = dict(mean=runs)
    exper_labels = dict(labels=task_labels_dataset)

    # If cross-validation flag is enabled, store the stuff in a text file
    if args.cross_validate_mode:
        acc_mean, acc_std = average_acc_stats_across_runs(runs, model.imp_method)
        fgt_mean, fgt_std = average_fgt_stats_across_runs(runs, model.imp_method)
        cross_validate_dump_file = args.log_dir + '/' + 'SPLIT_CIFAR_%s_%s'%(args.imp_method, args.optim) + '.txt'
        with open(cross_validate_dump_file, 'a') as f:
            if MULTI_TASK:
                f.write('HERDING: {} \t ARCH: {} \t LR:{} \t LAMBDA: {} \t ACC: {}\n'.format(args.arch, args.learning_rate, args.synap_stgth, acc_mean[-1,:].mean()))
            else:
                f.write('ARCH: {} \t LR:{} \t LAMBDA: {} \t ACC: {} \t Fgt: {} \t Time: {}\n'.format(args.arch, args.learning_rate,
                    args.synap_stgth, acc_mean, fgt_mean, str(time_spent)))

    # Store the experiment output to a file
    fgt_mean, fgt_std = average_fgt_stats_across_runs(runs)
    ltr_mean, ltr_std = average_ltr_across_runs(runs)
    print('fgt_mean', fgt_mean)
    print('fgt_std', fgt_std)
    print('ltr_mean', ltr_mean)
    print('ltr_std', ltr_std)
    snapshot_experiment_eval(args.log_dir, experiment_id, exper_acc)
    snapshot_task_labels(args.log_dir, experiment_id, exper_labels)

def find_distance(data_vec,centroid, distance_metric):
    if distance_metric == 'euclidean':
        return np.linalg.norm(data_vec-centroid)
    elif distance_metric == 'euclidean_squared':
        return np.square(np.linalg.norm(data_vec-centroid))
    elif distance_metric == 'cosine':
        return distance.cosine(data_vec,centroid)

if __name__ == '__main__':
    main()
