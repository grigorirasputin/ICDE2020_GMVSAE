import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses C++ level warnings
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import sys
sys.path.append("../")

import math
import time
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


from data_generator import DataGenerator
from model import Model

from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_curve, auc


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

optimizers = {
        'sgd': lambda lr: tf.train.MomentumOptimizer(lr, 0.0),
        'momentum': lambda lr: tf.train.MomentumOptimizer(lr, 0.9),
        'adagrad': lambda lr: tf.train.AdagradOptimizer(lr),
        'adam': lambda lr: tf.train.AdamOptimizer(lr)
}


def define_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="train",
                        help='choose a mode')

    parser.add_argument('--eval_split', type=str, default='val', choices=['val', 'test'],
                        help='Which split to evaluate (val or test)')

    parser.add_argument('--data_filename', type=str, default="./data/processed_porto.csv",
                        help='data file')

    parser.add_argument('--map_size', type=int, nargs=2, default=[98, 65],
                        help='Grid dimensions: M_columns and N_rows')

    parser.add_argument('--model', type=str, default="gmvsae",
                        help='choose a model')

    parser.add_argument('--token_dim', type=int, default=32,
                        help='size of input embedding')
    parser.add_argument('--rnn_dim', type=int, default=256,
                        help='size of RNN hidden state')
    parser.add_argument('--cluster_num', type=int, default=10,
                        help='size of Gaussian components')

    parser.add_argument('--model_dir', type=str, default="ckpt",
                        help='model dir')
    parser.add_argument('--pretrain_dir', type=str, default="",
                        help='model dir')

    parser.add_argument('--log_steps', type=int, default=10,
                        help='num of steps to print log')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='number of epochs')
    #parser.add_argument('--vocab_size', type=int, required=True,
                    #     help='The Actual Dense Vocab Size from your grid_config.json') # ADDED CUSTOM:

    parser.add_argument('--num_negs', type=int, default=64,
                        help='size of negative sampling')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')

    parser.add_argument('--partial_ratio', type=float, default=1.0,
                        help='partial trajectory evaluation')

    parser.add_argument('--optimizer', type=str, default="adam",
                        help='optimizer')
    parser.add_argument('--gpu_id', type=str, default="0")
    args = parser.parse_args()
    return args

def run_train(model, args, master='', is_chief=True):
    sampler = DataGenerator(args)

    batch_size = args.batch_size
    batch_data = sampler.next_batch(batch_size)

    loss, pretrain_loss, z = model(batch_data)

    optimizer_class = optimizers.get(args.optimizer)
    optimizer = optimizer_class(args.learning_rate)
    global_step = tf.train.get_or_create_global_step()

    if args.mode == 'train':
        train_op = optimizer.minimize(loss, global_step=global_step)
    elif args.mode == 'pretrain':
        train_op = optimizer.minimize(pretrain_loss, global_step=global_step)

    hooks = []

    # step hook
    num_steps = int(sampler.train_traj_num // batch_size * args.num_epochs)
    hooks.append(tf.train.StopAtStepHook(last_step=num_steps))

    # log hook
    tensor_to_log = {'step': global_step, 'loss': loss, 'total_steps': tf.constant(num_steps)}
    hooks.append(
        tf.train.LoggingTensorHook(
           tensor_to_log, every_n_iter=args.log_steps))

    # Build the folder name safely
    folder_name = f"{args.model}_{args.token_dim}_{args.rnn_dim}_{args.cluster_num}"
    
    # Use os.path.abspath and os.path.join to handle Windows slashes correctly
    ckpt_dir = os.path.abspath(os.path.join(args.model_dir, folder_name))
    output_dir = ckpt_dir
    
    # Ensure the directory exists
    os.makedirs(ckpt_dir, exist_ok=True)
    hooks.append(tf.train.CheckpointSaverHook(checkpoint_dir=ckpt_dir, save_steps=500,
                 saver=tf.train.Saver(max_to_keep=1)))

    with tf.train.MonitoredTrainingSession(
        master=master,
        is_chief=is_chief,
        checkpoint_dir=ckpt_dir,
        log_step_count_steps=None,
        hooks=hooks,
        config=config) as sess:
      while not sess.should_stop():
        sess.run(train_op)

    if args.mode == 'pretrain':
        sample_num = 10000
        num_steps = int(sample_num // batch_size)
        x_embedded = []
        with tf.train.MonitoredTrainingSession(
                master=master,
                is_chief=is_chief,
                checkpoint_dir=ckpt_dir,
                save_checkpoint_secs=None,
                log_step_count_steps=None,
                config=config) as sess:
                for _ in range(num_steps):
                    x_embedded.append(sess.run(z))
        
        x_embedded = np.concatenate(x_embedded, axis=0)
        x_embedded = x_embedded[:sample_num]
        print(x_embedded.shape)
                    
        kmeans = KMeans(n_clusters=args.cluster_num)
        kmeans.fit(x_embedded)
        init_mu_c = kmeans.cluster_centers_
        np.savez("{}/init_mu_c".format(output_dir), init_mu_c)

def run_evaluate(model, args, master='', is_chief=True):
    def auc_score(y_true, y_score):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    sampler = DataGenerator(args)

    eval_data = args.eval_split #remove hardcoded one 
    # eval_data = 'test' # remember to change to train and val 
    if eval_data == 'train':
        traj_num = sampler.train_traj_num
        sd_tids = sampler.train_sd
    elif eval_data == 'val':
        traj_num = sampler.val_traj_num
        sd_tids = sampler.val_sd
    elif eval_data == 'test':               # <--- ADD THIS BLOCK
        traj_num = sampler.test_traj_num
        sd_tids = sampler.test_sd

    #sampler.inject_outliers(otype='random', data_type=eval_data, ratio=0.05, 
    #                        level=3, point_prob=0.3, vary=False)

    batch_size = args.batch_size
    dataset = tf.data.Dataset.from_generator(
        sampler.data_iterator[eval_data], 
        (tf.int32, tf.int32, tf.int32), 
        ([batch_size, None], [batch_size, None], [batch_size]))
    source = dataset.make_one_shot_iterator().get_next()
    scores = model(source)

    global_step = tf.train.get_or_create_global_step()

    global_step = tf.train.get_or_create_global_step()

    # USE WINDOWS PATHS:
    folder_name = f"{args.model}_{args.token_dim}_{args.rnn_dim}_{args.cluster_num}"
    ckpt_dir = os.path.abspath(os.path.join(args.model_dir, folder_name))
    
    score_vals = []

    with tf.train.MonitoredTrainingSession(
        master=master,
        is_chief=is_chief,
        checkpoint_dir=ckpt_dir,
        save_checkpoint_secs=None,
        log_step_count_steps=None,
        config=config) as sess:
        while not sess.should_stop():
            score_vals.append(sess.run(scores))

    score_vals = np.concatenate(score_vals, axis=-1)
    score_vals = score_vals[:traj_num]
    """
    y_true = np.ones_like(score_vals)
    for idx in sampler.outlier_idx:
        if idx < y_true.shape[0]:
            y_true[idx] = 0

    sd_auc = []
    for sd, tids in sd_tids.items():
        if sum(y_true[tids]) > 0:
            sd_auc.append(auc_score(y_true=y_true[tids], y_score=score_vals[tids]))
    sd_auc = np.array(sd_auc)
    print("Average AUC:", np.mean(sd_auc))
    """
    save_path = f"scores_{args.model}_{eval_data}.npy"
    np.save(save_path, score_vals)
    print(f"Successfully saved {len(score_vals)} raw anomaly scores to {save_path}")
    # Force exit so it doesn't crash on their old AUC code
    import sys; sys.exit(0)
   
if __name__ == '__main__':
    args = define_args()

    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    model = Model(args)
    if args.mode == 'train':
        run_train(model, args)
    elif args.mode == 'pretrain':
        run_train(model, args)
    elif args.mode == 'eval':
        run_evaluate(model, args)
    else:
        raise ValueError("Mode not in ['pretrain', 'train', 'eval'].")

