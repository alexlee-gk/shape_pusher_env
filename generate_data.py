import argparse
import os
import random
import shutil
from multiprocessing import Pool

import numpy as np
import tensorflow as tf

from shape_pusher_env import ShapePusherEnv


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def save_tf_record(data_fname, trajs):
    """
    saves data_files from one sample trajectory into one tf-record file
    """
    print('Writing', data_fname)
    writer = tf.python_io.TFRecordWriter(data_fname)
    feature = {}
    for traj in trajs:
        for t, (obs, action) in enumerate(traj):
            if action is not None:
                feature['%d/action' % t] = _float_feature(action.tolist())
            feature['%d/state' % t] = _float_feature(obs['manip_xy'].tolist())
            feature['%d/obj_pose' % t] = _float_feature(obs['obj_pose'].tolist())
            feature['%d/image/encoded' % t] = _bytes_feature(obs['image'].tostring())
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()


def generate_and_save_data(start_traj_iter, end_traj_iter, num_steps, data_dir, momentum=0.9):
    print('started process with PID:', os.getpid())
    print('making trajectories {0} to {1}'.format(start_traj_iter, end_traj_iter))

    random.seed(None)
    np.random.seed(None)

    env = ShapePusherEnv(write_assets=False)

    trajs = []
    last_start_traj_iter = start_traj_iter
    for traj_iter in range(start_traj_iter, end_traj_iter):
        if not trajs:
            last_start_traj_iter = traj_iter
        # only collect trajectories that moved the object
        traj = []
        while not traj or np.linalg.norm(traj[0][0]['obj_pose'] - traj[-1][0]['obj_pose']) < 0.025:
            traj[:] = []
            obs = env.reset()
            last_action = 0.0
            for _ in range(num_steps):
                action = env.action_space.sample() + momentum * last_action - obs['manip_xy']
                traj.append((obs, action))
                obs, reward, done, info_dict = env.step(action)  # action is projected down to the action space (in-place)
                last_action = action
            traj.append((obs, None))

        trajs.append(traj)
        trajs_per_file = 256
        if len(trajs) == trajs_per_file or traj_iter == (end_traj_iter - 1):
            data_fname = 'traj_{0}_to_{1}'.format(last_start_traj_iter, traj_iter)
            data_fname = os.path.join(data_dir, data_fname) + '.tfrecords'
            save_tf_record(data_fname, trajs)
            trajs[:] = []


def generate_and_save_data_helper(args):
    generate_and_save_data(*args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--num_workers', type=int, default=1, help='number of parallel workers')
    parser.add_argument('--num_trajs', type=int, default=50000)
    parser.add_argument('--num_steps', type=int, default=30)
    args = parser.parse_args()

    train_data_dir = os.path.join(args.data_dir, 'train')
    os.makedirs(train_data_dir, exist_ok=True)
    test_data_dir = os.path.join(args.data_dir, 'test')
    os.makedirs(test_data_dir, exist_ok=True)

    num_trajs_per_worker = args.num_trajs // args.num_workers
    start_traj_iters = [num_trajs_per_worker * i for i in range(args.num_workers)]
    end_traj_iters = [num_trajs_per_worker * (i + 1) - 1 for i in range(args.num_workers)]
    end_traj_iters[-1] = args.num_trajs

    if args.num_workers > 1:
        p = Pool(args.num_workers)
        p.map(generate_and_save_data_helper, zip(start_traj_iters, end_traj_iters,
                                                 [args.num_steps] * args.num_workers,
                                                 [train_data_dir] * args.num_workers))
    else:
        start_traj_iter, = start_traj_iters
        end_traj_iter, = end_traj_iters
        generate_and_save_data(start_traj_iter, end_traj_iter, args.num_steps, train_data_dir)

    # move first file from train to test
    shutil.move(os.path.join(train_data_dir, 'traj_0_to_255.tfrecords'),
                os.path.join(test_data_dir, 'traj_0_to_255.tfrecords'))


if __name__ == '__main__':
    main()
