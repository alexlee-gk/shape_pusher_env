import time

from shape_pusher_env import ShapePusherEnv


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    env = ShapePusherEnv()
    num_trajs = 100
    num_steps = 100
    momentum = 0.9
    for _ in range(num_trajs):
        obs = env.reset()
        last_action = 0.0
        for _ in range(num_steps):
            action = env.action_space.sample() + momentum * last_action - obs['manip_xy']
            obs, reward, done, info_dict = env.step(action)  # action is projected down to the action space (in-place)
            last_action = action
            env.render()
            time.sleep(0.01)
