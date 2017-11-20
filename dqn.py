import argparse
import logging
import sys

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np


class QFunction(chainer.Chain):

    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_actions)

    def __call__(self, x):
        h = F.tanh(self.l0(x))
        h = F.tanh(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))


def main(args):
    # Make an environment
    env = gym.make(args.env_id)

    print('environment: {}'.format(args.env_id))
    print('observation space: {}'.format(env.observation_space))
    print('action space: {}'.format(env.action_space))

    # Setup a model(q_func)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    q_func = QFunction(obs_size, n_actions)
    # q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
    #     obs_size, n_actions, n_hidden_layers=2, n_hidden_channels=50)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        q_func.to_gpu()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)

    # Create an agent
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
    gamma = 0.95
    explorer = chainerrl.explorers.ConstantEpsilonGreedy(
        epsilon=0.3, random_action_func=env.action_space.sample)
    agent = chainerrl.agents.DQN(
        q_func, optimizer, replay_buffer, gamma, explorer,
        replay_start_size=500, update_interval=1, target_update_interval=100,
        phi=lambda x: x.astype(np.float32, copy=False))

    # Train
    gym.undo_logger_setup()  # Turn off gym's default logger settings
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

    chainerrl.experiments.train_agent_with_evaluation(
        agent, env,
        steps=args.step,
        eval_n_runs=10,
        max_episode_len=200,
        eval_interval=1000,
        outdir=args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0',
                        help='Select the environment to run.')
    parser.add_argument('--step', '-s', type=int, default=2000,
                        help='Number of total time steps for training.')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--render', '-r', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    main(parser.parse_args())
