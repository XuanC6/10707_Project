import gym

import numpy as np

from multiprocessing import Process, Pipe


def worker(env_name, remote, parent_remote):
    parent_remote.close()
    env = gym.make(env_name)
    env.reset()

    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            res = env.step(data)
            if res[2] == True:
                env.reset()
            remote.send(res)
        else:
            print("Invalid command!")


class EnvGroup():


    def __init__(self, env_name, num_processes):

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_processes)])
        self.procs = [Process(target=worker, args=(env_name, work_remote, remote))
                      for (work_remote, remote) in zip(self.work_remotes, self.remotes)]

        for p in self.procs:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()


    def step(self, actions):

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        results = [remote.recv() for remote in self.remotes]
        next_states, rewards, terminals, infos = zip(*results)

        return next_states, rewards, terminals, infos
