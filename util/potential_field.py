import random
import numbers
import os
import os.path
import numpy as np
import struct
import math
import time


class PotentialField:
    def __init__(self, node_num, dim):
        self.node_num = node_num
        self.dim = dim
        np.random.seed(2017)
        self.node = np.random.rand(self.node_num, self.dim) * 2 - 1
        np.random.seed()

        self.learning_rate = 0.01

    def node_force(self, src, dst):
        # return the force from src to dst
        f = dst - src
        f_norm = np.linalg.norm(f) + 0.00001
        f = f / f_norm / f_norm ** 2
        return f

    def wall_force(self, dst):
        f = np.zeros(self.dim)
        for i in range(self.dim):
            x = dst[i]
            # no force if far away
            if math.fabs(x) < 0.01:
                continue

            f_tmp = np.zeros(self.dim)
            f_tmp[i] = -1 * x * self.node_num/1.5
            f = f + f_tmp
        return f

    def get_total_node_force(self):
        force = np.zeros((self.node_num, self.dim))
        for j in range(self.node_num):
            dst = self.node[j]
            for k in range(self.node_num):
                force[j] += self.node_force(self.node[k], dst)
        return force

    def get_total_wall_force(self):
        force = np.zeros((self.node_num, self.dim))
        for j in range(self.node_num):
            dst = self.node[j]
            force[j] += self.wall_force(dst)
        return force

    def optimize(self):
        for i in range(100):
            learning_rate = self.learning_rate

            # cumulate the force
            force = np.zeros((self.node_num, self.dim))
            for j in range(self.node_num):
                dst = self.node[j]
                force[j] += self.wall_force(dst)

                for k in range(self.node_num):
                    force[j] += self.node_force(self.node[k], dst)

                    # apply the force
            self.node += force * learning_rate

        self.reorder()

    def reorder(self):
        node_ordered = self.node[self.node[:, 0].argsort()]

        rows = int(math.sqrt(self.node_num))
        cols = rows
        node_ordered = node_ordered.reshape((rows, cols, self.dim))
        for i in range(rows):
            node_row = node_ordered[i]
            node_row = node_row[node_row[:, 1].argsort()]
            node_ordered[i] = node_row
        node_ordered = node_ordered.reshape((self.node_num, self.dim))

        self.node = node_ordered

