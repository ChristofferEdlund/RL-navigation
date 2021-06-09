"""
Created on 2021-05-10. 22:19 

@author: Christoffer Edlund
"""
from unittest import TestCase
from .datatypes import SumTree

class TestSumTree(TestCase):

    def test_init(self):
        tree = SumTree(128)
        assert tree is not None


    def test_add(self):

        tree = SumTree(128)
        tree.add("asd", 1)

        assert True

    def test_look_up(self):
        tree = SumTree(4, 1)
        tree.add(1, 1)
        tree.add(2, 2)
        tree.add(3, 3)


        data, idx, _ = tree.lookup(1)
        assert data == 2

        data, idx, _ = tree.lookup(0.3)
        assert data == 1

        data, idx, _ = tree.lookup(5)
        assert data == 3

        total_priority = tree.root.priority
        assert total_priority == 6

    def test_update(self):
        tree = SumTree(4)
        tree.add(1, 1)
        tree.add(2, 2)
        tree.add(3, 3)

        data, idx, _ = tree.lookup(0.3)
        assert data == 1

        tree.update(idx, 10)

        data2, idx2, _ = tree.lookup(9)

        assert data == data2
        assert idx2 == idx
        prio1 = tree.root.priority
        tree.update(idx, 2, 10)

        data3, idx3, _ = tree.lookup(1.8)

        assert data3 == 10
        assert idx3 == idx
        assert prio1 > tree.root.priority

    def test_sum_priority(self):
        tree = SumTree(4)
        tree.add(1, 1)
        tree.add(2, 2)
        tree.add(3, 3)

        assert tree.sum_priority() == 6

        tree.add(4, 4)

        assert tree.sum_priority() == 10
        tree.add(2, 2)

        #Should cycle back to first element add change prio from 1 to 2, since buffer is only 5
        assert tree.sum_priority() == 11



    def test_sample(self):
        tree = SumTree(4)
        tree.add(1, 1)
        tree.add(2, 2.1)
        tree.add(3, 3)

        num_samples = 6
        samples, indices, prio = tree.sample(num_samples)

        assert num_samples == len(samples)
        assert num_samples == len(indices)

        tree = SumTree(2**8)
        for i in range(1000):
            tree.add(i, i)

        num_samples = 256
        samples, indices, _ = tree.sample(num_samples)
        assert num_samples == len(samples)
        assert num_samples == len(indices)


        num_samples = 64
        samples, indices, _ = tree.sample(num_samples)
        assert num_samples == len(samples)
        assert num_samples == len(indices)


    def test_low_priority(self):
        tree = SumTree(4, 0)
        tree.add(1, 1)
        tree.add(2, 2)
        tree.add(3, 3)
        tree.add(4, 4)

        for node in tree.nodes:
            assert node.priority == 1