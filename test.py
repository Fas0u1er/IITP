import unittest
from math import sqrt, isclose

import torch
import timeout_decorator
from sinkhorn import sinkhorn
from utils import rand_img


class CorrectnessTestCase(unittest.TestCase):
    def check_value(self, img1, img2, expected):
        v1 = sinkhorn(img1, img2)
        self.assertTrue(isclose(v1, expected, rel_tol=0.05, abs_tol=1e-2))

    def check_same_dist(self, img1, img2, img1_, img2_):
        self.check_value(img1, img2, sinkhorn(img1_, img2_))

    def test_same_img(self):
        img1 = [[1, 1],
                [1, 1]]

        img2 = [[1, 1],
                [1, 1]]

        self.check_value(img1, img2, 0)

    def test_one_bin(self):
        img1 = [[1, 0],
                [0, 0]]

        img2 = [[0, 0],
                [0, 1]]

        self.check_value(img1, img2, sqrt(2))

    def test_one_bin_rectangle(self):
        img1 = [[1, 0, 0],
                [0, 0, 0]]

        img2 = [[0, 0, 0],
                [0, 0, 1]]

        self.check_value(img1, img2, sqrt(5))

    def test_two_bins(self):
        img1 = [[1, 0],
                [1, 0]]

        img2 = [[0, 1],
                [0, 1]]

        self.check_value(img1, img2, 2)

        img1 = [[1, 0],
                [0, 1]]

        img2 = [[0, 1],
                [1, 0]]

        self.check_value(img1, img2, 2)

    def test_two_bins_overlap(self):
        img1 = [[1, 1],
                [0, 0]]

        img2 = [[0, 1],
                [0, 1]]

        self.check_value(img1, img2, sqrt(2))

    def test_two_bins_into_one(self):
        img1 = [[1, 1],
                [0, 0]]

        img2 = [[0, 0],
                [0, 2]]

        self.check_value(img1, img2, 1 + sqrt(2))

    def test_one_bin_into_two(self):
        img1 = [[0, 0],
                [0, 2]]

        img2 = [[1, 1],
                [0, 0]]

        self.check_value(img1, img2, 1 + sqrt(2))

    def test_4x4_4_bins(self):
        img1 = [[0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1]]

        img2 = [[0, 0, 0, 0],
                [2, 0, 0, 0],
                [2, 0, 0, 0],
                [0, 0, 0, 0]]

        self.check_value(img1, img2, 2 * (3 + sqrt(10)))

    def test_4x4_many_bins(self):
        img1 = [[1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1]]

        img2 = [[0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0]]

        self.check_value(img1, img2, 8)

    def test_one_bin_N_dim(self):
        N = 64
        img1 = torch.zeros(N, N)
        img2 = torch.zeros(N, N)

        img1[0, 0] = 1
        img2[-1, -1] = 1

        self.check_value(img1, img2, sqrt(2) * (N - 1))

    def test_two_bins_N_dim(self):
        N = 64
        img1 = torch.zeros(N, N)
        img2 = torch.zeros(N, N)

        img1[-1, 0] = 1
        img1[0, -1] = 1
        img2[-1, -1] = 2

        self.check_value(img1, img2, 2 * (N - 1))

    def test_many_bins_N_dim(self):
        N = 64

        img1 = [[(i + j) % 2 for i in range(N)] for j in range(N)]

        img2 = [[(i + j + 1) % 2 for i in range(N)] for j in range(N)]

        self.check_value(img1, img2, (N * N) / 2)

    def test_transpose(self):
        img1 = rand_img(10, 10)
        img2 = rand_img(10, 10)

        img1_ = img1.t()
        img2_ = img2.t()

        self.check_same_dist(img1, img2, img1_, img2_)

    def test_symmetry(self):
        img1 = rand_img(10, 10)
        img2 = rand_img(10, 10)

        self.check_same_dist(img1, img2, img2, img1)

    def test_shift(self):
        hist1 = rand_img(6, 6)
        hist2 = rand_img(6, 6)

        img1 = torch.zeros(10, 10)
        img2 = torch.zeros(10, 10)
        img1[0:6, 1:7] = hist1.clone()
        img2[1:7, 2:8] = hist2.clone()

        img1_ = torch.zeros(10, 10)
        img2_ = torch.zeros(10, 10)
        img1_[2:8, 3:9] = hist1.clone()
        img2_[3:9, 4:10] = hist2.clone()

        self.check_same_dist(img1, img2, img1_, img2_)


class SpeedTestCase(unittest.TestCase):
    @timeout_decorator.timeout(0.5)
    def test_stress_1(self):
        img1 = rand_img(64, 64)
        img2 = rand_img(64, 64)
        print(sinkhorn(img1, img2))

    @timeout_decorator.timeout(0.7)
    def test_stress_2(self):
        img1 = rand_img(128, 128)
        img2 = rand_img(128, 128)
        print(sinkhorn(img1, img2))


if __name__ == '__main__':
    unittest.main()
