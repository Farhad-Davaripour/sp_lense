import unittest

import torch

from src.recognition_readout_geometry import norm


class QwenReadoutNormTests(unittest.TestCase):
    def test_zero_centered_weight_includes_one(self):
        actual=norm(torch.tensor([2.,2.]),torch.tensor([1.,0.]),0.)
        torch.testing.assert_close(actual,torch.tensor([2.,1.]))


if __name__=='__main__': unittest.main()
