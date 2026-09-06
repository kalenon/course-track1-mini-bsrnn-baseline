import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_bsrnn.losses import MultiResL1SpecLoss
from mini_bsrnn.model import MiniBSRNN_SE, SUBBANDS


class BaselineTest(unittest.TestCase):
    def test_fixed_course_architecture(self):
        model = MiniBSRNN_SE()
        self.assertEqual(model.sample_rate, 16_000)
        self.assertEqual(model.bsrnn.embedding_dim, 64)
        self.assertEqual(model.bsrnn.num_layers, 2)
        self.assertEqual(sum(SUBBANDS), 161)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2_153_996)

    def test_forward_is_finite_and_length_preserving(self):
        torch.manual_seed(0)
        model = MiniBSRNN_SE().eval()
        noisy = torch.randn(2, 1600)
        lengths = torch.tensor([1600, 1200])
        with torch.no_grad():
            enhanced, spectrum = model(noisy, lengths, 16_000)
        self.assertEqual(enhanced.shape, noisy.shape)
        self.assertEqual(spectrum.shape[-1], 161)
        self.assertTrue(torch.isfinite(enhanced).all())
        self.assertEqual(torch.count_nonzero(enhanced[1, 1200:]).item(), 0)

    def test_stable_loss_on_near_silence(self):
        target = torch.randn(2, 2048)
        estimate = torch.zeros_like(target, requires_grad=True)
        loss = MultiResL1SpecLoss()(target, estimate).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(estimate.grad).all())


if __name__ == "__main__":
    unittest.main()
