"""Tes modul berbasis torch. Dilewati otomatis jika torch/torchvision belum terpasang."""
import importlib.util, unittest
from unittest import mock

HAS_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "torchvision"))


@unittest.skipUnless(HAS_TORCH, "torch/torchvision belum terpasang")
class TestClsModels(unittest.TestCase):
    def build(self, arch, mode):
        from torchvision import models
        from vision import cls_models
        r, m = models.resnet18, models.mobilenet_v3_small      # hindari unduhan bobot saat tes
        with mock.patch.object(models, "resnet18", lambda weights=None: r(weights=None)), \
             mock.patch.object(models, "mobilenet_v3_small", lambda weights=None: m(weights=None)):
            return cls_models.build(arch, mode, 5)

    def trainable(self, model):
        return {n for n, p in model.named_parameters() if p.requires_grad}

    def test_resnet_feature_only_head(self):
        model, groups = self.build("resnet18", "feature")
        self.assertTrue(self.trainable(model) and all(n.startswith("fc.") for n in self.trainable(model)))
        self.assertEqual(len(groups), 1)

    def test_resnet_partial_layer4_and_head(self):
        model, groups = self.build("resnet18", "partial")
        t = self.trainable(model)
        self.assertTrue(all(n.startswith(("layer4.", "fc.")) for n in t)); self.assertEqual(len(groups), 2)

    def test_scratch_all_trainable(self):
        model, _ = self.build("resnet18", "scratch")
        self.assertEqual(len(self.trainable(model)), len(list(model.parameters())))

    def test_mobilenet_partial(self):
        model, _ = self.build("mobilenet_v3_small", "partial")
        t = self.trainable(model)
        self.assertTrue(any(n.startswith("features.12.") for n in t))
        self.assertFalse(any(n.startswith("features.0.") for n in t))

    def test_output_shape(self):
        import torch
        model, _ = self.build("resnet18", "scratch")
        model.eval()
        self.assertEqual(tuple(model(torch.zeros(2, 3, 224, 224)).shape), (2, 5))

    def test_frozen_batchnorm_stays_eval(self):
        from vision import cls_models
        m_feat, _ = self.build("resnet18", "feature"); cls_models.set_train(m_feat)
        self.assertFalse(m_feat.bn1.training)
        m_scr, _ = self.build("resnet18", "scratch"); cls_models.set_train(m_scr)
        self.assertTrue(m_scr.bn1.training)

    def test_invalid_arch_or_mode(self):
        from vision import cls_models
        with self.assertRaises(ValueError):
            cls_models.build("vgg", "feature", 5)
        with self.assertRaises(ValueError):
            cls_models.build("resnet18", "lain", 5)


if __name__ == "__main__":
    unittest.main()
