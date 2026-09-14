import unittest
from unittest.mock import patch

import numpy as np

import app


class FakeJinaModel:
    def __init__(self):
        self.image_calls = []
        self.text_calls = []

    def encode_image(self, images, **kwargs):
        self.image_calls.append((images, kwargs))
        return np.asarray([[3.0, 4.0]], dtype=np.float32)

    def encode_text(self, texts, **kwargs):
        self.text_calls.append((texts, kwargs))
        return np.asarray([[0.0, 5.0]], dtype=np.float32)


class OutOfMemoryJinaModel:
    def encode_image(self, images, **kwargs):
        raise app.torch.cuda.OutOfMemoryError("out of memory")


class JinaClipAdapterTests(unittest.TestCase):
    def test_runtime_device_prefers_cuda(self):
        with patch.object(app.torch.cuda, "is_available", return_value=True):
            self.assertEqual(app.runtime_device(), "cuda")

    def test_runtime_device_falls_back_to_cpu(self):
        with patch.object(app.torch.cuda, "is_available", return_value=False):
            self.assertEqual(app.runtime_device(), "cpu")

    def test_encode_images_uses_matryoshka_dimension_and_normalizes(self):
        model = FakeJinaModel()

        embeddings = app.encode_images(model, ["image"])

        self.assertEqual(model.image_calls, [(["image"], {"truncate_dim": 512})])
        np.testing.assert_allclose(embeddings, [[0.6, 0.8]])

    def test_encode_query_marks_multilingual_text_as_retrieval_query(self):
        model = FakeJinaModel()

        embedding = app.encode_query(model, "붉은색 회복 포션")

        self.assertEqual(
            model.text_calls,
            [(["붉은색 회복 포션"], {"task": "retrieval.query", "truncate_dim": 512})],
        )
        np.testing.assert_allclose(embedding, [0.0, 1.0])

    def test_encode_images_explains_cuda_memory_failure(self):
        with self.assertRaisesRegex(RuntimeError, "GPU 메모리가 부족합니다"):
            app.encode_images(OutOfMemoryJinaModel(), ["image"])


if __name__ == "__main__":
    unittest.main()
