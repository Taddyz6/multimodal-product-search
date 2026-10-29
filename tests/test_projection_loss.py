"""Optional small tensor checks; no training, labels or checkpoint loading."""

import numpy as np
import pandas as pd
import pytest
from product_search.research.projection import project, validate_alignment_roles


def test_projection_normalizes_and_rejects_zero_vectors():
    value = project(np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32))
    np.testing.assert_array_equal(value, np.eye(2, dtype=np.float32))
    with pytest.raises(ValueError):
        project(np.zeros((1, 2), np.float32), np.eye(2, dtype=np.float32))


def test_alignment_forbids_cold_training_products():
    products = pd.DataFrame(
        [
            dict(
                product_locale="us",
                product_id="a",
                product_split="cold_val",
                embedding_row=0,
                has_image_embedding=True,
            ),
            dict(
                product_locale="us",
                product_id="b",
                product_split="warm",
                embedding_row=1,
                has_image_embedding=True,
            ),
        ]
    )
    train = products.iloc[[0]]
    validation = products.iloc[[1]]
    with pytest.raises(ValueError, match="Cold pairs forbidden"):
        validate_alignment_roles(products, train, validation, train)


def test_listnet_masks_padding_and_keeps_grade_softmax():
    torch = pytest.importorskip("torch")
    from product_search.research.ranking import listnet_loss

    logits = torch.tensor([[0.1, 0.4, 99.0]], requires_grad=True)
    grades = torch.tensor([[3.0, 0.0, 99.0]])
    mask = torch.tensor([[True, True, False]])
    loss = listnet_loss(logits, grades, mask)
    expected = -(torch.softmax(grades[:, :2], 1) * torch.log_softmax(logits[:, :2], 1)).sum()
    assert float(loss.detach()) == pytest.approx(float(expected.detach()))
    loss.backward()
    assert float(logits.grad[0, 2]) == 0


def test_symmetric_infonce_respects_pairing():
    torch = pytest.importorskip("torch")
    from product_search.research.projection import symmetric_infonce

    vectors = torch.eye(3)
    assert symmetric_infonce(vectors, vectors) < symmetric_infonce(vectors, vectors.flip(0))
    with pytest.raises(ValueError):
        symmetric_infonce(vectors[:1], vectors[:1])
