import numpy as np

from src.disintegration import DisintegrationEffect


def test_fallback_mask_is_not_full_frame():
    effect = DisintegrationEffect(200, 200)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    mask = effect._build_fallback_mask(frame)

    assert mask.dtype == np.uint8
    assert mask.shape == (200, 200)
    assert mask.sum() > 0
    assert mask.sum() < frame.shape[0] * frame.shape[1]
