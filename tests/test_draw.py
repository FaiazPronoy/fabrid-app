import numpy as np

from defects.draw import annotate, outline
from defects.labels import COLORS
from fakes import square


def test_mask_and_box_are_drawn_in_class_colour():
    pixels = np.zeros((120, 160, 3), dtype=np.uint8)
    item = square("hole", 0.9, 40, 50, 100, 110, size=(160, 120))
    out = annotate(pixels, [item])
    color = np.array(COLORS["hole"])
    inside = out[80, 70].astype(int)
    assert np.allclose(inside, np.round(color * 0.45), atol=1)
    assert np.array_equal(out[110 - 1, 70], color)
    assert np.array_equal(out[5, 5], [0, 0, 0])
    assert np.array_equal(pixels, np.zeros_like(pixels))


def test_label_tag_is_drawn_near_box():
    pixels = np.zeros((200, 200, 3), dtype=np.uint8)
    item = square("stain", 0.75, 60, 80, 140, 160, size=(200, 200))
    out = annotate(pixels, [item])
    tag = out[60:80, 60:140]
    assert (tag == np.array(COLORS["stain"])).all(axis=-1).any()
    assert (tag == 255).all(axis=-1).any()


def test_each_class_has_its_own_colour():
    pixels = np.zeros((400, 400, 3), dtype=np.uint8)
    found = [
        square("needle mark", 0.9, 20, 60, 120, 160, size=(400, 400)),
        square("missing yarn", 0.9, 240, 260, 360, 380, size=(400, 400)),
    ]
    out = annotate(pixels, found)
    assert np.array_equal(out[159, 70], COLORS["needle mark"])
    assert np.array_equal(out[379, 300], COLORS["missing yarn"])


def test_empty_prediction_leaves_image_unchanged():
    pixels = np.random.default_rng(0).integers(0, 256, (30, 40, 3), dtype=np.uint8)
    assert np.array_equal(annotate(pixels, []), pixels)


def test_outline_of_square():
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 1:5] = True
    edge = outline(mask)
    assert edge.sum() == 12
    assert not edge[2, 2]
