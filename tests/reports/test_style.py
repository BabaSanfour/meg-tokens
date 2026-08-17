import re

from meg_tokens.reports import style

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def test_every_ordered_entity_has_a_colour():
    for order, colors in (
        (style.CLASS_ORDER, style.CLASS_COLORS),
        (style.CONDITION_ORDER, style.CONDITION_COLORS),
        (style.MODEL_ORDER, style.MODEL_COLORS),
        (style.QC_ORDER, style.QC_COLORS),
    ):
        for key in order:
            assert key in colors
            assert _HEX_RE.match(colors[key]), colors[key]


def test_the_three_palettes_are_pairwise_disjoint():
    class_hues = set(style.CLASS_COLORS[key] for key in style.CLASS_ORDER)
    condition_hues = set(style.CONDITION_COLORS[key] for key in style.CONDITION_ORDER)
    model_hues = set(style.MODEL_COLORS[key] for key in style.MODEL_ORDER)

    assert class_hues.isdisjoint(condition_hues)
    assert class_hues.isdisjoint(model_hues)
    assert condition_hues.isdisjoint(model_hues)


def test_apply_publication_style_sets_editable_text_fonttypes():
    import matplotlib as mpl

    before_pdf = mpl.rcParams["pdf.fonttype"]
    with style.apply_publication_style():
        assert mpl.rcParams["pdf.fonttype"] == 42
        assert mpl.rcParams["ps.fonttype"] == 42
    # Restored after the context exits (scoped, never leaks).
    assert mpl.rcParams["pdf.fonttype"] == before_pdf


def test_figure_grid_returns_a_two_dimensional_axes_array():
    with style.apply_publication_style():
        fig, axes = style.figure_grid(1, 2)
        assert axes.shape == (1, 2)
        assert axes[0, 0] is not axes[0, 1]
