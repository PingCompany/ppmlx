"""Tests for the local memory graph view."""
from __future__ import annotations

from ppmlx.graph_view import render_graph_html


def test_render_graph_html_uses_pinned_antv_g6() -> None:
    html = render_graph_html({"status": "active", "limit": "120"})

    assert "https://cdn.jsdelivr.net/npm/@antv/g6@4.8.24/dist/g6.min.js" in html
    assert "new G6.Graph" in html
    assert "type: 'force'" in html
    assert "drag-canvas" in html
    assert "zoom-canvas" in html
    assert "drag-node" in html
    assert "Unable to load AntV G6 from the CDN" in html


def test_render_graph_html_does_not_include_manual_svg_ring_layout() -> None:
    html = render_graph_html()

    assert '<svg id="graph"' not in html
    assert "createElementNS" not in html
    assert "setAttribute('viewBox'" not in html
    assert "Math.cos" not in html
    assert "Math.sin" not in html
