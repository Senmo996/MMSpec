import pytest

from evaluation.audit_mmspec_disjoint_ranges import audit, parse_range


def test_parse_range_accepts_label_start_count():
    assert parse_range("confirmation:325:200") == ("confirmation", 325, 200)


@pytest.mark.parametrize(
    "value",
    ("missing", ":0:1", "label:-1:1", "label:0:0", "label:x:1"),
)
def test_parse_range_rejects_invalid_values(value):
    with pytest.raises(Exception):
        parse_range(value)


def test_audit_reports_disjoint_unique_ranges():
    rows = [{"id": f"sample-{index}", "image": f"image-{index}.jpg"} for index in range(12)]
    result = audit(
        rows,
        seed=7,
        ranges=[("development", 0, 4), ("validation", 4, 6)],
    )

    assert result["all_unique_within_range"] is True
    assert result["all_pairwise_overlaps_zero"] is True
    assert result["pairwise_overlaps"] == [
        {
            "left": "development",
            "right": "validation",
            "source_index_overlap": 0,
            "image_cluster_overlap": 0,
        }
    ]


def test_audit_detects_duplicate_image_clusters_across_ranges():
    rows = [
        {"id": "a", "image": "shared.jpg"},
        {"id": "b", "image": "shared.jpg"},
    ]
    result = audit(
        rows,
        seed=0,
        ranges=[("left", 0, 1), ("right", 1, 1)],
    )

    assert result["all_unique_within_range"] is True
    assert result["all_pairwise_overlaps_zero"] is False
    assert result["pairwise_overlaps"][0]["image_cluster_overlap"] == 1


def test_audit_rejects_range_beyond_dataset():
    with pytest.raises(ValueError, match="exceeds"):
        audit(
            [{"id": "only", "image": "only.jpg"}],
            seed=42,
            ranges=[("too_large", 0, 2)],
        )
