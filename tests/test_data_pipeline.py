import pytest
from mypkg.pipeline.setup.dataset import load_raw_dataset, filter_by_industry

def test_load_dataset():
    df = load_raw_dataset()
    assert not df.empty, "Dataset should not be empty"

def test_filter_by_industry():
    df = load_raw_dataset()
    it_df = filter_by_industry(df, "INFORMATION-TECHNOLOGY")
    assert all(it_df["Category"] == "INFORMATION-TECHNOLOGY"), "Should only have IT category"
