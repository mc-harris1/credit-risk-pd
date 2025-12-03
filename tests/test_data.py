from src.data.load_data import load_raw_loans

string = "accepted_2007_to_2018Q4.csv"


def test_load_raw_loans():
    df = load_raw_loans(string)
    assert not df.empty
