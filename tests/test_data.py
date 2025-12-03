from src.data.load_data import load_raw_loans

string = "test.csv"


def test_load_raw_loans():
    df = load_raw_loans(string)
    assert not df.empty


# TODO: Build out unittest class to set up (create sample data) and tear down (delete sample data)
