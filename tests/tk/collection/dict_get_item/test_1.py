import pytest
from cemo.tk.collection import dict_get_item


@pytest.mark.parametrize(
    "input_dict, index, expected_output",
    [
        # Test case 1: Get item from nested dict
        (
            {
                'a': [1, 2, 3],
                'b': [4, 5, 6]
            },
            1,
            {
                'a': 2,
                'b': 5
            },
        ),
    ]
)
def test_dict_get_item(input_dict, index, expected_output):
    result = dict_get_item(input_dict, index)
    assert result == expected_output
