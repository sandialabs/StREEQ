import pytest
import yaml
from pathlib import Path

from core.Parser.CategoryParser import param_list_items, path_str

#=======================================================================================================================
categories = []
for file_path in Path("core/Parser").glob("*/defaults.yaml"):
    categories.append((file_path.parents[0].parts[-1],))

#=======================================================================================================================
@pytest.mark.parametrize("category", categories)
def test_parser_param_consistency(category):
    path = Path("core/Parser") / category[0]
    defaults_keys = get_param_list_items_keys('defaults', path)
    datatypes_keys = get_param_list_items_keys('datatypes', path)
    descriptions_keys = get_param_list_items_keys('descriptions', path)
    perform_consistency_test(category[0], defaults_keys, datatypes_keys, descriptions_keys)

#=======================================================================================================================
def get_param_list_items_keys(filename, path):
    with open(path / f'{filename}.yaml') as file:
        params = yaml.safe_load(file)
    items_keys = []
    for item in param_list_items(params):
        if ('kwargs' in item[0]) and (not item[-1] == 'kwargs'):
            pass
        else:
            items_keys.append(path_str(item[0]))
    return set(items_keys)

#=======================================================================================================================
def perform_consistency_test(category, defaults_keys, datatypes_keys, descriptions_keys):
    print(f"Testing category: {category}")
    result = 'PASSED' if (defaults_keys == datatypes_keys) else 'DIFFED'
    print(f"  comparing defaults keys to datatypes keys--{result}")
    assert result == 'PASSED'
    result = 'PASSED' if (defaults_keys == descriptions_keys) else 'DIFFED'
    print(f"  comparing defaults keys to descriptions_keys keys--{result}")
    assert result == 'PASSED'
    print()
