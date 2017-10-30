import re

fkey_pattern = re.compile(r'^F(\d{1,2})$', re.I)


def validate_function_key(slot_key):
    # check keyboard string
    m = fkey_pattern.match(slot_key)
    assert m
    # check function key is in 1-12 range
    assert 0 < int(m.group(1)) < 13
