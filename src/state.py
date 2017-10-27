from pykeyboard import PyKeyboard

keyboard = PyKeyboard()


def save(slot_num):
    assert 0 < slot_num < 13
    keyboard.press_keys([keyboard.shift_key, keyboard.function_keys[slot_num]])
    keyboard.release_key(keyboard.shift_key)
    keyboard.release_key(keyboard.function_keys[slot_num])


def load(slot_num):
    assert 0 < slot_num < 13
    keyboard.tap_key(keyboard.function_keys[slot_num])
