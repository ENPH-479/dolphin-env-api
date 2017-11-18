import src.keylog as keylog
from src import dp_controller


class KeyPadMap:
    def __init__(self):
        self.state = dict((el.name, False) for el in keylog.Keyboard)

    def update(self, keys):
        # TODO track which keys are pressed and which are released.
        # TODO track 'toggled' MAIN stick positions as well.
        updates = []
        for key in keys:
            if self.state[key] == False:
                pass
            self.state[key] = True

    def convert_key(self, key, is_press):
        # TODO organize in general way such that an controller input can be sent to pipe only knowing the key
        key_pad = None
        if key == 'x':
            key_pad = dp_controller.Button.A
        elif key == 'z':
            key_pad = dp_controller.Button.B
        elif key == 'c':
            key_pad = dp_controller.Button.X
        elif key == 's':
            key_pad = dp_controller.Button.Y
        elif key == 'd':
            key_pad = dp_controller.Button.Z
        elif key == 'enter':
            key_pad = dp_controller.Button.START
        elif key == 'left' or key == 'right' or key == 'up' or key == 'down':
            key_pad = dp_controller.Stick.MAIN
        return key_pad
