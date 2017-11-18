import src.keylog as keylog
from src import pad


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
            key_pad = pad.Button.A
        elif key == 'z':
            key_pad = pad.Button.B
        elif key == 'c':
            key_pad = pad.Button.X
        elif key == 's':
            key_pad = pad.Button.Y
        elif key == 'd':
            key_pad = pad.Button.Z
        elif key == 'enter':
            key_pad = pad.Button.START
        elif key == 'left' or key == 'right' or key == 'up' or key == 'down':
            key_pad = pad.Stick.MAIN
        return key_pad
