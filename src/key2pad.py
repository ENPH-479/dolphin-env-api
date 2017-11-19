import src.keylog as keylog
from src import dp_controller


class KeyPadMap:
    def __init__(self):
        self.state = dict((el.name, False) for el in keylog.Keyboard)

    def update(self, keys):
        # TODO track which keys are pressed and which are released.
        # TODO track 'toggled' MAIN stick positions as well.
        updates_button = []
        for key in keys:
            if (key=='left' or 'right'or 'up' or 'down') &self.state[key] == 0:
                if updates_button[key]== 1:
                    self.convert_key(key,is_press=1)
                    updates_button.insert(key,1)
                else:pass
            elif (key=='left' or 'right'or 'up' or 'down') &self.state[key] == 1:
                if updates_button[key]== 0:
                    self.convert_key(key,is_press=0)
                    updates_button.insert(key,0)
                else:pass

            elif self.state[key] == 0:
                if updates_button[key]== 1:
                    self.convert_key(key,is_press=1)
                    updates_button.insert(key,1)
                else:pass
            else:
                if updates_button[key]== 0:
                    self.convert_key(key,is_press=0)
                    updates_button.insert(key,0)
                else:pass



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
        elif key == 'w':
            key_pad = dp_controller.Button.R
        elif key == 'q':
            key_pad = dp_controller.Button.L
        elif key == 't':
            key_pad = dp_controller.Button.D_UP
        elif key == 'f':
            key_pad = dp_controller.Button.D_LEFT
        elif key == 'h':
            key_pad = dp_controller.Button.D_RIGHT


        if key_pad != dp_controller.Stick.MAIN:
            if is_press==True:
                dp_controller.DolphinController.press_button(key_pad)
            else:
                dp_controller.DolphinController.release_button(key_pad)
        else:
            if is_press==True:
                if key=='left':
                    dp_controller.DolphinController.set_stick(key_pad,-1,0.5)
                elif key=='right':
                    dp_controller.DolphinController.set_stick(key_pad, 1,0.5)
                elif key=='up':
                    dp_controller.DolphinController.set_stick(key_pad, 0.5,1)
                elif key=='down':
                    dp_controller.DolphinController.set_stick(key_pad, 0.5,-1)

            else:
                dp_controller.DolphinController.set_stick(key_pad, 0.5, 0.5)
