import src.keylog as keylog
from src import dp_controller

class KeyPadMap:

    def __init__(self):
        self.previous_keys =dict((el.name, False) for el in keylog.Keyboard)


    def update(self, keys):
        # TODO track which keys are pressed and which are released.
        # TODO track 'toggled' MAIN stick positions as well.
            if self.previous_keys == {}:
                self.previous_keys = keys
                return
            for key in keys:
                # Ignore 'none'
                if key == 'none':
                    continue
                # Check for BUTTON
                if (key not in (('left','right', 'up', 'down')) ):
                    if keys[key]== True:
                        if self.previous_keys[key]== False:
                            self.convert_key(key,is_press=1)
                            self.previous_keys[key]=True
                    else:
                        if self.previous_keys[key]== True:
                            self.convert_key(key,is_press=0)
                            self.previous_keys[key]=False

                # Check for MAIN STICK
                elif keys[key] == False:
                    if self.previous_keys[key]== True:
                        self.convert_key(key,is_press=0)
                        self.previous_keys[key]=False

                else:
                    if self.previous_keys[key]== False:
                        self.convert_key(key,is_press=1)
                        self.previous_keys[key]=True


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

        # PRESS/RELEASE
        if key_pad != dp_controller.Stick.MAIN:
            with dp_controller.DolphinController("~/.dolphin-emu/Pipes/pipe") as p:
                if is_press==1:
                    p.press_button(key_pad)
                else:
                    p.release_button(key_pad)
        # SET STICK
        else:
            with dp_controller.DolphinController("~/.dolphin-emu/Pipes/pipe") as p:
                if is_press==1:
                    if key=='left':
                        p.set_stick(key_pad,x=0,y=0.5)
                    elif key=='right':
                        p.set_stick(key_pad, x=1,y=0.5)
                    elif key=='up':
                        p.set_stick(key_pad, x=0.5,y=1)
                    elif key=='down':
                        p.set_stick(key_pad, x=0.5,y=0)

                else:
                    p.set_stick(key_pad, x=0.5, y=0.5)
