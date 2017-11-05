import pad
import logging
import time

with pad.Pad("~/.dolphin-emu/Pipes/pipe") as p:
    for i in range (1,13):
        p.prre_button(pad.Button.START)
        time.sleep(2)