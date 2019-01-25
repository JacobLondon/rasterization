import pygame

from pyngine import Controller
from pyngine import Label
from engine3d import Graphics1, Graphics2

class GraphicsController(Controller):

    def __init__(self, interface):
        Controller.__init__(self, interface)

    def initialize_components(self):
        self.fps_label = Label(self, str(self.fps))

    def setup(self):
        #self.graphics = Graphics1(self.interface)
        self.graphics = Graphics2(self.interface)

    def update_actions(self):
        self.fps_label.text = 'FPS ' + str(int(self.fps))
        self.fps_label.load()

    def draw_midground(self):
        self.graphics.draw(self.delta_time)

    def escape_keydown(self):
        self.stop()
