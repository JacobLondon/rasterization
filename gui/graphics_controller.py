import pygame

from pyngine import Controller, Label, Relative, Anchor
from engine3d import Graphics1, Graphics2, Graphics3

class GraphicsController(Controller):

    def __init__(self, interface):
        Controller.__init__(self, interface)
        self.set_mouse_visible(False)
        self.center_mouse = True

    def initialize_components(self):
        self.layout = Relative(self.background_panel)
        self.fps_label = Label(self, str(self.fps))
        self.fps_label.background = None

        # position of camera
        self.pos_label = Label(self, '')
        self.pos_label.loc = self.layout.northeast
        self.pos_label.anchor = Anchor.northeast
        self.pos_label.background = None

    def setup(self):
        #self.graphics = Graphics1(self.interface)
        #self.graphics = Graphics2(self.interface)
        self.graphics = Graphics3(self)

    def update_actions(self):
        self.fps_label.text = 'FPS ' + str(int(self.fps))
        self.fps_label.load()

        self.pos_label.text = 'Position: ' + str(self.graphics.camera)
        self.pos_label.load()

    def draw_midground(self):
        self.graphics.update()

    # allow mouse control
    def return_keydown(self):
        self.center_mouse = not self.center_mouse
        self.set_mouse_visible(not self.mouse_visible)
        self.key_presses[pygame.K_RETURN] = False

    # stop the program
    def escape_keydown(self):
        self.stop()
