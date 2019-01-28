import pygame
from threading import Thread

from pyngine import Controller, Label, Grid, Anchor
from engine3d import Graphics1, Graphics2, Graphics3, Graphics4

class GraphicsController(Controller):

    def __init__(self, interface):
        Controller.__init__(self, interface)
        self.set_mouse_visible(False)
        self.center_mouse = True

    def initialize_components(self):
        self.layout = Grid(self.background_panel, 30, 30)
        self.fps_label = Label(self, str(self.fps))
        self.fps_label.loc = self.layout.get_pixel(0, 0)
        self.fps_label.background = None

        # position of camera
        self.pos_label = Label(self, '')
        self.pos_label.loc = self.layout.get_pixel(0, 1)
        self.pos_label.background = None

        self.yaw_label = Label(self, '')
        self.yaw_label.loc = self.layout.get_pixel(0, 2)
        self.yaw_label.background = None

    def setup(self):
        #self.graphics = Graphics1(self.interface)
        #self.graphics = Graphics2(self.interface)
        #self.graphics = Graphics3(self)
        self.graphics = Graphics4(self)

    def update_actions(self):
        self.fps_label.text = 'FPS ' + str(int(self.fps))
        self.fps_label.load()

        self.pos_label.text = 'Position: ' + str(self.graphics.camera)
        self.pos_label.load()

        self.yaw_label.text = 'Yaw: ' + str(self.graphics.yaw)
        self.yaw_label.load()

    def draw_midground(self):
        self.graphics.update()

    # allow mouse control
    def return_keydown(self):
        self.center_mouse = not self.center_mouse
        self.set_mouse_visible(not self.mouse_visible)
        self.key_presses[pygame.K_RETURN] = False
        self.graphics.turning = not self.graphics.turning

    # stop the program
    def escape_keydown(self):
        self.stop()
