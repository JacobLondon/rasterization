import pygame
from threading import Thread

from pyngine import Controller, Label, Grid, Anchor, Drawer
from engine3d import Graphics1, Graphics2, Graphics3, Graphics4

class GraphicsController(Controller):

    def __init__(self, interface):
        Controller.__init__(self, interface, debug=False)
        self.mouse.set_visible(False)
        self.mouse.locked = True

        self.keyboard.return_keydown = self.toggle_mouse
        self.keyboard.escape_keydown = self.stop

    def initialize_components(self):
        self.layout = Grid(self.background_panel, 30, 30)
        self.fps_label = Label(self, str(self.fps), z=3000)
        self.fps_label.loc = self.layout.get_pixel(0, 0)
        self.fps_label.background = None

        # position of camera
        self.pos_label = Label(self, '', z=3001)
        self.pos_label.loc = self.layout.get_pixel(0, 1)
        self.pos_label.background = None

        self.yaw_label = Label(self, '', z=3002)
        self.yaw_label.loc = self.layout.get_pixel(0, 2)
        self.yaw_label.background = None

        #self.component_drawer.refresh = self.update_info
        self.graphics = Graphics4(self)
        self.component_drawer = Drawer(self, self.update_info)
        self.graphcis_drawer = Drawer(self, self.graphics.update)

    def setup(self):
        #self.graphics = Graphics1(self.interface)
        #self.graphics = Graphics2(self.interface)
        #self.graphics = Graphics3(self)
        pass

    def update_info(self):
        self.fps_label.text = 'FPS ' + str(round(self.fps))
        self.fps_label.load()

        self.pos_label.text = 'Position: ' + str(self.graphics.camera)
        self.pos_label.load()

        self.yaw_label.text = 'Yaw: ' + str(self.graphics.yaw)
        self.yaw_label.load()

    # allow mouse control
    def toggle_mouse(self):
        self.mouse.locked = not self.mouse.locked
        self.mouse.toggle_visibility()
        self.keyboard.presses[pygame.K_RETURN] = False
        self.graphics.turning = not self.graphics.turning

