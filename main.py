from gui import Graphics
from pyngine import Interface

resolution = (720,720)
gwidth = gheight = 20
refresh_rate = 60

def main():
    interface_args = ['3D Engine', resolution, gwidth, gheight, refresh_rate]
    interface = Interface(*interface_args)
    controller = Graphics(interface)
    controller.run()

if __name__ == '__main__':
    main()
