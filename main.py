import cProfile

from gui import Graphics
from pyngine import Interface

resolution = (720,720)
gwidth = gheight = 20
refresh_rate = 200

def main():
    interface_args = ['3D Engine', resolution, gwidth, gheight, refresh_rate]
    interface = Interface(*interface_args)
    controller = Graphics(interface)

    '''pr = cProfile.Profile()
    pr.enable()'''
    controller.run()
    '''pr.disable()
    pr.print_stats(sort='time')'''

if __name__ == '__main__':
    main()
