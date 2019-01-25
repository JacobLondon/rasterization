# https://www.youtube.com/watch?v=ih20l3pJoeU
'''
introduction to the projection matrix
'''
import numpy as np
from math import pi
import copy

from .mesh import *

class Graphics(object):

    def __init__(self, interface):
        self.interface = interface
        self.mesh_cube = Mesh()
        self.proj_matrix = np.zeros(shape=(4,4), dtype=float)
        self.theta = 0

        # load each triangle in a clockwise manner
        self.mesh_cube.triangles = [
            # south
            Triangle([Vector3(0,0,0), Vector3(0,1,0), Vector3(1,1,0)]),
            Triangle([Vector3(0,0,0), Vector3(1,1,0), Vector3(1,0,0)]),
            # east
            Triangle([Vector3(1,0,0), Vector3(1,1,0), Vector3(1,1,1)]),
            Triangle([Vector3(1,0,0), Vector3(1,1,1), Vector3(1,0,1)]),
            # north
            Triangle([Vector3(1,0,1), Vector3(1,1,1), Vector3(0,1,1)]),
            Triangle([Vector3(1,0,1), Vector3(0,1,1), Vector3(0,0,1)]),
            # west
            Triangle([Vector3(0,0,1), Vector3(0,1,1), Vector3(0,1,0)]),
            Triangle([Vector3(0,0,1), Vector3(0,1,0), Vector3(0,0,0)]),
            # top
            Triangle([Vector3(0,1,0), Vector3(0,1,1), Vector3(1,1,1)]),
            Triangle([Vector3(0,1,0), Vector3(1,1,1), Vector3(1,1,0)]),
            # bottom
            Triangle([Vector3(1,0,1), Vector3(0,0,1), Vector3(0,0,0)]),
            Triangle([Vector3(1,0,1), Vector3(0,0,0), Vector3(1,0,0)])
        ]

        # projection matrix
        near = 0.1
        far = 1000.0
        fov = 90
        aspect_ratio = self.interface.resolution[0] / self.interface.resolution[1]
        fov_radius = 1.0 / np.tan(fov * 0.5 * pi / 180.0)

        self.proj_matrix[0][0] = aspect_ratio * fov_radius
        self.proj_matrix[1][1] = fov_radius
        self.proj_matrix[2][2] = far / (far - near)
        self.proj_matrix[3][2] = (-1 * far * near) / (far - near)
        self.proj_matrix[2][3] = 1.0
        self.proj_matrix[3][3] = 0.0

    def vec_matmul(self, vin, vout, m):
        vout.x = vin.x * m[0][0] + vin.y * m[1][0] + vin.z * m[2][0] + m[3][0]
        vout.y = vin.x * m[0][1] + vin.y * m[1][1] + vin.z * m[2][1] + m[3][1]
        vout.z = vin.x * m[0][2] + vin.y * m[1][2] + vin.z * m[2][2] + m[3][2]
        w = vin.x * m[0][3] + vin.y * m[1][3] + vin.z * m[2][3] + m[3][3]

        if not w == 0:
            vout.x /= w
            vout.y /= w
            vout.z /= w

    def draw(self, delta_time):
        
        rotx_matrix = np.zeros(shape=(4,4), dtype=float)
        rotz_matrix = np.zeros(shape=(4,4), dtype=float)
        self.theta += delta_time / 2 % 4 * pi

        # rotate x
        rotx_scale = 0.5
        rotx_matrix[0][0] = 1
        rotx_matrix[1][1] = np.cos(self.theta * rotx_scale)
        rotx_matrix[1][2] = np.sin(self.theta * rotx_scale)
        rotx_matrix[2][1] = -1 * np.sin(self.theta * rotx_scale)
        rotx_matrix[2][2] = np.cos(self.theta * rotx_scale)
        rotx_matrix[3][3] = 1

        # rotate z
        rotz_matrix[0][0] = np.cos(self.theta)
        rotz_matrix[0][1] = np.sin(self.theta)
        rotz_matrix[1][0] = -1 * np.sin(self.theta)
        rotz_matrix[1][1] = np.cos(self.theta)
        rotz_matrix[2][2] = 1
        rotz_matrix[3][3] = 1

        # draw all triangles onto screen
        for triangle in self.mesh_cube.triangles:
            tri_projected = Triangle()
            tri_translated = Triangle()
            tri_rotated_z = Triangle()
            tri_rotated_zx = Triangle()
            
            # rotate about z axis
            self.vec_matmul(triangle[0], tri_rotated_z[0], rotz_matrix)
            self.vec_matmul(triangle[1], tri_rotated_z[1], rotz_matrix)
            self.vec_matmul(triangle[2], tri_rotated_z[2], rotz_matrix)

            # rotate about x axis
            self.vec_matmul(tri_rotated_z[0], tri_rotated_zx[0], rotx_matrix)
            self.vec_matmul(tri_rotated_z[1], tri_rotated_zx[1], rotx_matrix)
            self.vec_matmul(tri_rotated_z[2], tri_rotated_zx[2], rotx_matrix)

            # translate the triangle
            tri_translated = copy.deepcopy(tri_rotated_zx)
            tri_translated[0].z = tri_rotated_zx[0].z + 3
            tri_translated[1].z = tri_rotated_zx[1].z + 3
            tri_translated[2].z = tri_rotated_zx[2].z + 3

            # multiply each vector by the matrix
            self.vec_matmul(tri_translated[0], tri_projected[0], self.proj_matrix)
            self.vec_matmul(tri_translated[1], tri_projected[1], self.proj_matrix)
            self.vec_matmul(tri_translated[2], tri_projected[2], self.proj_matrix)

            # move across screen
            x_trans = 1
            y_trans = 1
            tri_projected[0].x += x_trans
            tri_projected[0].y += y_trans
            tri_projected[1].x += x_trans
            tri_projected[1].y += y_trans
            tri_projected[2].x += x_trans
            tri_projected[2].y += y_trans

            # scale by screen resolution
            w_scale = 0.5 * self.interface.resolution[0]
            h_scale = 0.5 * self.interface.resolution[1]
            tri_projected[0].x *= w_scale
            tri_projected[0].y *= h_scale
            tri_projected[1].x *= w_scale
            tri_projected[1].y *= h_scale
            tri_projected[2].x *= w_scale
            tri_projected[2].y *= h_scale

            # draw to screen
            coords = [tri_projected[0].x, tri_projected[0].y,
                      tri_projected[1].x, tri_projected[1].y,
                      tri_projected[2].x, tri_projected[2].y]
            
            self.interface.draw_triangle(*coords)

