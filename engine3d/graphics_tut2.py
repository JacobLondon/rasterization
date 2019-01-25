# https://www.youtube.com/watch?v=XgMWc6LumG4
'''
normals, culling, lighting, and object files
Obj Files:
When saving an object in blender, check normals to be outwards
and save the obj with the following options:
    -Forward: Z Forward
    -triangulate faces
'''
import numpy as np
from math import pi
import copy

from pyngine import Color
from .mesh import *

class Graphics(object):

    def __init__(self, interface):
        self.interface = interface

        self.camera = Vector3()

        self.mesh_cube = Mesh()
        self.proj_matrix = np.zeros(shape=(4,4), dtype=float)
        self.theta = 0

        '''# load each triangle in a clockwise manner
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
        ]'''

        self.mesh_cube.load_obj('assets/ship.obj')

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

    @staticmethod
    def vec_matmul(vin, vout, m):
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
        self.theta += delta_time / 4 % 4 * pi

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

        triangles_to_raster = []

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
            distance = 10
            tri_translated[0].z = tri_rotated_zx[0].z + distance
            tri_translated[1].z = tri_rotated_zx[1].z + distance
            tri_translated[2].z = tri_rotated_zx[2].z + distance

            # get normal to cull triangles w/ normals pointing away from camera
            normal = Vector3()
            line1 = Vector3()
            line2 = Vector3()
            # calc line1 of triangle
            line1.x = tri_translated[1].x - tri_translated[0].x
            line1.y = tri_translated[1].y - tri_translated[0].y
            line1.z = tri_translated[1].z - tri_translated[0].z
            # calc line2 of triangle
            line2.x = tri_translated[2].x - tri_translated[0].x
            line2.y = tri_translated[2].y - tri_translated[0].y
            line2.z = tri_translated[2].z - tri_translated[0].z
            # calc the cross product of line1/2 to get normal of triangle
            normal.x = line1.y * line2.z - line1.z * line2.y
            normal.y = line1.z * line2.x - line1.x * line2.z
            normal.z = line1.x * line2.y - line1.y * line2.x
            # make the normal a unit vector
            norm_len = np.sqrt(normal.x ** 2 + normal.y ** 2 + normal.z ** 2)
            normal.x /= norm_len
            normal.y /= norm_len
            normal.z /= norm_len

            # if the normal (triangle face) is facing the camera
            #if normal.z < 0:
            # dot product to see z direction relative to camera
            if normal.x * (tri_translated[0].x - self.camera.x) + \
               normal.y * (tri_translated[0].y - self.camera.y) + \
               normal.z * (tri_translated[0].z - self.camera.z) < 0:

                # illumination
                light = Vector3(0.0, 0.0, -1.0)
                light_len = np.sqrt(light.x ** 2 + light.y ** 2 + light.z ** 2)
                # light unit vector
                light.x /= light_len
                light.y /= light_len
                light.z /= light_len
                # light dot product
                light_dp = normal.x * light.x + normal.y * light.y + normal.z * light.z
                # set grayscale color based on the dot product
                grayscale = abs(int(255 * light_dp))
                tri_translated.color = (grayscale, grayscale, grayscale)
                
                # project triangles from 3D to 2D
                self.vec_matmul(tri_translated[0], tri_projected[0], self.proj_matrix)
                self.vec_matmul(tri_translated[1], tri_projected[1], self.proj_matrix)
                self.vec_matmul(tri_translated[2], tri_projected[2], self.proj_matrix)
                tri_projected.color = tri_translated.color

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

                # store triangle for sorting, draw tris back to front
                triangles_to_raster.append(tri_projected)

        # sort triangles from back to front (sort by avg of z values of tri)
        triangles_to_raster.sort(key=lambda t: (t[0].z + t[1].z + t[2].z) / 3.0, reverse=True)

        # rasterize triangles
        for tri_projected in triangles_to_raster:

            coords = [tri_projected[0].x, tri_projected[0].y,
                    tri_projected[1].x, tri_projected[1].y,
                    tri_projected[2].x, tri_projected[2].y]
            # faces
            self.interface.fill_triangle(*coords, tri_projected.color)
            # wireframe
            #self.interface.draw_triangle(*coords, Color.black)
