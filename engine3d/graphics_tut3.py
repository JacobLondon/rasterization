# https://www.youtube.com/watch?v=HXSuNxpCzdM
'''
cameras and clipping

'''
import numpy as np
from math import pi
import copy, time, pygame

from pyngine import Color
from .mesh import *

class Graphics(object):

    def __init__(self, controller):
        self.controller = controller
        self.interface = controller.interface
        self.speed = 10

        self.camera = Vector3()
        self.look_dir = Vector3()
        self.yaw = 0

        self.mesh = Mesh()
        self.theta = 0

        self.mesh.load_obj('assets/axis.obj')

        # projection matrix
        near = 0.1
        far = 1000.0
        fov = 90
        aspect_ratio = self.interface.resolution[0] / self.interface.resolution[1]
        self.proj_matrix = Matrix.projection(fov, aspect_ratio, near, far)

    def update(self):

        # l/r/u/d movement
        if self.controller.key_presses[pygame.K_SPACE]:
            self.camera.y += self.speed * self.controller.delta_time
        if self.controller.key_presses[pygame.K_LSHIFT]:
            self.camera.y -= self.speed * self.controller.delta_time
            
        if self.controller.key_presses[pygame.K_a]:
            self.camera.x += self.speed * self.controller.delta_time
        if self.controller.key_presses[pygame.K_d]:
            self.camera.x -= self.speed * self.controller.delta_time

        # f/b movement
        forward_vec = Vector3.mul(self.look_dir, 8 * self.controller.delta_time)
        if self.controller.key_presses[pygame.K_w]:
            self.camera = Vector3.add(self.camera, forward_vec)
        if self.controller.key_presses[pygame.K_s]:
            self.camera = Vector3.sub(self.camera, forward_vec)

        # l/r mouse movement
        if self.controller.delta_x > 2:
            self.yaw -= self.controller.delta_x * self.controller.delta_time

        # rotation
        #self.theta += self.controller.delta_time / 4 % 4 * pi
        rotz_matrix = Matrix.rotate_z(self.theta * 0.5)
        rotx_matrix = Matrix.rotate_x(self.theta)

        trans_matrix = Matrix.translation(0, 0, 5)
        world_matrix = Matrix.identity()
        # transform by rotation
        world_matrix = Matrix.matmul(rotz_matrix, rotx_matrix)
        # transform by translation
        world_matrix = Matrix.matmul(world_matrix, trans_matrix)

        # set up camera looking vectors
        up_vec = Vector3(0, -1, 0)
        target_vec = Vector3(0, 0, 1)
        rotcamera_matrix = Matrix.rotate_y(self.yaw)
        self.look_dir = Vector3.vmatmul(rotcamera_matrix, target_vec)
        target_vec = Vector3.add(self.camera, self.look_dir)

        camera_matrix = Matrix.point_at(self.camera, target_vec, up_vec)
        view_matrix = Matrix.quick_inverse(camera_matrix)

        triangles_to_raster = []

        # draw all triangles onto screen
        tri_transformed = Triangle()
        for triangle in self.mesh.triangles:
            tri_projected = Triangle()
            tri_viewed = Triangle()
            tri_transformed[0] = Vector3.vmatmul(world_matrix, triangle[0])
            tri_transformed[1] = Vector3.vmatmul(world_matrix, triangle[1])
            tri_transformed[2] = Vector3.vmatmul(world_matrix, triangle[2])

            # get normal to cull triangles w/ normals pointing away from camera
            line1 = Vector3.sub(tri_transformed[1], tri_transformed[0])
            line2 = Vector3.sub(tri_transformed[2], tri_transformed[0])
            # take xproduct to get normal to triangle surface
            normal = Vector3.cross(line1, line2)
            # normalize
            normal = Vector3.normalize(normal)

            # get ray from triangle to camera
            camera_ray = Vector3.sub(tri_transformed[0], self.camera)

            # dot product to see z direction relative to camera
            if Vector3.dot(normal, camera_ray) < 0:

                # illumination
                light = Vector3(1, 1, -1)
                light = Vector3.normalize(light)
                # light dot product
                light_dp = max(0.1, Vector3.dot(light, normal))
                # set grayscale color based on the dot product
                grayscale = abs(int(255 * light_dp))
                tri_transformed.shade = (grayscale, grayscale, grayscale)
                
                # convert world space to view space
                tri_viewed[0] = Vector3.vmatmul(view_matrix, tri_transformed[0])
                tri_viewed[1] = Vector3.vmatmul(view_matrix, tri_transformed[1])
                tri_viewed[2] = Vector3.vmatmul(view_matrix, tri_transformed[2])

                # project triangles from 3D to 2D
                tri_projected[0] = Vector3.vmatmul(self.proj_matrix, tri_viewed[0])
                tri_projected[1] = Vector3.vmatmul(self.proj_matrix, tri_viewed[1])
                tri_projected[2] = Vector3.vmatmul(self.proj_matrix, tri_viewed[2])
                tri_projected.shade = tri_transformed.shade
                # manually normalize projection matrix
                tri_projected[0] = Vector3.div(tri_projected[0], tri_projected[0].w)
                tri_projected[1] = Vector3.div(tri_projected[1], tri_projected[1].w)
                tri_projected[2] = Vector3.div(tri_projected[2], tri_projected[2].w)

                '''# x/y are inverted, flip back
                tri_projected[0].x *= -1
                tri_projected[1].x *= -1
                tri_projected[2].x *= -1
                tri_projected[0].y *= -1
                tri_projected[1].y *= -1
                tri_projected[2].y *= -1'''

                # move across screen
                offset_view = Vector3(1, 1, 0)
                tri_projected[0] = Vector3.add(tri_projected[0], offset_view)
                tri_projected[1] = Vector3.add(tri_projected[1], offset_view)
                tri_projected[2] = Vector3.add(tri_projected[2], offset_view)
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
            self.interface.fill_triangle(*coords, tri_projected.shade)
            # wireframe
            #self.interface.draw_triangle(*coords, Color.black)

