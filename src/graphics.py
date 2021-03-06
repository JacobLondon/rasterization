import numpy as np
from math import pi
from collections import deque
import copy, time, pygame
from threading import Thread

from pyngine import Color, Painter
from .mesh import *

class Graphics(object):

    def __init__(self, controller):
        self.controller = controller
        self.interface = controller.interface
        self.speed = 10

        self.camera = vec()
        self.look_dir = vec()
        self.up_vec = vec(0, -1, 0)
        self.yaw = 0
        self.turning = True

        self.mesh_obj = Mesh()
        #self.theta = 0

        self.mesh_obj.load_obj('assets/axis.obj')

        self.painter = Painter(self.interface)

        # projection matrix
        near = 0.1
        far = 1000.0
        fov = 90
        aspect_ratio = self.interface.resolution[0] / self.interface.resolution[1]
        self.proj_matrix = m_projection(fov, aspect_ratio, near, far)

        self.triangles_to_raster = []

    def raster(self):
        for tri_to_raster in self.triangles_to_raster:
            # clip triangles against screen edges
            clipped = [Triangle(), Triangle()]
            triangles = deque([])
            # add initial triangle
            triangles.append(tri_to_raster)
            new_triangles = 1

            for p in range(4):
                tris_to_add = 0
                while new_triangles > 0:
                    test = triangles.popleft()
                    new_triangles -= 1

                    # top screen clip
                    if p == 0:
                        tris_to_add = t_clip_against_plane(vec(0, 0, 0), vec(0, 1, 0), test, clipped[0], clipped[1])
                    # bottom screen clip
                    elif p == 1:
                        tris_to_add = t_clip_against_plane(vec(0, self.interface.resolution[1] - 1, 0), vec(0, -1, 0), test, clipped[0], clipped[1])
                    # left screen clip
                    elif p == 2:
                        tris_to_add = t_clip_against_plane(vec(0, 0, 0), vec(1, 0, 0), test, clipped[0], clipped[1])
                    # right screen clip
                    elif p == 3:
                        tris_to_add = t_clip_against_plane(vec(self.interface.resolution[0] - 1, 0, 0), vec(-1, 0, 0), test, clipped[0], clipped[1])
                    
                    # add the new triangles to the back of the queue
                    for w in range(tris_to_add):
                        triangles.append(copy.deepcopy(clipped[w]))
                new_triangles = len(triangles)

            # draw transformed, viewed, clipped, projected, sorted triangles
            for t in triangles:
                coords = [t[0][0], t[0][1], t[1][0], t[1][1], t[2][0], t[2][1]]
                self.painter.fill_triangle(*coords, color=t.shade)
                self.painter.draw_triangle(*coords, color=Color['black'])

    def update(self):

        # f/b movement
        forward_vec = v_mul(self.look_dir, 8 * self.controller.delta_time)
        if self.controller.keyboard.presses[pygame.K_w]:
            self.camera = v_add(self.camera, forward_vec)
        if self.controller.keyboard.presses[pygame.K_s]:
            self.camera = v_sub(self.camera, forward_vec)

        # l/r/u/d movement
        if self.controller.keyboard.presses[pygame.K_SPACE]:
            self.camera[1] += self.speed * self.controller.delta_time
        if self.controller.keyboard.presses[pygame.K_LSHIFT]:
            self.camera[1] -= self.speed * self.controller.delta_time
        
        right_vec = v_cross(self.look_dir, self.up_vec)
        right_vec = v_mul(right_vec, 8 * self.controller.delta_time)
        if self.controller.keyboard.presses[pygame.K_a]:
            self.camera = v_add(self.camera, right_vec)
        if self.controller.keyboard.presses[pygame.K_d]:
            self.camera = v_sub(self.camera, right_vec)

        # l/r mouse movement
        if self.turning:
            self.yaw = self.controller.mouse.yaw

        if self.controller.keyboard.presses[pygame.K_LEFT]:
            self.yaw -= .1
        if self.controller.keyboard.presses[pygame.K_RIGHT]:
            self.yaw += .1

        self.triangles_to_raster.clear()
        
        rotz_matrix = m_rotate_z(0)
        rotx_matrix = m_rotate_x(0)
        trans_matrix = m_translation(0, 0, 5)
        
        # transform by rotation
        world_matrix = np.matmul(rotz_matrix, rotx_matrix)
        # transform by translation
        world_matrix = np.matmul(world_matrix, trans_matrix)
        
        # set up camera looking vectors
        target_vec = vec(0, 0, 1)
        rotcamera_matrix = m_rotate_y(self.yaw)
        self.look_dir = v_matmul(rotcamera_matrix, target_vec)
        target_vec = v_add(self.camera, self.look_dir)

        camera_matrix = m_point_at(self.camera, target_vec, self.up_vec)
        view_matrix = m_quick_inverse(camera_matrix)

        # draw all triangles onto screen
        for triangle in self.mesh_obj.triangles:
            tri_projected = Triangle()
            tri_transformed = Triangle()
            tri_viewed = Triangle()
            
            tri_transformed[0] = v_matmul(world_matrix, triangle[0])
            tri_transformed[1] = v_matmul(world_matrix, triangle[1])
            tri_transformed[2] = v_matmul(world_matrix, triangle[2])

            # get normal to cull triangles w/ normals pointing away from camera
            line1 = v_sub(tri_transformed[1], tri_transformed[0])
            line2 = v_sub(tri_transformed[2], tri_transformed[0])
            # take xproduct to get normal to triangle surface
            normal = v_cross(line1, line2)
            # normalize
            normal = v_normalize(normal)

            # get ray from triangle to camera
            camera_ray = v_sub(tri_transformed[0], self.camera)

            # dot product to see z direction relative to camera
            if v_dot(normal, camera_ray) < 0:
                
                # illumination
                light = vec(1, 1, -1)
                light = v_normalize(light)
                # light dot product
                light_dp = max(0.1, v_dot(light, normal))
                # set grayscale color based on the dot product
                grayscale = abs(int(255 * light_dp))
                tri_transformed.shade = (grayscale, grayscale, grayscale)
                
                # convert world space to view space
                tri_viewed[0] = v_matmul(view_matrix, tri_transformed[0])
                tri_viewed[1] = v_matmul(view_matrix, tri_transformed[1])
                tri_viewed[2] = v_matmul(view_matrix, tri_transformed[2])
                tri_viewed.shade = tri_transformed.shade
                
                clipped_triangles = 0
                clipped = [Triangle(), Triangle()]
                clipped_triangles = t_clip_against_plane(vec(0, 0, .1), vec(0, 0, 1), tri_viewed, clipped[0], clipped[1])
                
                for n in range(clipped_triangles):

                    # project triangles from 3D to 2D
                    tri_projected[0] = v_matmul(self.proj_matrix, clipped[n][0])
                    tri_projected[1] = v_matmul(self.proj_matrix, clipped[n][1])
                    tri_projected[2] = v_matmul(self.proj_matrix, clipped[n][2])
                    tri_projected.shade = clipped[n].shade
                    # manually normalize projection matrix
                    tri_projected[0] = v_div(tri_projected[0], tri_projected[0][3])
                    tri_projected[1] = v_div(tri_projected[1], tri_projected[1][3])
                    tri_projected[2] = v_div(tri_projected[2], tri_projected[2][3])
					
                    # offset vertices into visible normalized space
                    offset_view = vec(1, 1, 0)
                    tri_projected[0] = v_add(tri_projected[0], offset_view)
                    tri_projected[1] = v_add(tri_projected[1], offset_view)
                    tri_projected[2] = v_add(tri_projected[2], offset_view)
                    # scale by screen resolution
                    w_scale = 0.5 * self.interface.resolution[0]
                    h_scale = 0.5 * self.interface.resolution[1]
                    tri_projected[0][0] *= w_scale
                    tri_projected[0][1] *= h_scale
                    tri_projected[1][0] *= w_scale
                    tri_projected[1][1] *= h_scale
                    tri_projected[2][0] *= w_scale
                    tri_projected[2][1] *= h_scale

                    # store triangle for sorting, draw tris back to front
                    self.triangles_to_raster.append(copy.deepcopy(tri_projected))


        #print('calc time:', time.time() - start)
        # sort triangles from back to front (sort by avg of z values of tri)
        self.triangles_to_raster.sort(key=lambda t: (t[0][2] + t[1][2] + t[2][2]) / 3.0, reverse=True)
        self.raster()

        
