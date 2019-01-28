import os.path, numpy as np, copy
from math import pi

import numba

from pyngine import Color

#@numba.jit(nopython=True)
def intersect_plane(plane_p, plane_n, line_start, line_end):
    # detecting a vector intersecting a plane
    plane_n = v_normalize(plane_n)
    plane_d = -1 * v_dot(plane_n, plane_p)
    ad = v_dot(line_start, plane_n)
    bd = v_dot(line_end, plane_n)
    t = (-1. * plane_d - ad) / (bd - ad)
    line_start_to_end = v_sub(line_end, line_start)
    line_to_intersect = v_mul(line_start_to_end, t)
    return v_add(line_start, line_to_intersect)

def vec(x=0, y=0, z=0, w=1):
    return np.array([x, y, z, w], dtype=float)

def v_add(v1, v2):
    return vec(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])

def v_sub(v1, v2):
    return vec(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])

def v_mul(v1, k):
    return vec(v1[0] * k, v1[1] * k, v1[2] * k)

def v_div(v1, k):
    return vec(v1[0] / k, v1[1] / k, v1[2] / k)

def v_mag(v):
    return np.sqrt(np.dot(v[:3], v[:3]))

def v_dot(v1, v2):
    return np.dot(v1[0:3], v2[:3])

def v_normalize(v):
    l = v_mag(v)
    return vec(v[0] / l, v[1] / l, v[2] / l)

def v_cross(v1, v2):
    coords = np.cross(v1[:3], v2[:3])
    return vec(*coords[:3])

def v_matmul(mat, vin):
    coords = np.matmul(vin, mat)
    return vec(*coords[:3])

class Matrix(object):

    @staticmethod
    def zeros():
        return np.zeros(shape=(4,4), dtype=float)

    @staticmethod
    def identity():
        return np.identity(4, dtype=float)

    @staticmethod
    def rotate_x(radians):
        matrix = Matrix.zeros()
        matrix[0][0] = 1.0
        matrix[1][1] = np.cos(radians)
        matrix[1][2] = np.sin(radians)
        matrix[2][1] = -1 * np.sin(radians)
        matrix[2][2] = np.cos(radians)
        matrix[3][3] = 1.0
        return matrix

    @staticmethod
    def rotate_y(radians):
        matrix = Matrix.zeros()
        matrix[0][0] = np.cos(radians)
        matrix[0][2] = np.sin(radians)
        matrix[2][0] = -1 * np.sin(radians)
        matrix[1][1] = 1.0
        matrix[2][2] = np.cos(radians)
        matrix[3][3] = 1.0
        return matrix

    @staticmethod
    def rotate_z(radians):
        matrix = Matrix.zeros()
        matrix[0][0] = np.cos(radians)
        matrix[0][1] = np.sin(radians)
        matrix[1][0] = -1 * np.sin(radians)
        matrix[1][1] = np.cos(radians)
        matrix[2][2] = 1.0
        matrix[3][3] = 1.0
        return matrix

    @staticmethod
    def translation(x, y, z):
        matrix = Matrix.identity()
        matrix[3][0] = x
        matrix[3][1] = y
        matrix[3][2] = z
        return matrix

    @staticmethod
    def projection(fov, aspect_ratio, near, far):
        fov_rad = 1.0 / np.tan(fov * 0.5 * pi / 180)
        matrix = Matrix.zeros()
        matrix[0][0] = aspect_ratio * fov_rad
        matrix[1][1] = fov_rad
        matrix[2][2] = far / (far - near)
        matrix[3][2] = (-1 * far * near) / (far - near)
        matrix[2][3] = 1.0
        matrix[3][3] = 0.0
        return matrix

    @staticmethod
    def matmul(m1, m2):
        return np.matmul(m1, m2)

    '''
    pos: vector where the object should be
    target: forward vector for that object
    up: up vector
    '''
    @staticmethod
    def point_at(pos, target, up):
        # calculate new forward direction
        new_forward = v_sub(target, pos)
        new_forward = v_normalize(new_forward)

        # calculate new up direction
        a = v_mul(new_forward, v_dot(up, new_forward))
        new_up = v_sub(up, a)
        new_up = v_normalize(new_up)

        # calculate new right direction
        new_right = v_cross(new_up, new_forward)

        # construction dimensioning and translation matrix
        matrix = np.array([
            [new_right[0], new_right[1], new_right[2], 0],
            [new_up[0], new_up[1], new_up[2], 0],
            [new_forward[0], new_forward[1], new_forward[2], 0],
            [pos[0], pos[1], pos[2], 1]
        ], dtype=float)
        return matrix

    @staticmethod
    def quick_inverse(m):
        matrix = np.array([
            [m[0][0], m[1][0], m[2][0], 0],
            [m[0][1], m[1][1], m[2][1], 0],
            [m[0][2], m[1][2], m[2][2], 0],
            [
                -1 * (m[3][0] * m[0][0] + m[3][1] * m[1][0] + m[3][2] * m[2][0]),
                -1 * (m[3][0] * m[0][1] + m[3][1] * m[1][1] + m[3][2] * m[2][1]),
                -1 * (m[3][0] * m[0][2] + m[3][1] * m[1][2] + m[3][2] * m[2][2]),
                1
            ]
        ], dtype=float)
        return matrix

class Triangle(object):

    def __init__(self, vectors=None):
        if vectors is None:
            p0 = vec()
            p1 = vec()
            p2 = vec()
            self.p = [p0, p1, p2]
        else:
            self.p = [vectors[0], vectors[1], vectors[2]]

        self.shade = Color.white

    def __str__(self):
        return str(self.p[0]) + str(self.p[1]) + str(self.p[2])

    def __getitem__(self, arg):
        return self.p[arg]

    def __setitem__(self, idx, value):
        self.p[idx] = value

    # returns the number of triangles which will be needed
    @staticmethod
    def clip_against_plane(plane_p, plane_n, in_tri, out_tri1, out_tri2):

        # make sure the plane is normal
        plane_n = v_normalize(plane_n)

        # return shortest distance from point to normalized plane
        def dist(p):
            n = v_normalize(p)
            return (plane_n[0] * p[0] + plane_n[1] * p[1] + plane_n[2] * p[2] - v_dot(plane_n, plane_p))

        # classify points either in or outside of a plane
        # distance is positive, point is on inside of plane
        inside_points = []
        inside_point_count = 0
        outside_points = []
        outside_point_count = 0

        # calculate the distance of each point in triangle to plane
        d0 = dist(in_tri[0])
        d1 = dist(in_tri[1])
        d2 = dist(in_tri[2])

        if d0 >= 0:
            inside_points.append(in_tri[0])
            inside_point_count += 1
        else:
            outside_points.append(in_tri[0])
            outside_point_count += 1
        if d1 >= 0:
            inside_points.append(in_tri[1])
            inside_point_count += 1
        else:
            outside_points.append(in_tri[1])
            outside_point_count += 1
        if d2 >= 0:
            inside_points.append(in_tri[2])
            inside_point_count += 1
        else:
            outside_points.append(in_tri[2])
            outside_point_count += 1

        # classify points
        if inside_point_count == 0:
            # all points outside of plane, so clip entire triangle
            return 0

        # all points inside of plane, let triangle pass thru
        if inside_point_count == 3:
            out_tri1.shade = in_tri.shade
            for i in range(3):
                out_tri1[i][0] = in_tri[i][0]
                out_tri1[i][1] = in_tri[i][1]
                out_tri1[i][2] = in_tri[i][2]
            return 1

        # triangle should be clipped to smaller triangle, two pts outside
        if inside_point_count == 1 and outside_point_count == 2:
            # copy appearance to new triangle
            out_tri1.shade = in_tri.shade
            #out_tri1.shade = Color.blue

            # inside pt is valid
            out_tri1[0] = inside_points[0]

            # two other points at the intersection of plane/triangle
            out_tri1[1] = intersect_plane(plane_p, plane_n, inside_points[0], outside_points[0])
            out_tri1[2] = intersect_plane(plane_p, plane_n, inside_points[0], outside_points[1])

            return 1

        # triangle should be clipped into quad, 1 pt outside
        if inside_point_count == 2 and outside_point_count == 1:
            # copy appearance to new triangles
            out_tri1.shade = in_tri.shade
            out_tri2.shade = in_tri.shade
            #out_tri1.shade = Color.green
            #out_tri2.shade = Color.red

            # first triangle made of two inside pts and a new point at intersection
            out_tri1[0] = inside_points[0]
            out_tri1[1] = inside_points[1]
            out_tri1[2] = intersect_plane(plane_p, plane_n, inside_points[0], outside_points[0])
            
            # second triangle made of one inside pt, previously created pt, and at intersection
            out_tri2[0] = inside_points[1]
            out_tri2[1] = out_tri1[2]
            out_tri2[2] = intersect_plane(plane_p, plane_n, inside_points[1], outside_points[0])            

            return 2

class Mesh(object):

    def __init__(self):
        self.triangles = []

    # load an obj file from a given path
    def load_obj(self, path):
        # file not found
        if not os.path.isfile(path):
            return False

        with open(path) as obj:
            lines = obj.readlines()

        # local cache of vertices
        vertices = []

        # traverse all lines
        for line in lines:
            # current line is a vertex
            if line[0] == 'v':
                line = line.split()
                v = vec()
                v[0] = float(line[1])
                v[1] = float(line[2])
                v[2] = float(line[3])
                vertices.append(v)

            # current line is a face
            if line[0] == 'f':
                line = line.split()
                # get the indices in vertices for the corresponding face
                f1 = int(line[1]) - 1
                f2 = int(line[2]) - 1
                f3 = int(line[3]) - 1
                # lookup the corresponding vertices for each triangle
                vectors = [vertices[f1], vertices[f2], vertices[f3]]
                tri = Triangle(vectors)
                self.triangles.append(tri)

        # loaded successfully
        return True