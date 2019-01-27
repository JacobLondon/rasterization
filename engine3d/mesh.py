import os.path, numpy as np
from math import pi

from pyngine import Color

class Vector3(object):

    def __init__(self, x=0, y=0, z=0, w=1):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __str__(self):
        return '(' + str(int(self.x)) + ', ' + str(int(self.y)) + ', ' + str(int(self.z)) + ')'

    def array3(self):
        return np.array([self.x, self.y, self.z])

    def array4(self):
        return np.array([self.x, self.y, self.z, self.w])

    @staticmethod
    def add(v1, v2):
        return Vector3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z)

    @staticmethod
    def sub(v1, v2):
        return Vector3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z)

    @staticmethod
    def mul(v1, k):
        return Vector3(v1.x * k, v1.y * k, v1.z * k)

    @staticmethod
    def div(v1, k):
        return Vector3(v1.x / k, v1.y / k, v1.z / k)

    @staticmethod
    def mag(v):
        return np.sqrt(np.dot(v.array3(), v.array3()))

    @staticmethod
    def dot(v1, v2):
        return np.dot(v1.array3(), v2.array3())

    @staticmethod
    def normalize(v):
        l = Vector3.mag(v)
        return Vector3(v.x / l, v.y / l, v.z / l)

    @staticmethod
    def cross(v1, v2):
        coords = np.cross(v1.array3(), v2.array3())
        return Vector3(*coords)

    @staticmethod
    def vmatmul(mat, vin):
        coords = np.matmul(vin.array4(), mat)
        return Vector3(*coords)

    @staticmethod
    def intersect_plane(plane_p, plane_n, line_start, line_end):
        plane_n = Vector3.normalize(plane_n)
        plane_d = -1 * Vector3.dot(plane_n, plane_p)
        ad = Vector3.dot(line_start, plane_n)
        bd = Vector3.dot(line_end, plane_n)
        t = (-1 * plane_d - ad) / (bd - ad)
        line_start_to_end = Vector3.sub(line_end, line_start)
        line_to_intersect = Vector3.mul(line_start_to_end, t)
        return Vector3.add(line_start, line_to_intersect)

class Matrix(object):

    @staticmethod
    def zeros():
        return np.zeros(shape=(4,4), dtype=float)

    @staticmethod
    def identity():
        return np.identity(4)

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
        new_forward = Vector3.sub(target, pos)
        new_forward = Vector3.normalize(new_forward)

        # calculate new up direction
        a = Vector3.mul(new_forward, Vector3.dot(up, new_forward))
        new_up = Vector3.sub(up, a)
        new_up = Vector3.normalize(new_up)

        # calculate new right direction
        new_right = Vector3.cross(new_up, new_forward)

        # construction dimensioning and translation matrix
        matrix = np.array([
            [new_right.x, new_right.y, new_right.z, 0],
            [new_up.x, new_up.y, new_up.z, 0],
            [new_forward.x, new_forward.y, new_forward.z, 0],
            [pos.x, pos.y, pos.z, 1]
        ])
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
        ])
        return matrix

class Triangle(object):

    def __init__(self, vectors=None):
        if vectors is None:
            p0 = Vector3()
            p1 = Vector3()
            p2 = Vector3()
            self.p = [p0, p1, p2]
        else:
            self.p = [vectors[0], vectors[1], vectors[2]]

        self.shade = Color.white

    def __getitem__(self, arg):
        return self.p[arg]

    def __setitem__(self, idx, value):
        self.p[idx] = value

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
                v = Vector3()
                v.x = float(line[1])
                v.y = float(line[2])
                v.z = float(line[3])
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