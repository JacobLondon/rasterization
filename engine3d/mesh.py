import os.path

from pyngine import Color

class Vector3(object):

    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ')'

class Triangle(object):

    def __init__(self, vectors=None):
        if vectors is None:
            p0 = Vector3()
            p1 = Vector3()
            p2 = Vector3()
            self.p = [p0, p1, p2]
        else:
            self.p = [vectors[0], vectors[1], vectors[2]]

        self.color = Color.white

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