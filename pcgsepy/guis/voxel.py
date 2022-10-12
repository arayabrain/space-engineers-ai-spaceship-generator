import numpy as np


# modified code based off https://github.com/olive004/Plotly-voxel-renderer
# TODO: improve readibility
class VoxelData():

    def __init__(self, data):
        self.data = data
        self.intensities = []
        self.triangles = np.zeros((np.size(np.shape(self.data)), 1))
        self.xyz = self.get_coords()
        self.x_length = np.size(data, 0)
        self.y_length = np.size(data, 1)
        self.z_length = np.size(data, 2)
        self.vert_count = 0
        self.vertices = self.make_edge_verts()
        self.triangles = np.delete(self.triangles, 0, 1)

    def get_coords(self):
        indices = np.nonzero(self.data)
        indices = np.stack((indices[0], indices[1], indices[2]))
        return indices

    def has_voxel(self, neighbor_coord):
        return self.data[neighbor_coord[0], neighbor_coord[1], neighbor_coord[2]]

    def get_neighbor(self, voxel_coords, direction):
        x = voxel_coords[0]
        y = voxel_coords[1]
        z = voxel_coords[2]
        offset_to_check = CubeData.offsets[direction]
        neighbor_coord = [x + offset_to_check[0], y +
                          offset_to_check[1], z+offset_to_check[2]]

        # return 0 if neighbor out of bounds or nonexistent
        if (any(np.less(neighbor_coord, 0)) | (neighbor_coord[0] >= self.x_length) | (neighbor_coord[1] >= self.y_length) | (neighbor_coord[2] >= self.z_length)):
            return 0
        else:
            return self.has_voxel(neighbor_coord)

    def make_face(self, voxel, direction):
        voxel_coords = self.xyz[:, voxel]
        explicit_dir = CubeData.direction[direction]
        vert_order = CubeData.face_triangles[explicit_dir]

        next_i = [self.vert_count, self.vert_count]
        next_j = [self.vert_count+1, self.vert_count+2]
        next_k = [self.vert_count+2, self.vert_count+3]

        next_tri = np.vstack((next_i, next_j, next_k))
        self.triangles = np.hstack((self.triangles, next_tri))

        face_verts = np.zeros((len(voxel_coords), len(vert_order)))
        for i in range(len(vert_order)):
            face_verts[:, i] = voxel_coords + \
                CubeData.cube_verts[vert_order[i]]

        self.vert_count = self.vert_count+4

        return face_verts

    def make_cube_verts(self, voxel):
        voxel_coords = self.xyz[:, voxel]
        cube = np.zeros((len(voxel_coords), 1))

        # only make a new face if there's no neighbor in that direction
        dirs_no_neighbor = []
        for direction in range(len(CubeData.direction)):
            if np.any(self.get_neighbor(voxel_coords, direction)):
                continue
            else:
                dirs_no_neighbor = np.append(dirs_no_neighbor, direction)
                face = self.make_face(voxel, direction)
                cube = np.append(cube, face, axis=1)

            m, n, p = voxel_coords
            self.intensities.extend([self.data[m, n, p]] * 2)

        # remove cube initialization
        cube = np.delete(cube, 0, 1)

        return cube

    def make_edge_verts(self):
        # make only outer vertices
        edge_verts = np.zeros((np.size(self.xyz, 0), 1))
        num_voxels = np.size(self.xyz, 1)
        for voxel in range(num_voxels):
            # passing voxel num rather than
            cube = self.make_cube_verts(voxel)
            edge_verts = np.append(edge_verts, cube, axis=1)
        edge_verts = np.delete(edge_verts, 0, 1)
        return edge_verts


class CubeData:
    # all data and knowledge from https://github.com/boardtobits/procedural-mesh-tutorial/blob/master/CubeMeshData.cs
    # for creating faces correctly by direction
    face_triangles = {
        'North':  [0, 1, 2, 3],        # +y
        'East': [5, 0, 3, 6],         # +x
        'South': [4, 5, 6, 7],        # -y
        'West': [1, 4, 7, 2],         # -x
        'Up': [5, 4, 1, 0],           # +z
        'Down': [3, 2, 7, 6]          # -z
    }

    cube_verts = [
        [1, 1, 1],
        [0, 1, 1],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 0, 0],
        [0, 0, 0],
    ]

    direction = [
        'North',
        'East',
        'South',
        'West',
        'Up',
        'Down'
    ]

    opposing_directions = [
        ['North', 'South'],
        ['East', 'West'],
        ['Up', 'Down']
    ]

    # xyz direction corresponding to 'Direction'
    offsets = [
        [0, 1, 0],
        [1, 0, 0],
        [0, -1, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, 0, -1],
    ]
