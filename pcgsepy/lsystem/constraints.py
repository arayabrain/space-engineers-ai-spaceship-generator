from enum import Enum, auto
from Geometry3D import *
import itertools
from typing import Callable

from ..common.vecs import Vec


class ConstraintLevel(Enum):
    SOFT_CONSTRAINT = auto()
    HARD_CONSTRAINT = auto()


class ConstraintTime(Enum):
    DURING = auto()
    END = auto()


class ConstraintHandler:
    def __init__(self,
                 name: str,
                 level: ConstraintLevel,
                 when: ConstraintTime,
                 f: Callable[[str], bool]):
        self.name = name
        self.level = level
        self.when = when
        self.constraint = f

    def __repr__(self) -> str:
        return f'Constraint {self.name} ({self.level.name}) at {self.when.name}'


class HLStructure:
    def __init__(self):
        self.polygons = []
        self.intersections = []

    def add_hl_poly(self,
                    p: ConvexPolyhedron) -> None:
        self.polygons.append(p)

    def test_intersections(self) -> bool:
        for p1, p2 in list(itertools.combinations(self.polygons, 2)):
            i = intersection(p1, p2)
            # faces on the same plane are ok, we don't want intersections
            if i is not None and type(i) == ConvexPolyhedron:
                self.intersections.append(i)
        return len(self.intersections) > 0

    def show(self) -> None:
        r = Renderer()
        for p in self.polygons:
            r.add((p, 'b', 1),
                  normal_length=0)
        for i in self.intersections:
            r.add((i, 'r', 3),
                  normal_length=0)
        r.show()


def build_polyhedron(position: Vec,
                     dims: Vec) -> ConvexPolyhedron:
    dim_x, dim_y, dim_z = dims.as_tuple()
    x0, y0, z0 = position.as_tuple()
    base = ConvexPolygon((Point(x0, y0, z0),
                         Point(x0+dim_x, y0, z0),
                         Point(x0+dim_x, y0+dim_y, z0),
                         Point(x0, y0+dim_y, z0)))
    top = ConvexPolygon((Point(x0, y0, z0+dim_z),
                        Point(x0+dim_x, y0, z0+dim_z),
                        Point(x0+dim_x, y0+dim_y, z0+dim_z),
                        Point(x0, y0+dim_y, z0+dim_z)))
    l1 = ConvexPolygon((Point(x0, y0, z0),
                       Point(x0+dim_x, y0, z0),
                       Point(x0+dim_x, y0, z0+dim_z),
                       Point(x0, y0, z0+dim_z)))
    l2 = ConvexPolygon((Point(x0+dim_x, y0, z0),
                       Point(x0+dim_x, y0+dim_y, z0),
                       Point(x0+dim_x, y0+dim_y, z0+dim_z),
                       Point(x0+dim_x, y0, z0+dim_z)))
    l3 = ConvexPolygon((Point(x0, y0+dim_y, z0),
                       Point(x0+dim_x, y0+dim_y, z0),
                       Point(x0+dim_x, y0+dim_y, z0+dim_z),
                       Point(x0, y0+dim_y, z0+dim_z)))
    l4 = ConvexPolygon((Point(x0, y0, z0),
                       Point(x0, y0+dim_y, z0),
                       Point(x0, y0+dim_y, z0+dim_z),
                       Point(x0, y0, z0+dim_z)))
    return ConvexPolyhedron((base, l1, l2, l3, l4, top))
