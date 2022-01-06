import os
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Tuple

from .common.vecs import Orientation, orientation_from_vec, Vec
from .structure import Block, Structure

to_orientation = {
    'Up': Orientation.UP,
    'Down': Orientation.DOWN,
    'Left': Orientation.LEFT,
    'Right': Orientation.RIGHT,
    'Forward': Orientation.FORWARD,
    'Backward': Orientation.BACKWARD
}

orientations_str = {
    Orientation.FORWARD: 'F',
    Orientation.BACKWARD: 'B',
    Orientation.RIGHT: 'R',
    Orientation.LEFT: 'L',
    Orientation.UP: 'U',
    Orientation.DOWN: 'D',
}

grid_enum_to_offset = {'Small': 1, 'Normal': 2, 'Large': 5}


def convert_xml_to_structure(root_node: ET.Element,
                             struct_dim: int = 100) -> Structure:
    """
    Convert the XML-defined structure to a `Structure` object.

    Parameters
    ----------
    root_node : ET.Element
        The XML root node.
    struct_dim : int
        The dimension of the Structure.

    Returns
    -------
    Structure
        The converted Structure object.
    """
    structure = Structure(origin=Vec.v3f(0., 0., 0.),
                          orientation_forward=Orientation.FORWARD.value,
                          orientation_up=Orientation.UP.value)
    for grid in root_node.findall('.//CubeGrid'):
        grid_size = None
        for child in grid:
            if child.tag == 'GridSizeEnum':
                grid_size = grid_enum_to_offset[child.text]
            elif child.tag == 'CubeBlocks':
                for block_node in child:  # CubeBlocks node
                    if block_node.tag.startswith('MyObjectBuilder_'):
                        block_type = ''
                        position = (0, 0, 0)
                        orientations = {
                            'Forward': Orientation.FORWARD,
                            'Up': Orientation.UP
                        }
                        for p in block_node:
                            if p.tag == 'SubtypeName':
                                block_type = p.text
                            elif p.tag == 'Min':
                                position = (grid_size * int(p.attrib['x']),
                                            grid_size * int(p.attrib['y']),
                                            grid_size * int(p.attrib['z']))
                            elif p.tag == 'BlockOrientation':
                                orientations = {
                                    'Forward':
                                        to_orientation[p.attrib['Forward']],
                                    'Up':
                                        to_orientation[p.attrib['Up']]
                                }
                        if not block_type:
                            continue
                        block = Block(
                            block_type=block_type,
                            orientation_forward=orientations['Forward'],
                            orientation_up=orientations['Up'])
                        structure.add_block(block=block, grid_position=position)
    return structure


def _at_same_x(x: int,
               blocks: List[Block]) -> List[Block]:
    r = []
    for b in blocks:
        if b.position.x == x:
            r.append(b)
    return r


def _at_same_y(y: int,
               blocks: List[Block]) -> List[Block]:
    r = []
    for b in blocks:
        if b.position.y == y:
            r.append(b)
    return r


def _at_same_z(z: int,
               blocks: List[Block]) -> List[Block]:
    r = []
    for b in blocks:
        if b.position.z == z:
            r.append(b)
    return r


# TODO: refactor/reduce complexity & return offset
def extract_rule(bp_dir: str,
                 title: str = '') -> Tuple[str, Tuple[int, int, int]]:
    bp = os.path.join(bp_dir, 'bp.sbc')
    root = ET.parse(bp).getroot()
    structure = convert_xml_to_structure(root_node=root)
    structure.sanify()
    structure.show(title)
    blocks = structure.get_all_blocks()

    max_x, max_y, max_z = 0., 0., 0.
    for block in blocks:
        x, y, z = block.position.as_tuple()
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y
        if z > max_z:
            max_z = z

    min_x, min_y, min_z = max_x, max_y, max_z
    for block in blocks:
        x, y, z = block.position.as_tuple()
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if z < min_z:
            min_z = z

    for block in blocks:
        new_p = block.position.sum(Vec.v3f(-min_x, -min_y, -min_z))
        block.position = Vec.v3f(x=np.round(new_p.x, 1),
                                 y=np.round(new_p.y, 1),
                                 z=np.round(new_p.z, 1))

    max_x -= min_x
    max_y -= min_y
    max_z -= min_z

    ordered_blocks = []
    x, y, z = 0., 0., 0.
    while z <= max_z:
        bs1 = _at_same_z(z, blocks)
        while y <= max_y:
            bs2 = _at_same_y(y, bs1)
            while x <= max_x:
                b = _at_same_x(x, bs2)
                if b:
                    ordered_blocks.append(b[0])
                x += 0.5
            x = 0.
            y += 0.5
        x = 0.
        y = 0.
        z += 0.5

    rule = ''
    x, y, z = 0., 0., 0.
    for block in ordered_blocks:
        if block.position.z != z:
            if block.position.z > z:
                dz = block.position.z - z
                rule += f'<({int(dz // 0.5)})'
                z = block.position.z
            else:
                dz = z - block.position.z
                rule += f'>({int(dz // 0.5)})'
                z = block.position.z
        if block.position.y != y:
            if block.position.y > y:
                dy = block.position.y - y
                rule += f'!({int(dy // 0.5)})'
                y = block.position.y
            else:
                dy = y - block.position.y
                rule += f'?({int(dy // 0.5)})'
                y = block.position.y
        if block.position.x != x:
            if block.position.x > x:
                dx = block.position.x - x
                rule += f'+({int(dx // 0.5)})'
                x = block.position.x
            else:
                dx = x - block.position.x
                rule += f'-({int(dx // 0.5)})'
                x = block.position.x
        of = orientations_str[orientation_from_vec(block.orientation_forward)]
        ou = orientations_str[orientation_from_vec(block.orientation_up)]
        rule += f'{block.block_type}({of},{ou})'

    if x != 0.:
        if x > 0.:
            rule += f'-({int(x // 0.5)})'
        if x < 0.:
            rule += f'+({int(-x // 0.5)})'
    if y != 0.:
        if y > 0.:
            rule += f'?({int(y // 0.5)})'
        if y < 0.:
            rule += f'!({int(-y // 0.5)})'
    if z != 0.:
        if z > 0.:
            rule += f'>({int(z // 0.5)})'
        if z < 0.:
            rule += f'<({int(-z // 0.5)})'

    x, y, z = structure._max_dims
    # TODO: change this from the grid size
    x += 5
    y += 5
    z += 5

    return rule, (x, y, z)