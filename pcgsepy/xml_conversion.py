import xml.etree.ElementTree as ET

from .structure import *

to_orientation = {
    'Up': Orientation.UP,
    'Down': Orientation.DOWN,
    'Left': Orientation.LEFT,
    'Right': Orientation.RIGHT,
    'Forward': Orientation.FORWARD,
    'Backward': Orientation.BACKWARD
}

grid_enum_to_offset = {
    'Small': 1,
    'Normal': 2,
    'Large': 5
}

def convert_xml_to_structure(root_node: ET.Element,
                             struct_dim: int = 100) -> Structure:
    structure = Structure(origin=Vec.v3f(0., 0., 0.),
                            orientation_forward=Orientation.FORWARD.value,
                            orientation_up=Orientation.UP.value,
                            dimensions=(struct_dim, struct_dim, struct_dim))
    for grid in root_node.findall('.//CubeGrid'):
        grid_size = None
        for child in grid:
            if child.tag == 'GridSizeEnum':
                grid_size = grid_enum_to_offset[child.text]
            elif child.tag == 'CubeBlocks':
                for block_node in child:  #  CubeBlocks node
                    if block_node.tag == 'MyObjectBuilder_CubeBlock':
                        block_type = ''
                        position = (0, 0, 0)
                        orientations = {'Forward': Orientation.FORWARD, 'Up': Orientation.UP}  # to change
                        for p in block_node:
                            if p.tag == 'SubtypeName':
                                block_type = p.text
                            elif p.tag == 'Min':
                                position = (grid_size * int(p.attrib['x']),
                                            grid_size * int(p.attrib['y']),
                                            grid_size * int(p.attrib['z']))
                            elif p.tag == 'BlockOrientation':
                                orientations = {
                                    'Forward': to_orientation[p.attrib['Forward']],
                                    'Up': to_orientation[p.attrib['Up']]
                                }
                        if not block_type:
                            continue
                    block = Block(block_type=block_type,
                                orientation_forward=orientations['Forward'],
                                orientation_up=orientations['Up'])
                    structure.add_block(block=block,
                                        grid_position=((struct_dim // 2) + position[0],
                                                       (struct_dim // 2) + position[1],
                                                       (struct_dim // 2) + position[2]))
    return structure
