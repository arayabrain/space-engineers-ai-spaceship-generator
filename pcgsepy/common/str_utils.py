from typing import List, Tuple


def get_atom_indexes(string: str,
                     atom: str) -> List[Tuple[int, int]]:
    """Get the indexes of the positions of the given atom in the string.
    Args:
        string (str): The string.
        atom (str): The atom.
    Returns:
        List[Tuple[int, int]]: The list of pair indexes.
    """
    indexes = []
    for i, _ in enumerate(string):
        if string[i:].startswith(atom):
            cb = string.find(')', i + len(atom))
            indexes.append((i, cb))
            i = cb
    return indexes


def get_matching_brackets(string: str) -> List[Tuple[int, int]]:
    """Get indexes of matching square brackets.

    Args:
        string (str): The string.

    Returns:
        List[Tuple[int, int]]: The list of pair indexes.
    """
    brackets = []
    for i, c in enumerate(string):
        if c == '[':
            # find first closing bracket
            idx_c = string.index(']', i)
            # update closing bracket position in case of nested brackets
            ni_o = string.find('[', i + 1)
            while ni_o != -1 and string.find('[', ni_o) < idx_c:
                idx_c = string.index(']', idx_c + 1)
                ni_o = string.find('[', ni_o + 1)
            # add to list of brackets
            brackets.append((i, idx_c))
    return brackets
