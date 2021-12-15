from random import randint
from typing import List

from pcgsepy.lsystem.parser import axiom_to_tree, HLtoMLTranslator
from pcgsepy.config import MUTATION_DEFAULT_N, MUTATION_HIGH, MUTATION_LOW


def mutate(axiom: str,
           translator: HLtoMLTranslator,
           to_mutate: int = MUTATION_DEFAULT_N) -> str:
    """
    Apply mutation to the parameters of the axiom.

    Parameters
    ----------
    axiom : str
        The axiom to mutate.
    translator : HLtoMLTranslator
        The `HLtoMLTranslator` used to convert the axiom to a `TreeNode`.
    to_mutate : int
        The number of nodes to apply mutation to (default: *see config file*)

    Returns
    -------
    str
        The mutate axiom.
    """
    r = axiom_to_tree(axiom=axiom,
                      translator=translator)

    for_mutation = []
    while len(for_mutation) < to_mutate:
        node = r.pick_random_subnode(has_n=True)
        if node is not None and node not in for_mutation:
            for_mutation.append(node)

    for node in for_mutation:
        new_param = node.param + randint(MUTATION_LOW, MUTATION_HIGH+1)
        node.param = new_param if new_param > 0 else 1

    return str(r)


def crossover(a1: str,
              a2: str,
              n_childs: int,
              translator: HLtoMLTranslator,) -> List[str]:
    """
    Apply crossover between two axioms.
    Note: may never terminate if `n_childs` is greater than the maximum number of possible offsprings.

    Parameters
    ----------
    a1 : str
        The first axiom.
    a2 : str
        The second axiom.
    n_childs : int
        The number of offsprings to produce (suggested: 2).
    translator : HLtoMLTranslator
        The `HLtoMLTranslator` used to convert the axiom to a `TreeNode`.

    Returns
    -------
    List[str]
        A list containing the new offspring axioms.
    """
    a1 = axiom_to_tree(axiom=a1,
                       translator=translator)
    a2 = axiom_to_tree(axiom=a2,
                       translator=translator)
    childs = []
    while len(childs) < n_childs:
        s1 = str(a1)
        s2 = str(a2)

        b1, b2 = None, None
        while b1 is None or b2 is None:
            if b1:
                b2 = a2.pick_random_subnode()
            elif b2:
                b1 = a1.pick_random_subnode()
            else:
                b1 = a1.pick_random_subnode()
                b2 = a2.pick_random_subnode()

        s1 = s1.replace(str(b1), str(b2))
        s2 = s2.replace(str(b2), str(b1))

        childs.append(s1)
        childs.append(s2)

    return childs[:n_childs]