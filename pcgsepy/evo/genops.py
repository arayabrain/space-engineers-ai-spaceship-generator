import math
from random import random, randint, choices, sample
from typing import List, Tuple

from pcgsepy.config import CROSSOVER_P, MUTATION_INITIAL_P, MUTATION_DECAY
from pcgsepy.config import MUTATION_HIGH, MUTATION_LOW
from pcgsepy.config import PL_LOW, PL_HIGH


def roulette_wheel_selection(axioms: List[str],
                             fitnesses: List[float],
                             minimize: bool = False) -> str:
    """
    Apply roulette-wheel (fitness proportional) selection with fixed point
    trick.

    Parameters
    ----------
    axioms: List[str]
        The list of axioms from which select from.
    fitnesses : List[float]
        The list of fitness values associated to each axiom
    minimize : bool
        Flag to determine whether to pick least-fit or fittest axioms

    Returns
    -------
    str
        The selected axiom.
    """
    if not minimize:
        fs = fitnesses
    else:
        fs = [1 / f for f in fitnesses]
    s = sum(fs)
    r = s * random()
    p = 0.
    for i, f in enumerate(fs):
        p += f
        if p >= r:
            return axioms[i]
    raise Exception('Unable to find axiom')


def mutate(axiom: str,
           n_iteration: int) -> str:
    """
    Apply mutation to the parameters of the axiom.

    Parameters
    ----------
    axiom : str
        The axiom to mutate.
    n_iteration : int
        The current iteration number (used to compute decayed mutation
        probability).

    Returns
    -------
    str
        The mutate axiom.
    """
    idxs = get_atom_indexes(axiom=axiom,
                            atom='corridor')  # TODO: Perhaps read from config instead
    n = len(idxs)
    p = max(MUTATION_INITIAL_P / math.exp(n_iteration * MUTATION_DECAY), 0)
    to_mutate = int(p * n)

    for_mutation = sample(population=idxs, k=to_mutate)

    for idx in for_mutation:
        curr_param = int(axiom[axiom.find('(', idx[0]) + 1:idx[1]])
        new_param = max(PL_LOW, min(curr_param + randint(MUTATION_LOW, MUTATION_HIGH + 1), PL_HIGH))
        axiom = axiom[:axiom.find('(', idx[0]) + 1] + str(new_param) + axiom[idx[1]:]

    return axiom


def crossover(a1: str,
              a2: str,
              n_childs: int) -> List[str]:
    """
    Apply crossover between two axioms.
    Note: may never terminate if `n_childs` is greater than the maximum number
    of possible offsprings.

    Parameters
    ----------
    a1 : str
        The first axiom.
    a2 : str
        The second axiom.
    n_childs : int
        The number of offsprings to produce (suggested: 2).

    Returns
    -------
    List[str]
        A list containing the new offspring axioms.
    """
    idxs1 = get_matching_brackets(axiom=a1)
    idxs2 = get_matching_brackets(axiom=a2)
    # if crossover can't be applied, use atoms
    if not idxs1:
        idxs1 = get_atom_indexes(axiom=a1,
                                 atom='corridor')  # TODO: Perhaps read from config instead
    if not idxs2:
        idxs2 = get_atom_indexes(axiom=a2,
                                 atom='corridor')  # TODO: Perhaps read from config instead
    ws1 = [CROSSOVER_P for _ in range(len(idxs1))]
    ws2 = [CROSSOVER_P for _ in range(len(idxs2))]
    childs = []
    while len(childs) < n_childs:
        s1 = a1[:]
        s2 = a2[:]

        idx1 = choices(population=idxs1, weights=ws1, k=1)[0]
        b1 = a1[idx1[0]:idx1[1] + 1]

        idx2 = choices(population=idxs2, weights=ws2, k=1)[0]
        b2 = a2[idx2[0]:idx2[1] + 1]

        s1 = s1[:idx1[0]] + s1[idx1[0]:].replace(b1, b2, 1)
        s2 = s2[:idx2[0]] + s2[idx2[0]:].replace(b2, b1, 1)

        childs.append(s1)
        childs.append(s2)
    return childs[:n_childs]


# TODO: These methods should be moved elsewhere since they're geric enough
def get_matching_brackets(axiom: str) -> List[Tuple[int, int]]:
    brackets = []
    for i, c in enumerate(axiom):
        if c == '[':
            # find first closing bracket
            idx_c = axiom.index(']', i)
            # update closing bracket position in case of nested brackets
            ni_o = axiom.find('[', i + 1)
            while ni_o != -1 and axiom.find('[', ni_o) < idx_c:
                idx_c = axiom.index(']', idx_c + 1)
                ni_o = axiom.find('[', ni_o + 1)
            # add to list of brackets
            brackets.append((i, idx_c))
    return brackets


# TODO: These methods should be moved elsewhere since they're geric enough
def get_atom_indexes(axiom: str,
                     atom: str) -> List[Tuple[int, int]]:
    indexes = []
    for i, _ in enumerate(axiom):
        if axiom[i:].startswith(atom):
            cb = axiom.find(')', i + len(atom))
            indexes.append((i, cb))
            i = cb
    return indexes