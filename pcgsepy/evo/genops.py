import math
from itertools import accumulate
from random import choices, random, sample
from typing import List, Tuple

import numpy as np

from ..config import (CROSSOVER_P, MUTATION_DECAY, MUTATION_INITIAL_P, PL_HIGH,
                      PL_LOW)
from ..lsystem.rules import StochasticRules
from ..lsystem.solution import CandidateSolution


class EvoException(Exception):
    pass


def roulette_wheel_selection(pop: List[CandidateSolution],
                             minimize: bool = False) -> CandidateSolution:
    """Apply roulette-wheel (fitness proportional) selection with fixed point
    trick.

    Args:
        pop (List[CandidateSolution]): The list of solutions from which select from.
    Raises:
        Exception: If no solution is found.

    Returns:
        CandidateSolution: The individual solution.
    """
    fs = [cs.c_fitness if not minimize else 1/(cs.c_fitness + 1e-6) for cs in pop]
    # fs = [cs.c_fitness for cs in pop]
    s = sum(fs)
    r = s * random()
    
    p = 0.
    for i, f in enumerate(fs):
        p += f
        if p >= r:
            return pop[i]
    raise EvoException('Unable to find valid solution')


class SimplifiedExpander:
    def __init__(self):
        self.rules: StochasticRules = None

    def initialize(self,
                   rules: StochasticRules):
        self.rules = rules

    def get_rhs(self,
                string: str,
                idx: int) -> str:
        assert self.rules is not None, 'SimplifiedExpander has not been initialized'
        for k in self.rules.lhs_alphabet:
            if string[idx:].startswith(k):
                offset = len(k)
                lhs = k
                n = None
                # check if there are parameters
                if idx + offset < len(string) and string[idx + offset] == '(':
                    params = string[idx + offset:string.index(')', idx + offset + 1) + 1]
                    offset += len(params)
                    n = int(params.replace('(', '').replace(')', ''))
                    lhs += '(x)'
                    if idx + offset < len(string) and string[idx + offset] == ']':
                        lhs += ']'
                        offset += 1
                rhs = self.rules.get_rhs(lhs=lhs)
                if '(x)' in rhs or '(X)' in rhs or '(Y)' in rhs:
                    # update rhs to include parameters
                    rhs = rhs.replace('(x)', f'({n})')
                    rhs_n = np.random.randint(PL_LOW, PL_HIGH)
                    rhs = rhs.replace('(X)', f'({rhs_n})')
                    if n is not None:
                        rhs = rhs.replace('(Y)', f'({max(1, n - rhs_n)})')
                return rhs, offset


expander = SimplifiedExpander()


def mutate(cs: CandidateSolution,
           n_iteration: int) -> None:
    """Apply mutation to the parameters of the solution string.

    Args:
        cs (CandidateSolution): The olution to mutate.
        n_iteration (int): The current iteration number (used to compute decayed mutation
        probability).
    """
    idxs = get_atom_indexes(string=cs.string,
                            atom='corridorsimple')  # TODO: Perhaps read from config instead
    idxs = remove_immutable_idxs(cs=cs,
                                 idxs=idxs)
    n = len(idxs)
    if n > 0:
        p = max(MUTATION_INITIAL_P / math.exp(n_iteration * MUTATION_DECAY), 0)
        to_mutate = int(p * n)
        for_mutation = sample(population=idxs,
                              k=to_mutate)
        for_mutation = sorted(for_mutation,
                              key=lambda x: x[0],
                              reverse=True)

        for idx in for_mutation:
            rhs, offset = expander.get_rhs(string=cs.string,
                                           idx=idx[0])
            d = offset - (idx[1] + 1 - idx[0])
            cs.string = cs.string[:idx[0]] + rhs + cs.string[idx[1] + 1 + d:]
        # Update hls_mod 'string' values
        mod_intervals = list(accumulate([len(cs.hls_mod[k]['string']) for k in cs.hls_mod.keys()]))
        prev = 0
        for interval, k in zip(mod_intervals, cs.hls_mod.keys()):
            cs.hls_mod[k]['string'] = cs.string[prev:interval]
            prev += interval
    else:
        raise EvoException(f'No mutation could be applied to {cs.string}.')


def crossover(a1: CandidateSolution,
              a2: CandidateSolution,
              n_childs: int) -> List[CandidateSolution]:
    """Apply 2-point crossover between two solutions.
    Note: may never terminate if `n_childs` is greater than the maximum number
    of possible offsprings.

    Args:
        a1 (CandidateSolution): The first solution.
        a2 (CandidateSolution): The second solution.
        n_childs (int): The number of offsprings to produce (suggested: 2).

    Returns:
        List[CandidateSolution]: A list containing the new offspring solutions.
    """
    idxs1 = get_matching_brackets(string=a1.string)
    idxs1 = remove_immutable_idxs(cs=a1,
                                  idxs=idxs1)
    idxs2 = get_matching_brackets(string=a2.string)
    idxs2 = remove_immutable_idxs(cs=a2,
                                  idxs=idxs2)
    # if crossover can't be applied, use atoms
    if not idxs1:
        idxs1 = get_atom_indexes(string=a1.string,
                                 atom='corridor')  # TODO: Perhaps read from config instead
        idxs1 = remove_immutable_idxs(cs=a1,
                                    idxs=idxs1)
    if not idxs2:
        idxs2 = get_atom_indexes(string=a2.string,
                                 atom='corridor')  # TODO: Perhaps read from config instead
        idxs2 = remove_immutable_idxs(cs=a2,
                                    idxs=idxs2)
    ws1 = [CROSSOVER_P for _ in range(len(idxs1))]
    ws2 = [CROSSOVER_P for _ in range(len(idxs2))]
    childs = []
    if len(idxs1) > 0 and len(idxs2) > 0:
        while len(childs) < n_childs:
            s1 = a1.string[:]
            s2 = a2.string[:]

            idx1 = choices(population=idxs1, weights=ws1, k=1)[0]
            b1 = a1.string[idx1[0]:idx1[1] + 1]

            idx2 = choices(population=idxs2, weights=ws2, k=1)[0]
            b2 = a2.string[idx2[0]:idx2[1] + 1]

            s1 = s1[:idx1[0]] + s1[idx1[0]:].replace(b1, b2, 1)
            s2 = s2[:idx2[0]] + s2[idx2[0]:].replace(b2, b1, 1)

            for s, idx in [(s1, idx1), (s2, idx2)]:
                o = CandidateSolution(string=s)
                # Update hls_mod
                # mod_intervals = list(accumulate([len(a1.hls_mod[k]['string']) for k in a1.hls_mod.keys()]))
                # prev = 0
                # for interval, k in zip(mod_intervals, a1.hls_mod.keys()):
                #     o.hls_mod[k] = {
                #         'string': o.string[prev:interval],
                #         'mutable': a1.hls_mod[k]['mutable']
                #     }
                #     prev += interval
                # TODO: Figure out how to implement this correctly 😔
                for module in ['HeadModule', 'BodyModule', 'TailModule']:
                    s = 'cockpit' if module == 'HeadModule' else 'thrusters' if module == 'TailModule' else o.string.replace('cockpit', '').replace('thrusters', '')
                    o.hls_mod[module] = {
                        'string': s,
                        'mutable': a1.hls_mod[module]['mutable']
                    }
                
                childs.append(o)
                a1.n_offspring += 1
                a2.n_offspring += 1
    else:
        print(a1.string, a1.hls_mod, idxs1)
        print(a2.string, a2.hls_mod, idxs2)
        raise EvoException(f'No cross-over could be applied ({a1.string} w/ {a2.string}).')
    return childs[:n_childs]


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


def remove_immutable_idxs(cs: CandidateSolution,
                          idxs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Remove indexes of an immutable part of the string.

    Args:
        cs (CandidateSolution): The solution
        idxs (List[Tuple[int, int]]): The initial list of indexes

    Returns:
        List[Tuple[int, int]]: The filtered list of indexes
    """
    valid = []
    mod_intervals = list(accumulate(
        [len(cs.hls_mod[k]['string']) for k in cs.hls_mod.keys()]))
    for idx in idxs:
        for i, interval in enumerate(mod_intervals):
            if idx[0] < interval and (i == len(mod_intervals) - 1 or idx[0] < mod_intervals[i + 1]):
                if cs.hls_mod[list(cs.hls_mod.keys())[i]]['mutable']:
                    valid.append(idx)
                break
    return valid
