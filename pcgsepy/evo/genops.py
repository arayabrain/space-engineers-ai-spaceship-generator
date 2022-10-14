import logging
import math
from random import choices, random, sample
import re
from typing import List, Tuple

import numpy as np
from pcgsepy.common.regex_handler import MyMatch, extract_regex
from pcgsepy.config import (CROSSOVER_P, MUTATION_DECAY, MUTATION_INITIAL_P,
                            PL_HIGH, PL_LOW)
from pcgsepy.lsystem.rules import StochasticRules
from pcgsepy.lsystem.solution import CandidateSolution, string_merging


class EvoException(Exception):
    pass


atoms_re = re.compile('corridor[a-z]*\(\d\)')


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
    fs = [cs.c_fitness if not minimize else 1 / (cs.c_fitness + 1e-6) for cs in pop]
    r = random() * sum(fs)
    p = 0.

    for i, f in enumerate(fs):
        p += f
        if p >= r:
            return pop[i]

    raise EvoException('Unable to find valid solution')


class SimplifiedExpander:
    def __init__(self):
        self.rules: StochasticRules = None
        self.compiled_lhs = None

    def initialize(self,
                   rules: StochasticRules):
        """Initialize the expander.

        Args:
            rules (StochasticRules): The set of expansion rules.
        """
        self.rules = rules
        self.compiled_lhs = [extract_regex(lhs) for lhs in rules.get_lhs()]


# module-scoped uninitialized variable
expander = SimplifiedExpander()


def mutate(cs: CandidateSolution,
           n_iteration: int) -> None:
    """Apply mutation to the parameters of the solution string.

    Args:
        cs (CandidateSolution): The solution to mutate.
        n_iteration (int): The current iteration number (used to compute decayed mutation probability).

    Raises:
        EvoException: Raised if no mutation can be applied to the candidate solution.
    """
    mutated = False
    for module in cs.hls_mod.keys():
        logging.getLogger('genops').debug(f'[{__name__}.mutate] {module=}; mutable={cs.hls_mod[module]["mutable"]}')
        if cs.hls_mod[module]['mutable']:
            # TODO: for enforcing symmetry automatically, should check if the atom to expand
            # is within a bracketed along the symmetry axis. If it is, mutate and copy the mutation
            # on the "next" bracketed enclosure.
            # Use get_matching_brackets to get all brackets and find which tuple contains the mutating atom.
            # example:
            # ...[RotYccwZ corridorsimple corridorsimple][RotYcwZ corridorsimple corridorsimple]...
            # ->
            # ...[RotYccwZ [RotYcwX corridorsimple] corridorsimple][RotYcwZ [RotYcwX corridorsimple] corridorsimple]...
            # get all matches with regex
            matches: List[MyMatch] = []
            for r, rule in zip(expander.compiled_lhs, expander.rules.get_lhs()):
                matches.extend([MyMatch(lhs=rule,
                                        span=match.span(),
                                        lhs_string=match.group()) for match in r.finditer(string=cs.hls_mod[module]['string'])])
            logging.getLogger('genops').debug(f'[{__name__}.mutate] {len(matches)=}')
            # sort matches in-place
            matches.sort()
            if matches:
                # filter out matches
                filtered_matches = [matches[0]]
                for match in matches:
                    logging.getLogger('genops').debug(f'[{__name__}.mutate] {match=}; {filtered_matches=}')
                    if match.start != filtered_matches[-1].start:
                        filtered_matches.append(match)
                        logging.getLogger('genops').debug(f'[{__name__}.mutate] {match=} appended')
                p = max(MUTATION_INITIAL_P / math.exp(n_iteration * MUTATION_DECAY), 0)
                to_mutate = math.ceil(p * len(filtered_matches))
                for_mutation = sample(population=filtered_matches,
                                      k=to_mutate)
                for_mutation = sorted(for_mutation)
                logging.getLogger('genops').debug(f'[{__name__}.mutate] {p=}; {to_mutate=} {len(for_mutation)=}')
                offset = 0
                for match in for_mutation:
                    rhs = expander.rules.get_rhs(lhs=match.lhs)
                    # update numerical parameters
                    if '(x)' in rhs or '(X)' in rhs or '(Y)' in rhs:
                        n = [m for m in re.compile(r'\d').finditer(match.lhs_string)]
                        n = int(n[0].group()) if n else None
                        # update rhs to include parameters
                        rhs = rhs.replace('(x)', f'({n})')
                        rhs_n = np.random.randint(PL_LOW, PL_HIGH)
                        rhs = rhs.replace('(X)', f'({rhs_n})')
                        if n is not None:
                            rhs = rhs.replace('(Y)', f'({max(1, n - rhs_n)})')
                    # apply expansion in string
                    cs.hls_mod[module]['string'] = cs.hls_mod[module]['string'][:match.start + offset] + rhs + cs.hls_mod[module]['string'][match.end + offset:]
                    offset += len(rhs) - len(match.lhs_string)
                    mutated |= True
    if not mutated:
        raise EvoException(f'No mutation could be applied to {cs.string}.')
    # Update solution string
    cs.string = string_merging(ls=[x['string'] for x in cs.hls_mod.values()])


def crossover(a1: CandidateSolution,
              a2: CandidateSolution,
              n_childs: int) -> List[CandidateSolution]:
    """Apply n-point crossover between two solutions.
    Note: may never terminate if `n_childs` is greater than the maximum number of possible offsprings.

    Args:
        a1 (CandidateSolution): The first solution.
        a2 (CandidateSolution): The second solution.
        n_childs (int): The number of offsprings to produce (suggested: 2).

    Raises:
        EvoException: Raised if no crossover could be applied to the candidate solutions.

    Returns:
        List[CandidateSolution]: A list containing the new offspring solutions.
    """
    childs = []
    for module in a1.hls_mod.keys():
        if a1.hls_mod[module]['mutable']:
            string1 = a1.hls_mod[module]['string'][:]
            string2 = a2.hls_mod[module]['string'][:]
            idxs1 = get_matching_brackets(string=string1)
            idxs2 = get_matching_brackets(string=string2)
            if not idxs1:
                idxs1 = [match.span() for match in atoms_re.finditer(string=string1)]
            if not idxs2:
                idxs2 = [match.span() for match in atoms_re.finditer(string=string2)]
            if len(idxs1) == 0 or len(idxs2) == 0:
                pass
            else:
                ws1 = [CROSSOVER_P for _ in range(len(idxs1))]
                ws2 = [CROSSOVER_P for _ in range(len(idxs2))]
                s1 = string1[:]
                s2 = string2[:]
                idx1 = choices(population=idxs1, weights=ws1, k=1)[0]
                b1 = string1[idx1[0]:idx1[1] + 1]
                idx2 = choices(population=idxs2, weights=ws2, k=1)[0]
                b2 = string2[idx2[0]:idx2[1] + 1]
                s1 = s1[:idx1[0]] + s1[idx1[0]:].replace(b1, b2, 1)
                s2 = s2[:idx2[0]] + s2[idx2[0]:].replace(b2, b1, 1)
                for solution, mutated in [(a1, s1), (a2, s2)]:
                    modified_hls_mod = dict(solution.hls_mod)
                    modified_hls_mod[module]['string'] = mutated
                    modified_string = string_merging([x['string'] for x in modified_hls_mod.values()])
                    o = CandidateSolution(string=modified_string)
                    o.hls_mod = modified_hls_mod
                    childs.append(o)
                    a1.n_offspring += 1
                    a2.n_offspring += 1
    if len(childs) == 0:
        raise EvoException(
            f'No cross-over could be applied ({a1.string} w/ {a2.string}).')
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