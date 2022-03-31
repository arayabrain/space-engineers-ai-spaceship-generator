from dataclasses import replace
import math
from random import choices, sample, uniform
from tqdm.notebook import trange
from typing import Any, Dict, List, Optional, Tuple

from pcgsepy.config import (CROSSOVER_P, MUTATION_DECAY, MUTATION_INITIAL_P, N_GENS, N_RETRIES,
                            POP_SIZE)
from pcgsepy.evo.fitness import Fitness
from pcgsepy.evo.genops import EvoException, get_atom_indexes, get_matching_brackets, roulette_wheel_selection
from pcgsepy.fi2pop.utils import reduce_population, subdivide_solutions
from pcgsepy.lsystem.actions import AtomAction
from pcgsepy.lsystem.constraints import ConstraintLevel
from pcgsepy.lsystem.lsystem import LSystem
from pcgsepy.lsystem.parser import LLParser
from pcgsepy.lsystem.solution import CandidateSolution


class LGPSolver:
    def __init__(self,
                 alphabet: Dict[str, Any],
                 feasible_fitnesses: List[Fitness],
                 lsystem: LSystem):
        self.alphabet = alphabet.copy()
        
        # remove unneeded atoms from alphabet
        self.alphabet.pop('RotXcwY')
        self.alphabet.pop('RotXcwZ')
        self.alphabet.pop('RotXccwY')
        self.alphabet.pop('RotXccwZ')
        self.alphabet.pop('RotZcwX')
        self.alphabet.pop('RotZcwY')
        self.alphabet.pop('RotZccwX')
        self.alphabet.pop('RotZccwY')
        self.alphabet.pop('+')
        self.alphabet.pop('-')
        self.alphabet.pop('!')
        self.alphabet.pop('?')
        self.alphabet.pop('<')
        self.alphabet.pop('>')
        
        self.feasible_fitnesses = feasible_fitnesses
        self.lsystem = lsystem
        self.starting_string = 'cockpit(1)corridorsimple(1)thrusters(1)'
        
        self.inner_lgp_iterations = 5
        self.mutations = {
            self.__mutation: 0.5,
            self.__insertion: 0.4,
            self.__deletion: 0.1
        }
        self.ftop = []
        self.itop = []
        self.fmean = []
        self.imean = []

        self.ffs, self.ifs = [], []

        # number of total soft constraints
        self.nsc = [c for c in self.lsystem.all_hl_constraints if c.level == ConstraintLevel.SOFT_CONSTRAINT]
        self.nsc = [c for c in self.lsystem.all_ll_constraints if c.level == ConstraintLevel.SOFT_CONSTRAINT]
        self.nsc = len(self.nsc)

    def reset(self):
        self.ftop = []
        self.itop = []
        self.fmean = []
        self.imean = []
        self.ffs, self.ifs = [], []
    
    def _compute_fitness(self,
                         cs: CandidateSolution,
                         extra_args: Dict[str, Any]) -> List[float]:
        """Compute the fitness of a single candidate solution.

        Args:
            cs (CandidateSolution): The candidate solution.
            extra_args (Dict[str, Any]): Additional arguments used in the fitness function.

        Returns:
            float: The fitness value.
        """
        return [f(cs, extra_args) for f in self.feasible_fitnesses]
    
    def __mutation(self,
                   i: int,
                   s: List[str]):
        """Mutate the atom parameters in `s` at index `i`. Mutation only occurs for movements or blocks.

        Args:
            i (int): The index of the atom.
            s (List[str]): The list of atoms.
        """
        # get the atom
        a = s[i]
        # if tile, change the repetition amount or block type
        if self.alphabet[a['atom']]['action'] == AtomAction.PLACE:
            if uniform(a=0, b=1) > 0.5:
                pop = [-1, 0, 1] if int(a['n']) > 1 else [0, 1]
                a['n'] = str(int(a['n']) + choices(population=pop)[0])
            else:
                new_atom = ''
                placeable = False
                while not placeable:
                    new_atom = choices(population=list(self.alphabet.keys()))[0]
                    placeable = self.alphabet[new_atom]['action'] == AtomAction.PLACE
                a['atom'] = new_atom
    
    def _brackets_check(self,
                        s: Dict[str, Any]) -> int:
        """Do a stack simulation to check for `[` and `]` inconsistencies.

        Args:
            s (Dict[str, Any]): The list of atoms.

        Returns:
            int: The number of open brackets left.
        """
        stack = 0  # simulate stack push-pop
        for c in s:
            if c['atom'] == '[':
                stack += 1
            elif c['atom'] == ']':
                stack -= 1
            # if at any point we have more `]` than `[`, exit
            if stack < 0:
                return stack
        return stack
    
    def _interval_indexes(self,
                          i: int,
                          s: List[Dict[str, Any]],
                          c1: str = '[',
                          c2: str = ']') -> Optional[Tuple[int, int]]:
        """Get the indexes between `[` and `]` near `i` (if they exist).

        Args:
            i (int): The index.
            s (List[Dict[str, Any]]): The list of atoms.

        Returns:
            Optional[Tuple[int, int]]: The indexes of the surrounding atoms (if they exist).
        """
        res = None
        # get all occurrences' index before and after `i` in `s`
        ibs = [idx for idx, c in enumerate(s[:i]) if c['atom'] == c1]
        ias = [idx for idx, c in enumerate(s[i:]) if c['atom'] == c2]
        # if possible, get surrounding indexes
        if ibs != [] and ias != []:
            ib, ia = 0, 0
            # get closest index before `i`
            for idx in ibs:
                if idx < i:
                    ib = idx
                else:
                    break
            # get closest index after `i`
            for idx in ias:
                if idx > ib and idx > i:
                    ia = idx
                else:
                    break
            # assign result
            res = (ib, ia)
        return res

    def __insertion(self,
                   i: int,
                   s: List[Dict[str, Any]]):
        """Insert a new atom in the list. Additional checks are done for `]` atoms.

        Args:
            i (int): The index to insert a new atom at.
            s (List[Dict[str, Any]]): The list of atoms.
        """
        # pick a random new atom
        a = choices(population=list(self.alphabet.keys()))[0]
        # set parameters if it accepts them
        n = '0'
        if self.alphabet[a]['action'] == AtomAction.PLACE:
            n = '1'
        # In order to have a valid structure, the `]` must be placed only if possible, otherwise don't add anything
        # if a == ']':
            # if i > 0:
            #     # check brackets before `i`
            #     stack = self._brackets_check(s=s[:i])
            #     # simulate adding the new `]`
            #     stack -= 1
            #     if stack >= 0:
            #         # check brackets after `i` (in case they now are invalid)
            #         stack += self._brackets_check(s=s[i:])
            #         # if we have 1 or more `[`, we can add `]`
            #         if stack >= 0:
            #             s.insert(i + 1,
            #                     {'atom': a,
            #                     'n': '0'})
        # only add `[` with correct syntax
        if self.alphabet[s[i]['atom']]['action'] == AtomAction.PLACE:
            if a == '[':
                if not s[i - 1]['atom'].startswith('Rot') or not s[i - 1]['atom'] == '[':
                    s.insert(i + 1,
                            {'atom': a,
                            'n': '0'})
                    s.insert(i + 2,
                            {'atom': choices(population=[r for r in self.alphabet.keys() if r.startswith('Rot')])[0],
                            'n': '0'})
                    s.insert(i + 3,
                            {'atom': choices(population=[r for r in self.alphabet.keys() if r.startswith('corridor')])[0],
                            'n': '1'})
                    s.insert(i + 4,
                            {'atom': ']',
                            'n': '0'})
            elif a != ']' and not a.startswith('Rot'):
                # freely add the new atom
                s.insert(i + 1,
                        {'atom': a,
                        'n': n})
    
    def __deletion(self,
                   i: int,
                   s: List[str]):
        """Remove the atom at index `i` from the list of atoms.
        Only tiles and rotations can be removed from the list.

        Args:
            i (int): The index to insert a new atom at.
            s (List[str]): The list of atoms.
        """
        if self.alphabet[s[i]['atom']]['action'] == AtomAction.PLACE:
            # idxs = self._interval_indexes(i=i,
            #                               s=s)
            # if idxs is not None:
            #     ib, ia = idxs
            #     fas = sum([1 if self.alphabet[x['atom']]['action'] == AtomAction.PLACE else 0 for x in s[ib:ia]])
            # else:
            #     fas = sum([1 if self.alphabet[x['atom']]['action'] == AtomAction.PLACE else 0 for x in s])
            # if fas > 2 and i != 0:
            #     s.pop(i)
            if i > 0:
                if not self.alphabet[s[i - 1]['atom']]['action'] == AtomAction.ROTATE:
                    s.pop(i)
        elif self.alphabet[s[i]['atom']]['action'] == AtomAction.PUSH:
            n_o = 0
            for j, a in enumerate(s[i+1:]):
                if a['atom'] == '[':
                    n_o += 1
                elif a['atom'] == ']':
                    if n_o != 0:
                        n_o -= 1
                    else:
                        for k in range(i + j + 1, i - 1, -1):
                            s.pop(k)
                        break
    
    def _string_as_list(self,
                        string: str) -> List[Dict[str, Any]]:
        """Convert a string into a list of atoms.

        Args:
            string (str): The string.

        Returns:
            List[Dict[str, Any]]: The list of atoms (atom string & parameters).
        """
        atoms_list = []
        i = 0
        while i < len(string):
            offset = 0
            for k in self.alphabet.keys():
                if string[i:].startswith(k):
                    offset += len(k)
                    n = '0'
                    # check if there are parameters (multiplicity)
                    if i + offset < len(string) and string[i + offset] == '(':
                        params = string[i + offset:
                                        string.index(')', i + offset + 1) + 1]
                        offset += len(params)
                        n = params.replace('(', '').replace(')', '')
                    atoms_list.append({'atom': k,
                                       'n': n})
                    i += len(k) + (len(f'({n})') if n != '0' else 0) - 1
                    break
            i += 1
        return atoms_list

    def _list_as_string(self,
                        s: List[Dict[str, Any]]) -> str:
        """Convert a list of atoms to string.

        Args:
            s (List[Dict[str, Any]]): The list of atoms (string & parameters).

        Returns:
            str: The string.
        """
        return ''.join([a['atom'] + (f'({a["n"]})' if a["n"] != '0' else '') for a in s])
    
    def mutate(self,
               cs: CandidateSolution,
               n_iteration: int = 0) -> None:
        """Apply mutation to a given solution.

        Args:
            cs (CandidateSolution): The solution to mutate.
            n_iteration (int, optional): The current evolution iteration. Defaults to 0.
        """
        # we only mutate the string
        s = self._string_as_list(string=cs.string)
        # select the atoms to mutate
        p = max(MUTATION_INITIAL_P / math.exp(n_iteration * MUTATION_DECAY), 0)
        to_mutate = math.ceil(p * len(s))
        for_mutation = sample(population=list(range(len(s))),
                              k=to_mutate)
        # reverse to keep order in case of addition or deletion
        for_mutation.sort(reverse=True)
        # mutate each atom
        for atom in for_mutation:
            # select possible mutation
            f = choices(population=list(self.mutations.keys()),
                        weights=self.mutations.values())[0]
            # apply mutation
            f(i=atom,
              s=s)
        # assign new mutated low-level string to solution
        cs.string = self._list_as_string(s=s)
    
    def crossover(self,
                  cs1: CandidateSolution,
                  cs2: CandidateSolution,
                  n_childs: int = 2) -> List[CandidateSolution]:
        """Applies 1-point crossover between two solutions.

        Args:
            cs1 (CandidateSolution): The first solution.
            cs2 (CandidateSolution): The second solution.
            n_childs (int, optional): The number of offsprings to create. Defaults to 2.

        Returns:
            List[CandidateSolution]: The offsprings.
        """
        idxs1 = get_matching_brackets(string=cs1.string)
        idxs2 = get_matching_brackets(string=cs2.string)
        # if crossover can't be applied, use atoms
        if not idxs1:
            for atom in [a for a in self.alphabet.keys() if self.alphabet[a]['action'] == AtomAction.PLACE]:
                idxs1.extend(get_atom_indexes(string=cs1.string,
                                              atom=atom))
        if not idxs2:
            for atom in [a for a in self.alphabet.keys() if self.alphabet[a]['action'] == AtomAction.PLACE]:
                idxs2.extend(get_atom_indexes(string=cs2.string,
                                              atom=atom))
        i = None
        for j, idx in enumerate(idxs1):
            if idx[0] == 0:
                i = j
        if i is not None:
            idxs1.remove(idxs1[i])
        i = None
        for j, idx in enumerate(idxs2):
            if idx[0] == 0:
                i = j
        if i is not None:
            idxs2.remove(idxs2[i])
        ws1 = [CROSSOVER_P for _ in range(len(idxs1))]
        ws2 = [CROSSOVER_P for _ in range(len(idxs2))]
        childs = []
        if len(idxs1) > 0 and len(idxs2) > 0:
            while len(childs) < n_childs:
                s1 = cs1.string[:]
                s2 = cs2.string[:]

                idx1 = choices(population=idxs1, weights=ws1, k=1)[0]
                b1 = cs1.string[idx1[0]:idx1[1] + 1]

                idx2 = choices(population=idxs2, weights=ws2, k=1)[0]
                b2 = cs2.string[idx2[0]:idx2[1] + 1]

                s1 = s1[:idx1[0]] + s1[idx1[0]:].replace(b1, b2, 1)
                s2 = s2[:idx2[0]] + s2[idx2[0]:].replace(b2, b1, 1)

                for s, idx in [(s1, idx1), (s2, idx2)]:
                    o = CandidateSolution(string=s)                   
                    childs.append(o)
        else:
            print(cs1.string, idxs1)
            print(cs2.string, idxs2)
            raise EvoException(f'No cross-over could be applied ({cs1.string} w/ {cs2.string}).')
        return childs[:n_childs]
        # # convert the strings to list of atoms
        # ls1 = self._string_as_list(cs1.string)
        # ls2 = self._string_as_list(cs2.string)
        # offsprings = set()
        # while len(offsprings) < n_childs:
        #     idxs1 = [i for i, x in enumerate(ls1) if self.alphabet[x['atom']]['action'] == AtomAction.PLACE]
        #     idxs2 = [i for i, x in enumerate(ls2) if self.alphabet[x['atom']]['action'] == AtomAction.PLACE]
        #     # pick an atom index for both parents
        #     idx1 = sample(population=list(range(len(idxs1))),
        #                   k=1)[0]
        #     idx2 = sample(population=list(range(len(idxs2))),
        #                   k=1)[0]
        #     # apply 1-point swap between parents to generate offsprings
        #     lo1 = ls1[:idx1] + ls2[idx2:]
        #     lo2 = ls2[:idx2] + ls1[idx1:]
            
        #     if lo1[0]['atom'] == '[' or lo2[0]['atom'] == '[':
        #         print(f'error: {cs1.string} and {cs2.string} ({idx1}, {idx2})')
        #         continue
            
        #     # check consistency requirements
        #     if self._brackets_check(s=lo1) < 0 or self._brackets_check(s=lo2) < 0:
        #         continue
        #     # create offsprings solutions
        #     o1 = CandidateSolution(string=self._list_as_string(s=lo1))
        #     offsprings.add(o1)
        #     o2 = CandidateSolution(string=self._list_as_string(s=lo2))
        #     offsprings.add(o2)
        # return list(offsprings)[:n_childs]
    
    def _generate_initial_populations(self,
                                      pops_size: int = POP_SIZE,
                                      n_retries: int = N_RETRIES) -> Tuple[List[CandidateSolution], List[CandidateSolution]]:
        """Generate initial populations.

        Args:
            pops_size (int, optional): The size of each population. Defaults to POP_SIZE.
            n_retries (int, optional): The number of initialization retries. Defaults to N_RETRIES.

        Returns:
            Tuple[List[CandidateSolution], List[CandidateSolution]]: The feasible and infeasible populations.
        """
        assert pops_size * 2 < n_retries, f'Invalid pops_size ({pops_size}) and n_retries ({n_retries}): pops_size must be at most half n_retries'
        fpop, ipop = set(), set()
        with trange(n_retries, desc='Initialization ') as iterations:
            for i in iterations:
                # create empty solution
                cs = CandidateSolution(string=self.starting_string)
                # apply mutations
                for _ in range(self.inner_lgp_iterations):
                    self.mutate(cs=cs)
                # determine feasibility
                subdivide_solutions(lcs=[cs],
                                    lsystem=self.lsystem)
                # assign to corresponding population
                if cs.is_feasible and len(fpop) < pops_size:
                    cs.fitness = self._compute_fitness(cs=cs,
                                                    extra_args={
                                                        'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                        })
                    cs.c_fitness = sum(cs.fitness) + (self.nsc - cs.ncv)
                    fpop.add(cs)
                elif not cs.is_feasible and len(ipop) < pops_size:
                    cs.c_fitness = cs.ncv
                    ipop.add(cs)
                iterations.set_postfix(ordered_dict={'fpop-size': f'{len(fpop)}/{pops_size}',
                                                     'ipop-size': f'{len(ipop)}/{pops_size}'},
                                       refresh=True)
                # break out if n_retries is reached or both populations are full
                if i == n_retries or (len(fpop) == pops_size and len(ipop) == pops_size):
                    break
        return list(fpop), list(ipop)
    
    def initialize(self,
                   pops_size: int = POP_SIZE,
                   n_retries: int = N_RETRIES) -> Tuple[List[CandidateSolution], List[CandidateSolution]]:
        """Initialize the solver by generating the initial populations.

        Returns:
            Tuple[List[CandidateSolution], List[CandidateSolution]]: The Feasible and Infeasible populations.
        """
        f_pop, i_pop = self._generate_initial_populations(pops_size=pops_size,
                                                          n_retries=n_retries)
        f_fitnesses = [cs.c_fitness for cs in f_pop]
        i_fitnesses = [cs.c_fitness for cs in i_pop]
        self.ftop.append(max(f_fitnesses))
        self.fmean.append(sum(f_fitnesses) / len(f_fitnesses))
        self.itop.append(min(i_fitnesses))
        self.imean.append(sum(i_fitnesses) / len(i_fitnesses))
        self.ffs.append([self.ftop[-1], self.fmean[-1]])
        self.ifs.append([self.itop[-1], self.imean[-1]])
        print(f'Created Feasible population of size {len(f_pop)}: t:{self.ftop[-1]};m:{self.fmean[-1]}')
        print(f'Created Infeasible population of size {len(i_pop)}: t:{self.itop[-1]};m:{self.imean[-1]}')
        return f_pop, i_pop
    
    def _create_new_pool(self,
                         population: List[CandidateSolution],
                         generation: int,
                         n_individuals: int = POP_SIZE,
                         minimize: bool = False) -> List[CandidateSolution]:
        """Create a new pool of solutions.

        Args:
            population (List[CandidateSolution]): Initial population of solutions.
            generation (int): Current generation number.
            n_individuals (int, optional): The number of individuals in the pool. Defaults to POP_SIZE.
            minimize (bool, optional): Whether to minimize or maximize the fitness. Defaults to False.

        Raises:
            EvoException: If same parent is picked twice for crossover.

        Returns:
            List[CandidateSolution]: The pool of new solutions.
        """
        pool = set()
        while len(pool) < n_individuals:
            # fitness-proportionate selection
            p1 = roulette_wheel_selection(pop=population,
                                          minimize=minimize)
            new_pop = []
            new_pop[:] = population[:]
            new_pop.remove(p1)
            p2 = roulette_wheel_selection(pop=new_pop,
                                          minimize=minimize)
            if p1 != p2:
                # crossover
                o1, o2 = self.crossover(cs1=p1, cs2=p2, n_childs=2)
                for o in [o1, o2]:
                    # mutation
                    self.mutate(cs=o, n_iteration=generation)
                    pool.add(o)
            else:
                raise EvoException('Picked same parents, this should never happen.')
        return list(pool)
    
    def fi2pop(self,
               f_pop: List[CandidateSolution],
               i_pop: List[CandidateSolution],
               n_iter: int = N_GENS) -> Tuple[List[CandidateSolution], List[CandidateSolution]]:
        """Apply the FI2Pop algorithm to the given populations for `n_iter` steps.

        Args:
            f_pop (List[CandidateSolution]): The Feasible population.
            i_pop (List[CandidateSolution]): The Infeasible population.
            n_iter (int, optional): The number of iterations to run for. Defaults to N_GENS.

        Returns:
            Tuple[List[CandidateSolution], List[CandidateSolution]]: The Feasible and the Infeasible populations.
        """
        f_pool = []
        i_pool = []
        with trange(n_iter, desc='Generation ') as gens:
            for gen in gens:
                # place the infeasible population in the infeasible pool
                i_pool.extend(i_pop)
                
                f_pool.extend(f_pop)
                
                # create offsprings from feasible population
                new_pool = self._create_new_pool(population=f_pop,
                                                 generation=gen)
                # if feasible, add to feasible pool
                # if infeasible, add to infeasible pool
                subdivide_solutions(lcs=new_pool,
                                    lsystem=self.lsystem)
                for cs in new_pool:
                    if cs.ll_string == '':
                        cs.ll_string = self.lsystem.hl_to_ll(cs=cs).string
                    if cs.is_feasible:
                        cs.fitness = self._compute_fitness(cs=cs,
                                                           extra_args={
                                                               'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                               })
                        cs.c_fitness = sum(cs.fitness) + (self.nsc - cs.ncv)
                        f_pool.append(cs)
                    else:
                        cs.c_fitness = cs.ncv
                        i_pool.append(cs)
                
                i_pool = list(set(i_pool))
                
                # reduce the infeasible pool if > pops_size
                if len(i_pool) > POP_SIZE:
                    i_pool = reduce_population(population=i_pool,
                                               to=POP_SIZE,
                                               minimize=True)
                # set the infeasible pool as the infeasible population
                i_pop[:] = i_pool[:]
                # create offsprings from infeasible population
                new_pool = self._create_new_pool(population=i_pop,
                                                 generation=gen,
                                                 minimize=True)
                # if feasible, add to feasible pool
                # if infeasible, add to infeasible pool
                subdivide_solutions(lcs=new_pool,
                                    lsystem=self.lsystem)
                for cs in new_pool:
                    if cs.is_feasible:
                        f_pool.append(cs)
                        cs.ll_string = self.lsystem.hl_to_ll(cs=cs).string
                        cs.fitness = self._compute_fitness(cs=cs,
                                                           extra_args={
                                                               'alphabet': self.lsystem.ll_solver.atoms_alphabet
                                                               })
                        cs.c_fitness = sum(cs.fitness) + (self.nsc - cs.ncv)
                    else:
                        cs.c_fitness = cs.ncv
                        i_pool.append(cs)
                
                f_pool = list(set(f_pool))        
                
                # reduce the feasible pool if > pops_size
                if len(f_pool) > POP_SIZE:
                    f_pool = reduce_population(population=f_pool,
                                               to=POP_SIZE)
                # set the feasible pool as the feasible population
                f_pop[:] = f_pool[:]
                # update tracking
                f_fitnesses = [cs.c_fitness for cs in f_pop]
                i_fitnesses = [cs.c_fitness for cs in i_pop]
                self.ftop.append(max(f_fitnesses))
                self.fmean.append(sum(f_fitnesses) / len(f_fitnesses))
                self.itop.append(min(i_fitnesses))
                self.imean.append(sum(i_fitnesses) / len(i_fitnesses))
                self.ffs.append([self.ftop[-1], self.fmean[-1]])
                self.ifs.append([self.itop[-1], self.imean[-1]])
                gens.set_postfix(ordered_dict={'top-f': self.ftop[-1],
                                               'mean-f': self.fmean[-1],
                                               'top-i': self.itop[-1],
                                               'mean-i': self.imean[-1]},
                                 refresh=True)

        return f_pop, i_pop
