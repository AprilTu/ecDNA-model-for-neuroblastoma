"""
Simulates the dynamics of ecDNA copy number evolution under a sigmoid fitness function.

Author: Tutu
Date: 2024-09-22
"""

import numpy as np

# Constants
MAX_EC_COPIES = 5001
inx_ec_copies = np.arange(0, MAX_EC_COPIES, 1)


def sigmoid_fitness_function(ec_copy_num, max_s, k_e, k_m):
    """
    Calculate fitness using a sigmoid function based on ecDNA copy number.

    Args:
        ec_copy_num (ndarray): ecDNA copy numbers for which to compute fitness.
        max_s (float): Maximum selection coefficient (fitness).
        k_e (int): ecDNA copy number at which fitness reaches max_s.
        k_m (int): Parameter controlling sigmoid steepness.

    Returns:
        ndarray: Selection coefficients for each ecDNA copy number.
    """
    selection_coeff = 1 + ((max_s - 1) * (1 + ((k_e - ec_copy_num) /
                           (k_e - k_m))) * (ec_copy_num / k_e) ** (k_e / (k_e - k_m)))

    # If copy number exceeds k_e, cap fitness at max_s
    selection_coeff[ec_copy_num > k_e] = max_s

    return selection_coeff


def fix_time_model(s_max, k_e, k_m, k_0, t, p_s=0.5):
    """
    Simulates ecDNA dynamics over a fixed period of time starting from one cell.

    Args:
        s_max (float): Maximum selection coefficient.
        k_e (int): ecDNA copy number where fitness is maximal.
        k_m (int): Parameter controlling sigmoid steepness.
        k_0 (int): Initial ecDNA copy number of the cell.
        t (float): Simulation time.
        p_s (float, optional): Probability of ecDNA being passed to one daughter cell. Default is 0.5.

    Returns:
        tuple: (final cell count, total simulation time, ecDNA copy number distribution)
    """
    # Initialization
    ec_copies_list = inx_ec_copies
    fitness_list = sigmoid_fitness_function(ec_copies_list, s_max, k_e, k_m)
    copies_num_list = np.zeros(len(ec_copies_list), dtype=int)
    copies_num_list[k_0] = 1

    # Initialize simulation variables
    rng = np.random.default_rng()
    simulation_time = 0
    cell_num = 1

    while simulation_time <= t:
        # Calculate reaction rates based on fitness and current population
        a_0 = fitness_list * copies_num_list
        a_sum = a_0.sum()

        # Calculate time step until next division event
        tau = -(1 / a_sum) * np.log(rng.random())

        if tau > t:
            break
        else:
            # Choose a cell type to divide based on weighted probabilities
            reaction_index = rng.choice(ec_copies_list, p=a_0 / a_sum)

            if reaction_index == 0:
                # Division of ecDNA-negative cell (no segregation)
                copies_num_list[0] += 1
            else:
                # Division of ecDNA-positive cell with random segregation
                daughter_1 = int(rng.binomial(reaction_index * 2, p_s))
                daughter_2 = reaction_index * 2 - daughter_1

                # Update ecDNA copy number distribution
                copies_num_list[reaction_index] -= 1
                copies_num_list[daughter_1] += 1
                copies_num_list[daughter_2] += 1

            cell_num += 1
            simulation_time += tau

    return cell_num, simulation_time, copies_num_list






def treatment_model_a(copies_num_list, s_max, k_e, k_m, t, toxicity_i):
    """
    Treatment model A: Assumes drug cytotoxicity is independent of ecDNA copy number.

    Cells grow or die at rates determined by fitness and a constant toxicity rate.
    This model reflects a non-specific treatment that kills all cells equally, regardless of ecDNA load.

    Parameters:
        copies_num_list (array-like): initial distribution of ecDNA copy numbers
        s_max (float): maximum growth advantage
        k_e (int)
        k_m (int)
        t (float): total simulation time
        toxicity_i (float): treatment-induced death rate (uniform)

    Returns:
        cell_num (int): total remaining cell count
        simulation_time (float): total time elapsed
        copies_num_list_ (array-like): final distribution of ecDNA copy numbers
    """
    # Initialization
    ec_copies_list = inx_ec_copies
    fitness_list = sigmoid_fitness_function(
        ec_copies_list, s_max, k_e, k_m)

    copies_num_list_ = np.copy(copies_num_list)

    # Main simulation
    rng = np.random.default_rng()
    simulation_time = 0  # initial time
    cell_num = np.sum(copies_num_list_)  # initial cell number

    if cell_num == 0:
        print('No cell!')
        return (cell_num,
                simulation_time, copies_num_list_)
    else:

        while simulation_time <= t:
            # Calculate the reaction rate
            a_0 = fitness_list * copies_num_list_
            a_1 = toxicity_i * copies_num_list_
            a_sum = a_0.sum() + a_1.sum()

            # Calculate the reaction time with a random number
            tau = -(1/a_sum) * np.log(rng.random())

            # Generate another random number to determine which reaction happens and update the state
            ec_copies_list_bi = np.arange(0, 10002, 1)
            reaction_index_ = rng.choice(
                ec_copies_list_bi, p=np.concatenate((a_0, a_1))/a_sum)

            if reaction_index_ <= 5000:
                reaction_index_1 = 0
                reaction_index_2 = reaction_index_

            else:
                reaction_index_1 = 1
                reaction_index_2 = reaction_index_-5001

            # Growth
            if reaction_index_1 == 0:

                if reaction_index_2 == 0:
                    # ecDNA- cell division
                    copies_num_list_[0] += 1
                else:
                    # ecDNA+ cell division
                    # ecDNA random segregation
                    dauther_1 = int(rng.binomial(reaction_index_2*2, 0.5))
                    dauther_2 = int(reaction_index_2*2 - dauther_1)

                    # Update the list of ecDNA+ cells
                    copies_num_list_[reaction_index_2] -= 1
                    copies_num_list_[dauther_1] += 1
                    copies_num_list_[dauther_2] += 1

                cell_num += 1

            # Treatment death
            elif reaction_index_1 == 1:
                copies_num_list_[reaction_index_2] -= 1
                cell_num -= 1

            simulation_time += tau

            if cell_num == 0:
                # print("All cells are dead.")
                break

        return (cell_num,
                simulation_time, copies_num_list_)


def treatment_model_b(copies_num_list, s_max, k_e, k_m, t, toxicity_i, k_lethal, p):
    """
    Treatment model B: Assumes high ecDNA burden sensitizes cells to treatment due to replication stress or instability.

    In addition to uniform cytotoxicity, cells with ecDNA copies exceeding a probabilistic lethal threshold may spontaneously die during attempted division.

    Parameters:
        copies_num_list (array-like): initial distribution of ecDNA copy numbers
        s_max (float): maximum growth advantage
        k_e (int): 
        k_m (int): 
        t (float): total simulation time
        toxicity_i (float): treatment-induced death rate (uniform)
        k_lethal (int): lethal threshold for ecDNA burden
        p (float): probability of each ecDNA copy being 'hit' by stress

    Returns:
        cell_num (int): total remaining cell count
        simulation_time (float): total time elapsed
        copies_num_list_ (array-like): final distribution of ecDNA copy numbers
    """
    # Initialization
    ec_copies_list = inx_ec_copies
    fitness_list = sigmoid_fitness_function(
        ec_copies_list, s_max, k_e, k_m)

    copies_num_list_ = np.copy(copies_num_list)

    # Main simulation
    rng = np.random.default_rng()
    simulation_time = 0  # initial time
    cell_num = np.sum(copies_num_list_)  # initial cell number

    if cell_num == 0:
        print('No cell!')
        return (cell_num,
                simulation_time, copies_num_list_)
    else:

        while simulation_time <= t:
            # Calculate the reaction rate
            a_0 = fitness_list * copies_num_list_
            a_1 = toxicity_i * copies_num_list_
            a_sum = a_0.sum() + a_1.sum()

            # Calculate the reaction time with a random number
            tau = -(1/a_sum) * np.log(rng.random())

            # Generate another random number to determine which reaction happens and update the state
            ec_copies_list_bi = np.arange(0, 10002, 1)
            reaction_index_ = rng.choice(
                ec_copies_list_bi, p=np.concatenate((a_0, a_1))/a_sum)

            if reaction_index_ <= 5000:
                reaction_index_1 = 0
                reaction_index_2 = reaction_index_

            else:
                reaction_index_1 = 1
                reaction_index_2 = reaction_index_-5001

            # Growth
            if reaction_index_1 == 0:
                k_hit = rng.binomial(reaction_index_2, p, 1)
                if k_hit < k_lethal:
                    # cell replication
                    if reaction_index_2 == 0:
                        # ecDNA- cell division
                        copies_num_list_[0] += 1
                    else:
                        # ecDNA+ cell division
                        # ecDNA random segregation
                        dauther_1 = int(rng.binomial(reaction_index_2*2, 0.5))
                        dauther_2 = int(reaction_index_2*2 - dauther_1)

                        # Update the list of ecDNA+ cells
                        copies_num_list_[reaction_index_2] -= 1
                        copies_num_list_[dauther_1] += 1
                        copies_num_list_[dauther_2] += 1

                    cell_num += 1

                elif k_hit >= k_lethal:
                    # cell death
                    copies_num_list_[reaction_index_2] -= 1
                    cell_num -= 1

            # Treatment death
            elif reaction_index_1 == 1:
                copies_num_list_[reaction_index_2] -= 1
                cell_num -= 1

            simulation_time += tau

            if cell_num == 0:
                # print("All cells are dead.")
                break

        return (cell_num,
                simulation_time, copies_num_list_)


