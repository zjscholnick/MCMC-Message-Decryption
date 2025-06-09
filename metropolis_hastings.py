# File: metropolis_hastings.py

import numpy as np
import time
import random
import psutil
import shutil

def metropolis_hastings(initial_state,
                        proposal_function,
                        log_density,
                        iters=1000,
                        print_every=10,
                        tolerance=0.02,
                        error_function=None,
                        pretty_state=None):
    """
    Runs a Metropolis–Hastings sampler, while tracking detailed performance stats.
    Returns: (states, log_probs, errors, performance)
    """

    # Initialize performance bookkeeping
    proc = psutil.Process()
    memory_start = proc.memory_info().rss / (1024**2)
    time_start = time.time()
    performance = {
        'time': {
            'start': time_start,
            'end': None,
            'total': None,
            'proposal_fn': 0.0,
            'prob_fn': 0.0,
            'per_iteration': []
        },
        'memory': {
            'start': memory_start,
            'peak': memory_start,
            'end': None,
            'sampled_memory': []
        },
        'algorithm': {
            'num_iterations': 0,
            'num_proposals': 0,
            'num_accepted': 0,
            'accept_rate': 0.0,
            'best_entropy': float('inf')
        },
        'cpu': {
            'samples': []
        }
    }

    # initial log‐density
    p1 = log_density(initial_state)
    # initial entropy = – log_prob
    performance['algorithm']['best_entropy'] = min(performance['algorithm']['best_entropy'],
                                                   -p1)

    state = initial_state
    cnt = 0
    accept_cnt = 0
    it = 0

    states = [initial_state]
    cross_entropies = []
    errors = [] if error_function else None

    entropy_print = 100000

    while it < iters:
        # PROPOSAL
        t0 = time.time()
        new_state = proposal_function(state)
        dt = time.time() - t0
        performance['time']['proposal_fn'] += dt

        performance['algorithm']['num_proposals'] += 1
        cnt += 1

        # LOG‐DENSITY
        t1 = time.time()
        p2 = log_density(new_state)
        dp = time.time() - t1
        performance['time']['prob_fn'] += dp

        # MH acceptance
        u = random.random()
        if (p2 - p1) > np.log(u):
            # accept
            state = new_state
            p1 = p2
            accept_cnt += 1
            it += 1

            # record state
            states.append(state)
            cross_entropies.append(p1)

            # error if requested
            if error_function:
                err = error_function(state)
                errors.append(err)

            # update algorithm stats
            performance['algorithm']['num_iterations'] = it
            performance['algorithm']['num_accepted'] = accept_cnt
            performance['algorithm']['best_entropy'] = min(
                performance['algorithm']['best_entropy'],
                -p1
            )

            # sample memory & CPU
            mem = proc.memory_info().rss / (1024**2)
            performance['memory']['sampled_memory'].append(mem)
            performance['memory']['peak'] = max(performance['memory']['peak'], mem)
            performance['cpu']['samples'].append(proc.cpu_percent(interval=None))

            # record per-iteration timings
            performance['time']['per_iteration'].append({
                'proposal': dt,
                'prob': dp
            })

            # diagnostic printing
            if -p1 < 0.995 * entropy_print:
                entropy_print = -p1
                acc_rate = accept_cnt / float(cnt)
                print(shutil.get_terminal_size().columns * '-')
                print(f"\n Entropy : {round(p1,4)}"
                      f", Iteration : {it}"
                      f", Acceptance : {round(acc_rate,4)}")
                if pretty_state:
                    print(pretty_state(state))
                print(shutil.get_terminal_size().columns * '-')

                if acc_rate < tolerance:
                    break

                # reset counters
                cnt = 0
                accept_cnt = 0
                #time.sleep(.1)

    # finalize performance
    performance['time']['end'] = time.time()
    performance['time']['total'] = performance['time']['end'] - performance['time']['start']
    performance['memory']['end'] = proc.memory_info().rss / (1024**2)
    performance['algorithm']['accept_rate'] = (
        performance['algorithm']['num_accepted'] /
        max(1, performance['algorithm']['num_proposals'])
    )

    return states, cross_entropies, errors, performance

