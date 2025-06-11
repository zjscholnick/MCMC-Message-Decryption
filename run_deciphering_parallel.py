import sys
import shutil
import os
from optparse import OptionParser
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from metropolis_hastings import metropolis_hastings
from deciphering_utils import (
    compute_statistics, get_state,
    propose_a_move, compute_probability_of_state,
    pretty_state
)

def run_chain(scrambled_text, tr, fr, char_to_ix, iters, tol, print_every):
    print(f"Running chain in PID {os.getpid()}")
    init_state = get_state(scrambled_text, tr, fr, char_to_ix)
    return metropolis_hastings(
        init_state,
        proposal_function=propose_a_move,
        log_density=compute_probability_of_state,
        iters=iters,
        tolerance=tol,
        print_every=print_every,
        pretty_state=pretty_state
    )

def main(argv):
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="inputfile")
    parser.add_option("-d", "--decode", dest="decode")
    parser.add_option("-e", "--iters", dest="iterations", default=5000)
    parser.add_option("-t", "--tolerance", dest="tolerance", default=0.02)
    parser.add_option("-p", "--print_every", dest="print_every", default=10000)
    opts, _ = parser.parse_args(argv)

    if not opts.inputfile or not opts.decode:
        print("Usage: decode.py -i <input> -d <decode>")
        sys.exit(1)

    c2i, i2c, tr, fr = compute_statistics(opts.inputfile)
    sc = list(open(opts.decode).read())

    metrics = []
    all_states = []
    all_entropies = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_chain, sc, tr, fr, c2i,
                int(opts.iterations), float(opts.tolerance), int(opts.print_every)
            ) for _ in range(3)
        ]

        for future in as_completed(futures):
            states, lps, _, perf = future.result()
            all_states.extend(states)
            all_entropies.extend(lps)
            metrics.append(perf)

    paired = sorted(zip(all_states, all_entropies), key=lambda x: x[1])
    print("\nBest Guesses:" + "*" * shutil.get_terminal_size().columns)
    for j in range(1, 4):
        st, ent = paired[-j]
        print(f"\nGuess {j} (entropy={round(-ent, 4)}):\n")
        print(pretty_state(st, full=True))
        print("*" * shutil.get_terminal_size().columns)
    total_runtime = max(m['time']['total'] for m in metrics)
    total_proposals = sum(m['algorithm']['num_proposals'] for m in metrics)
    total_accepted = sum(m['algorithm']['num_accepted'] for m in metrics)
    total_iterations = sum(m['algorithm']['num_iterations'] for m in metrics)
    overall_accept = (total_accepted / max(1, total_proposals)) * 100

    print("\nOVERALL STATISTICS:")
    print(f"  Total runtime: {total_runtime:.2f} s")
    print(f"  Total iterations: {total_iterations}")
    print(f"  Total proposals : {total_proposals}")
    print(f"  Overall accept rate: {overall_accept:.2f}%")

    prop_time = sum(m['time']['proposal_fn'] for m in metrics)
    dens_time = sum(m['time']['prob_fn'] for m in metrics)
    print("\nTIME BREAKDOWN:")
    print(f"  Proposal fn: {prop_time:.2f}s ({100 * prop_time / total_runtime:.1f}%)")
    print(f"  Density fn : {dens_time:.2f}s ({100 * dens_time / total_runtime:.1f}%)")
    print(f"  Overhead   : {total_runtime - prop_time - dens_time:.2f}s")

    peak_mem = max(m['memory']['peak'] for m in metrics)
    print("\nMEMORY USAGE:")
    print(f"  Peak memory: {peak_mem:.2f} MB")

    #CPU usage
    total_cpu_time = sum(
        (m['cpu']['avg'] / 100.0) * m['time']['total']
        for m in metrics
    )
    avg_cpu = sum(m['cpu']['avg'] for m in metrics) / len(metrics)
    cpu_core_utilization = total_cpu_time / total_runtime

    print("\nCPU USAGE:")
    print(f"  Total CPU time used: {total_cpu_time:.3f} core-seconds")
    print(f"  Avg CPU usage per chain: {avg_cpu:.3f}%")
    print(f"  Avg core utilization across all chains: {cpu_core_utilization:.3f} cores")

    for idx, m in enumerate(metrics, 1):
        print(f"\nRUN {idx}:")
        print(f"  Runtime      : {m['time']['total']:.2f} s")
        print(f"  Iterations   : {m['algorithm']['num_iterations']}")
        print(f"  Proposals    : {m['algorithm']['num_proposals']}")
        print(f"  Accept rate  : {m['algorithm']['accept_rate'] * 100:.2f}%")
        print(f"  Best entropy : {m['algorithm']['best_entropy']:.4f}")

if __name__ == "__main__":
    main(sys.argv)
