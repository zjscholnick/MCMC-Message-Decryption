import sys,shutil
from optparse import OptionParser
from concurrent.futures import ProcessPoolExecutor,as_completed
from metropolis_hastings import metropolis_hastings
from deciphering_utils import (
    compute_statistics, get_state,
    propose_a_move, compute_probability_of_state,
    pretty_state
)

def run_chain(init,its,tol,pe):
    return metropolis_hastings(init,
        proposal_function=propose_a_move,
        log_density=compute_probability_of_state,
        iters=its,tolerance=tol,print_every=pe,
        pretty_state=pretty_state)

def main(argv):
    p=OptionParser()
    p.add_option("-i","--input",dest="inputfile")
    p.add_option("-d","--decode",dest="decode")
    p.add_option("-e","--iters",dest="iterations",default=5000)
    p.add_option("-t","--tolerance",dest="tolerance",default=0.02)
    p.add_option("-p","--print_every",dest="print_every",default=10000)
    opts,_=p.parse_args(argv)
    if not opts.inputfile or not opts.decode:
        print("Usage: decode.py -i <input> -d <decode>"); sys.exit(1)

    c2i, i2c, tr, fr = compute_statistics(opts.inputfile)
    sc = list(open(opts.decode).read())
    init = get_state(sc,tr,fr,c2i)

    metrics=[];all_s=[];all_e=[]
    with ProcessPoolExecutor() as exec:
        futs = [exec.submit(run_chain,init,
                    int(opts.iterations),float(opts.tolerance),
                    int(opts.print_every)) for _ in range(3)]
        for f in as_completed(futs):
            s,lps,_,perf = f.result()
            all_s.extend(s); all_e.extend(lps); metrics.append(perf)

    paired = sorted(zip(all_s,all_e), key=lambda x:x[1])
    print("\nBest Guesses:"+"*"*shutil.get_terminal_size().columns)
    for j in (1,2,3):
        st,ent=paired[-j]
        print(f"Guess {j} (ent={-ent:.4f}):"); print(pretty_state(st,full=True))
        print("*"*shutil.get_terminal_size().columns)

    print("\n Best Guesses :\n" + ("*" * shutil.get_terminal_size().columns))
    for j in range(1, 4):
        st, ent = paired[-j]
        print(f"\nGuess {j} (entropy={round(-ent,4)}):\n")
        print(pretty_state(st, full=True))
        print("*" * shutil.get_terminal_size().columns)

    # aggregate and print overall metrics
    total_runtime    = sum(m['time']['total'] for m in metrics)
    total_proposals  = sum(m['algorithm']['num_proposals'] for m in metrics)
    total_accepted   = sum(m['algorithm']['num_accepted'] for m in metrics)
    total_iterations = sum(m['algorithm']['num_iterations'] for m in metrics)
    overall_accept   = (total_accepted / max(1, total_proposals)) * 100

    print("\nOVERALL STATISTICS:")
    print(f"  Total runtime: {total_runtime:.2f} s")
    print(f"  Total iterations: {total_iterations}")
    print(f"  Total proposals : {total_proposals}")
    print(f"  Overall accept rate: {overall_accept:.2f}%")

    # time breakdown
    prop_time = sum(m['time']['proposal_fn'] for m in metrics)
    dens_time = sum(m['time']['prob_fn']     for m in metrics)
    print("\nTIME BREAKDOWN:")
    print(f"  Proposal fn: {prop_time:.2f}s ({100*prop_time/total_runtime:.1f}%)")
    print(f"  Density fn : {dens_time:.2f}s ({100*dens_time/total_runtime:.1f}%)")
    print(f"  Overhead   : {total_runtime-prop_time-dens_time:.2f}s")

    # memory
    peak_mem = max(m['memory']['peak'] for m in metrics)
    print("\nMEMORY USAGE:")
    print(f"  Peak memory: {peak_mem:.2f} MB")

    # per-run details
    for idx, m in enumerate(metrics, 1):
        print(f"\nRUN {idx}:")
        print(f"  Runtime      : {m['time']['total']:.2f} s")
        print(f"  Iterations   : {m['algorithm']['num_iterations']}")
        print(f"  Proposals    : {m['algorithm']['num_proposals']}")
        print(f"  Accept rate  : {m['algorithm']['accept_rate']*100:.2f}%")
        print(f"  Best entropy : {m['algorithm']['best_entropy']:.4f}")

if __name__ == "__main__":
    main(sys.argv)

