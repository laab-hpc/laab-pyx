import os
import sys
import statistics
from jinja2 import Template
from pathlib import Path
import json
import pickle
import argparse
from .laab_results import LAABResults

def format_floats_recursive(data: dict, precision: int = 2) -> dict:
    """
    Recursively traverses a dictionary and formats all float values
    to a specified number of decimal places.
    """
    for key, value in data.items():
        if isinstance(value, float):
            # Format the float to the given precision
            data[key] = float(f"{value:.{precision}f}")
        elif isinstance(value, dict):
            # If the value is a dict, recurse into it
            format_floats_recursive(value, precision)
    return data

def format_cutoff_results_md(cutoff_results):
    for exp, tests in cutoff_results.items():
        for test, result in tests.items():
            if result == True:
                cutoff_results[exp][test] = ":white_check_mark:"
            else:
                cutoff_results[exp][test] = ":x:"
    return cutoff_results


def dump_results_pickle(laab_results, src_config_file, exp_config, pickle_path, cutoff=0.05):
    
    if not os.path.exists(src_config_file):
        raise FileNotFoundError(f"Config file {src_config_file} not found")
    config = json.load(open(src_config_file, "r"))

    if set(laab_results.data.keys()) != set(config.keys()):
        missing = set(config.keys()) - set(laab_results.data.keys())
        print("Missing experiments in data file:", missing)
        raise ValueError("Experiments in data file and config file do not match")
    
    min_exec_times = laab_results.get_min_test_times()
    laab_results.compute_loss()
    losses = laab_results.loss
    mean_loss = laab_results.mean_loss
    ret = laab_results.apply_cutoff(cutoff=cutoff)
    cutoff_results = ret.results
    score = ret.score
    
    prec=3
    data = {
        "eb_name": laab_results.eb_version,
        "system": laab_results.system,
        "cpu_model": laab_results.cpu_model,
        "losses": format_floats_recursive(losses,prec),
        "slow_down": format_floats_recursive(laab_results.slow_down,prec),
        "mean_loss": f"{mean_loss:.{prec}f}",
        "cutoff_results": format_cutoff_results_md(cutoff_results),
        "score": score,
        "num_tests": len(losses),
        "times": format_floats_recursive(min_exec_times,prec),
        "cutoff": f"{cutoff:.2f}",
        "config": config,
        "exp_config": exp_config
    }
    
    #dump to pickle
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Results dumped to {pickle_path}")
    
def main():
    parser = argparse.ArgumentParser(description="Prepare a markdown report from a pickle file")
    parser.add_argument("log", type=str, help="Path to log file")
    parser.add_argument("--outfile", type=str, required=True, help="Path to the output markdown file")
    parser.add_argument("--name", required=True, help="Name of the experiment")
    parser.add_argument("--system", required=True, help="System name to use for the experiment")
    parser.add_argument("--lib", required=True, help="Library name to use for the experiment")
    parser.add_argument("--version", required=True, help="Library version to use for the experiment")
    parser.add_argument("--toolchain", required=True, help="Toolchain name to use for the experiment")
    parser.add_argument("--config_path", required=True, help="Path to the config file")
    parser.add_argument("--prob_size", type=str, required=True, help="Problem size for the experiment")
    parser.add_argument("--reps", type=int, default=10, help="Number of repetitions for the experiment")
    parser.add_argument("--omp_num_threads", type=int, default=1, help="Number of OpenMP threads for the experiment")
    
    args = parser.parse_args()
    
    log_file = Path(args.log)
    if not log_file.exists():
        raise FileNotFoundError(f"Log file {log_file} not found")
    laab_results = LAABResults(log_file, f"{args.lib}/{args.version}-{args.toolchain}", args.system)
    
    exp_config = {
        "laab_n":args.prob_size,
        "laab_rep":args.reps,
        "name": args.name,
        "omp_num_threads": args.omp_num_threads
    }

    pickle_path = Path(args.outfile).with_suffix(".pkl")
    dump_results_pickle(laab_results, args.config_path, exp_config, pickle_path, cutoff=0.05)

    
    
