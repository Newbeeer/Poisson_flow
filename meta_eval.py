import os
import json
import numpy as np
import pprint

root = os.getcwd()
results_dir = "results"

active_dir = os.path.join(root, results_dir)
nets = os.listdir(active_dir)

results_dict = {}

for net in nets:
    results_dict[net]={}
    active_dir = os.path.join(active_dir, net)
    checkpoints = [dr for dr in os.listdir(active_dir) if os.path.isdir(os.path.join(active_dir, dr))]
    for checkpoint in checkpoints:
        results_dict[net][checkpoint]={}
        active_dir = os.path.join(active_dir, checkpoint)
        experiments = [dr for dr in os.listdir(active_dir) if os.path.isdir(os.path.join(active_dir, dr))]
        runs = set(["_".join(exp.split('_')[:-1]) for exp in experiments])
        
        for run in runs:
            results_dict[net][checkpoint][run]={}
            results_dict[net][checkpoint][run]["nfe"]={}
            results_dict[net][checkpoint][run]["benchmarks"]={}
            results_dict[net][checkpoint][run]["metrics"]={}

            metrics = {}
            metrics["fid"] = []
            metrics["is"] = []
            metrics["mis"] = []
            metrics["am"] = []
            metrics["ndb"] = []


            nfe = []
            bench = []
            seeds = set([exp.split('_')[-1] for exp in experiments if run in exp])
            for seed in seeds:
                active_dir = os.path.join(active_dir, run + '_' + seed)
                files = os.listdir(active_dir)
                for file in files:
                    if "metrics" in file:
                        with open(os.path.join(active_dir, file)) as f:
                            content = f.readline()
                            for (k,v) in json.loads(content).items():
                                metrics[k].append(v)
                    elif "nfe" in file:
                        with open(os.path.join(active_dir, file)) as f:
                            content = [float(line.rstrip()) for line in f]
                            nfe.append(content)
                    elif "benchmarking" in file:
                        with open(os.path.join(active_dir, file)) as f:
                            content = [float(line.rstrip()) for line in f]
                            bench.append(content)
                    else:
                        pass
                
                active_dir = "/".join(active_dir.split('/')[:-1])
            nfe_flist = [item for sublist in nfe for item in sublist]
            results_dict[net][checkpoint][run]["nfe"]["mean"] = np.array(nfe_flist).mean()
            results_dict[net][checkpoint][run]["nfe"]["std"] = np.array(nfe_flist).std()

            bench_flist = [item for sublist in bench for item in sublist]
            results_dict[net][checkpoint][run]["benchmarks"]["mean"] = np.array(bench_flist).mean()
            results_dict[net][checkpoint][run]["benchmarks"]["std"] = np.array(bench_flist).std()

            metrics_mean = {k: np.array(v).mean() for (k,v) in metrics.items()}
            metrics_std = {k: np.array(v).std() for (k,v) in metrics.items()}

            for k in metrics.keys():
                results_dict[net][checkpoint][run]["metrics"][k]={"mean": metrics_mean[k], "std": metrics_std[k]}
        active_dir = "/".join(active_dir.split('/')[:-1])
    active_dir = "/".join(active_dir.split('/')[:-1])

    pprint.pprint(results_dict)
