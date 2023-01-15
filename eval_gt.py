from evaluation.metrics import compute_metrics

#print("Compute metrics ... ")
metrics = compute_metrics(f"/cluster/scratch/tshpakov/results/diffwave_samples/audio", gt_metrics=True)

# Log metrics
with open(f'/cluster/scratch/tshpakov/results/diffwave_samples/metrics.txt', 'w+') as metric_file:
     metric_file.write(json.dumps(metrics))
