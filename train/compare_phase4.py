"""Phase 4 comparison: CNN vs MLP"""
import json
import os

models = ['phase4_mlp', 'phase4_resnet_s', 'phase4_resnet_m']
archs = ['MLP (257K)', 'ResNet-S (330K)', 'ResNet-M (1.8M)']

print("=" * 75)
print("  Phase 4: CNN vs MLP Comparison (teacher800 + D4 augment, 80 epochs)")
print("=" * 75)
print(f"{'Model':<20} {'ValTop1':<9} {'ValTop3':<9} {'ValTop5':<9} {'ValMAE':<9} {'TrTop1':<9} {'Time':<8}")
print("-" * 75)

for name, arch in zip(models, archs):
    path = os.path.join("..", "data", "models", f"{name}_report.json")
    r = json.load(open(path))
    s = r['summary']
    val_recs = [rec for rec in r['metrics_history'] if 'val_policy_acc_top1' in rec]
    last = val_recs[-1]
    time_s = s['total_time_s']
    print(f"{arch:<20} {s['best_val_policy_acc_top1']:<9.4f} {s['best_val_policy_acc_top3']:<9.4f} "
          f"{s['best_val_policy_acc_top5']:<9.4f} {last['val_value_mae']:<9.4f} "
          f"{last['train_policy_acc_top1']:<9.4f} {time_s:.0f}s")

print("-" * 75)
print()
print("Best ValTop1 epochs:")
for name, arch in zip(models, archs):
    path = os.path.join("..", "data", "models", f"{name}_report.json")
    r = json.load(open(path))
    s = r['summary']
    print(f"  {arch}: epoch {s['best_val_policy_acc_top1_epoch']}")
