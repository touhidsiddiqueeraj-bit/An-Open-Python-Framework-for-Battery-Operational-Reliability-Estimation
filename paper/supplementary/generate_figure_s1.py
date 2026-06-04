import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('../../code/results/supplementary_analysis.json') as f:
    d = json.load(f)

kl_data = d['per_feature_kl']
features = list(kl_data.keys())
values = [kl_data[f]['kl_divergence_nats'] for f in features]

fig, ax = plt.subplots(figsize=(6, 3.5))
colors = ['#1b3a5c', '#4a7a9c', '#7a9aae', '#a0b8c4', '#c4d4dc', '#d8e2e8', '#e8eef2']
bars = ax.barh(range(len(features)), values, color=colors[:len(features)], edgecolor='white', height=0.6)
ax.set_yticks(range(len(features)))
ax.set_yticklabels(features)
ax.set_xlabel('KL Divergence (nats)')
ax.set_title('Per-Feature KL Divergence: Synthetic vs NASA')
ax.set_xlim(0, max(values) * 1.15)
for bar, v in zip(bars, values):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{v:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('figure_s1.png', dpi=300, bbox_inches='tight')
plt.close()
print('figure_s1.png created')
