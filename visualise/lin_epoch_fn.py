import matplotlib.pyplot as plt

w, h = plt.figaspect(3/4)
fig, ax = plt.subplots(figsize=(w*0.75, h*0.75))

ax.plot([1, 20], [60, 60], label='Base Experiment')
ax.plot([1, 20], [30, 70], label='Experiment 6')
ax.set_xlim(1, 20)
ax.set_ylim(30, 70)
ax.set_xticks(range(1, 21, 2))
ax.set_xlabel('Generation')
ax.set_ylabel('Number of training epochs')
ax.legend()
plt.tight_layout()
plt.savefig('visualise/exp6func.pdf')
plt.show()
