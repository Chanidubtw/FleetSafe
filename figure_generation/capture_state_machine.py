import matplotlib

try:
    import tkinter  # noqa: F401
    matplotlib.use('TkAgg')
    _HAS_GUI = True
except ImportError:
    matplotlib.use('Agg')
    _HAS_GUI = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#0f1117')
ax.set_facecolor('#0f1117')
ax.set_xlim(0, 12); ax.set_ylim(0, 7)
ax.axis('off')

fig.suptitle('FleetSafe — Violation Detection State Machine',
             color='white', fontsize=13, fontweight='bold')

# ── State boxes ───────────────────────────────────────────────────────────────
def state_box(ax, cx, cy, label, sublabel, fc, ec, r=0.5):
    circle = plt.Circle((cx, cy), r, facecolor=fc, edgecolor=ec,
                         linewidth=2.5, zorder=4)
    ax.add_patch(circle)
    ax.text(cx, cy + 0.12, label, ha='center', va='center',
            fontsize=11, fontweight='bold', color='white', zorder=5)
    ax.text(cx, cy - 0.22, sublabel, ha='center', va='center',
            fontsize=7.5, color='#B0BEC5', zorder=5, style='italic')

# Start marker
ax.add_patch(plt.Circle((2.0, 5.5), 0.15, facecolor='white',
                          edgecolor='white', zorder=4))
ax.annotate('', xy=(2.0, 4.55), xytext=(2.0, 5.35),
            arrowprops=dict(arrowstyle='->', color='white',
                            lw=2, mutation_scale=14))

# States
state_box(ax, 2.0, 4.0,  'IDLE',          'Default monitoring',    '#1B5E20', '#4CAF50')
state_box(ax, 6.0, 4.0,  'CONFIRMING',    '0.5 s timer running',   '#E65100', '#FF9800')
state_box(ax, 10.0, 4.0, 'COOLING_DOWN',  '2 s cooldown active',   '#B71C1C', '#F44336')

# ── Arrows ────────────────────────────────────────────────────────────────────
arrowprops_fwd = dict(arrowstyle='->', color='#90CAF9',
                      lw=2, mutation_scale=16)
arrowprops_back = dict(arrowstyle='->', color='#A5D6A7',
                       lw=2, mutation_scale=16)
arrowprops_loop = dict(arrowstyle='->', color='#EF9A9A',
                       lw=2, mutation_scale=16,
                       connectionstyle='arc3,rad=-0.45')

# IDLE → CONFIRMING
ax.annotate('', xy=(5.5, 4.15), xytext=(2.5, 4.15),
            arrowprops=arrowprops_fwd)
ax.text(4.0, 4.55,
        'Vehicle crosses double-solid boundary\nby > 25 px  AND  confidence ≥ 75%',
        ha='center', fontsize=8, color='#90CAF9', fontweight='bold')

# CONFIRMING → IDLE (back, curved below)
ax.annotate('', xy=(2.5, 3.85), xytext=(5.5, 3.85),
            arrowprops=dict(arrowstyle='->', color='#A5D6A7',
                            lw=2, mutation_scale=16,
                            connectionstyle='arc3,rad=0.35'))
ax.text(4.0, 3.05,
        'Condition resolves\nbefore 0.5 s',
        ha='center', fontsize=8, color='#A5D6A7')

# CONFIRMING → COOLING_DOWN
ax.annotate('', xy=(9.5, 4.15), xytext=(6.5, 4.15),
            arrowprops=arrowprops_fwd)
ax.text(8.0, 4.55,
        'Condition persists ≥ 0.5 s\n→  violation event emitted',
        ha='center', fontsize=8, color='#90CAF9', fontweight='bold')

# COOLING_DOWN → IDLE (loop back along top)
ax.annotate('', xy=(2.2, 4.5), xytext=(9.8, 4.5),
            arrowprops=dict(arrowstyle='->', color='#EF9A9A',
                            lw=2, mutation_scale=16,
                            connectionstyle='arc3,rad=-0.4'))
ax.text(6.0, 5.9, '2-second cooldown expires',
        ha='center', fontsize=8.5, color='#EF9A9A', fontweight='bold')

# ── Legend ────────────────────────────────────────────────────────────────────
for x, col, lbl in [(0.3, '#90CAF9', 'Forward transition'),
                     (0.3, '#A5D6A7', 'Abort / reset transition'),
                     (0.3, '#EF9A9A', 'Cooldown reset')]:
    pass

legend_items = [
    mpatches.Patch(facecolor='#1B5E20', edgecolor='#4CAF50', label='IDLE — no active event'),
    mpatches.Patch(facecolor='#E65100', edgecolor='#FF9800', label='CONFIRMING — 0.5 s confirmation window'),
    mpatches.Patch(facecolor='#B71C1C', edgecolor='#F44336', label='COOLING_DOWN — 2 s duplicate prevention'),
]
ax.legend(handles=legend_items, loc='lower center',
          ncol=3, fontsize=8, framealpha=0.2,
          labelcolor='white', facecolor='#1a1a2e',
          edgecolor='#455A64')

# ── Source note ───────────────────────────────────────────────────────────────
ax.text(6.0, 0.3,
        'Source: LaneLines.py — violation_active, violation_start_time, '
        'violation_last_emit_time variables (FleetSafe backend)',
        ha='center', fontsize=7, color='#546E7A', style='italic')

plt.tight_layout()
plt.savefig('fig3_state_machine.png', dpi=200,
            bbox_inches='tight', facecolor='#0f1117')
print('Saved: fig3_state_machine.png')

if _HAS_GUI:
    plt.show()
else:
    print('GUI backend unavailable; figure was saved without opening a window.')
