from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
D=json.loads((P/'plot_values.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(7.5,3.55),layout='constrained',sharex=True,sharey=True)
fig.suptitle('MetaMatch-TTA vs. MetaMatch',fontweight='bold',fontsize=11)
for ax,pr,title,color in zip(axs,['pix1','perturb'],['(a) Single-pixel reference P1','(b) Perturbed reference P4'],['#0072B2','#A65300']):
    delta=D['paired_deltas'][pr];best=max(round(v,3) for v in delta)
    ax.axvline(0,color='#888888',lw=.8)
    for i,v in enumerate(delta):
        ax.plot([0,v],[i,i],color=color,lw=1.4)
        ax.annotate(f'{v:+.3f}',(v,i),xytext=(5 if v>=0 else -5,0),textcoords='offset points',ha='left' if v>=0 else 'right',va='center',fontsize=8.5,fontweight='bold' if round(v,3)==best else 'normal')
    ax.scatter(delta,range(6),color=color,marker='o',s=28,zorder=3)
    ax.set_yticks(range(6),D['scenes']);ax.set_title(title,fontsize=10)
    ax.set_xlabel('ROC-AUC change (percentage points)',fontsize=9)
    ax.set_xlim(-.95,.98);ax.grid(axis='x',alpha=.23);ax.set_ylim(-.6,5.6)
axs[0].invert_yaxis()
fig.savefig(P/'refinement_effect.pdf',facecolor='white');fig.savefig(P/'refinement_effect.png',dpi=180,facecolor='white');plt.close(fig)

plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7})
prots=['mean','pix1','pixk3','pixk5','perturb','cross']
labels=['P0\nFull mean','P1\nSingle pixel','P2\nThree pixels','P2\nFive pixels','P4\nPerturbed','P3\nCross-image']
colors={'zero':'#0072B2','refined':'#8D3989','FAME':'#4A6A55'}
fig,ax=plt.subplots(figsize=(7.5,3.8),layout='constrained');macro=D['macro']
for method,off,marker,label in [('FAME',-.14,'s','FAME-Net (2026, published)'),('zero',0,'^','MetaMatch'),('refined',.14,'D','MetaMatch-TTA')]:
    xx=[i+off for i,pr in enumerate(prots) if method in macro[pr]];yy=[macro[pr][method] for pr in prots if method in macro[pr]]
    ax.scatter(xx,yy,label=label,color=colors[method],marker=marker,s=36,zorder=3)
for i,pr in enumerate(prots):
    best=max(macro[pr].values());ax.annotate(f'{best:.4f}',(i,best),xytext=(0,9),textcoords='offset points',ha='center',fontweight='bold',fontsize=9)
ax.set_xticks(range(6),labels);ax.set_ylabel('Scene-macro-average ROC-AUC');ax.set(ylim=(.98,1.004),xlim=(-.55,5.55));ax.grid(axis='y',alpha=.23)
legend=ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.18),frameon=False,fontsize=8)
for t in legend.get_texts():
    if t.get_text().startswith('MetaMatch'):t.set_fontweight('bold')
fig.savefig(P/'protocol_comparison.pdf',bbox_inches='tight');fig.savefig(P/'protocol_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
print('Two English figures generated with original values and axis ranges.')
