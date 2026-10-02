"""Reproduce the public curve figure from the frozen numeric summary; no model inference."""
import argparse
import json
import os
import tempfile
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('summary',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    data=json.loads(args.summary.read_text(encoding='utf-8-sig'))
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'research3_plot_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.0),sharey=True)
    colors={'reference':'#2166ac','coverage':'#d95f02'}
    for arm in ('reference','coverage'):
        points=sorted(data['fits'][arm]['checkpoints'],key=lambda p:p['update'])
        assert [p['update'] for p in points]==[0,7,14,28,56]
        x=[p['update'] for p in points]
        label='Original replay' if arm=='reference' else 'State-complete replay'
        axes[0].plot(x,[p['old_outcomes'] for p in points],'-o',color=colors[arm],label=label)
        axes[1].plot(x,[p['new_continuation'] for p in points],'-o',color=colors[arm],label=label+' / extension')
        axes[1].plot(x,[p['new_workflow'] for p in points],'--s',color=colors[arm],label=label+' / full workflow')
    axes[0].set_title('Original completed-work continuation')
    axes[1].set_title('Pending work in the two newer settings')
    axes[0].set_ylabel('Successful cases out of four')
    for ax in axes:
        ax.set_xlabel('Optimizer updates')
        ax.set_xticks([0,7,14,28,56])
        ax.set_yticks(range(5))
        ax.set_ylim(-0.12,4.2)
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(loc='lower left',fontsize=9,frameon=False)
    axes[1].legend(loc='lower right',fontsize=8,frameon=False)
    fig.suptitle('One seed; fixed known development panels; unchanged update dose',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ('png','pdf'):
        fig.savefig(args.output/('RETENTION_TRANSFER.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
    print('Created retention/transfer PNG and PDF from observed counts.')


if __name__=='__main__':
    main()
