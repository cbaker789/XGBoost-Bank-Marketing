"""Export aggregate experiment figures from reproducible result tables."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
from sklearn.calibration import calibration_curve
from research import ROOT

BLUE, GOLD = '#27659B', '#C78B26'


def build():
    root = ROOT/'results'; destination = root/'figures'; destination.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titlepad':16,'axes.edgecolor':'#B8C1CA','svg.fonttype':'none'})
    def save(fig, name, note):
        fig.text(.08,.025,note,fontsize=8,color='#637180')
        for ext in ['png','svg']:
            fig.savefig(destination/f'{name}.{ext}',dpi=170,facecolor='white',bbox_inches='tight')
        plt.close(fig)
    leaderboard = pd.read_csv(root/'leaderboard.csv').sort_values('mean_ap_ratio')
    fig, ax = plt.subplots(figsize=(10,6)); fig.subplots_adjust(left=.27,bottom=.17)
    ax.barh(leaderboard.name,leaderboard.mean_ap_ratio,color=BLUE)
    ax.axvline(1,color='#253342',ls='--',label='No-skill reference')
    ax.set(xlabel='Mean average precision / evaluation-window prevalence', title='Development benchmark: three temporal evaluation windows')
    ax.legend(frameon=False); ax.set_axisbelow(True); ax.grid(axis='x',alpha=.15)
    save(fig,'01_model_benchmark','Source: development_metrics.csv | Fixed candidate set; higher AP ratio is better. No confidence intervals asserted.')
    blocks = pd.read_csv(root/'source_order_blocks.csv')
    fig, ax = plt.subplots(figsize=(10,5.8)); fig.subplots_adjust(bottom=.2)
    ax.plot(blocks.source_order_decile,blocks.subscription_rate,'o-',color=BLUE)
    ax.set(ylim=(0,1),xticks=range(1,11),xlabel='Consecutive source-order block (10% of records each)',ylabel='Observed subscription rate',title='Response rates across the source file')
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis='y',alpha=.15)
    save(fig,'02_response_shift','Source: source_order_blocks.csv | Blocks preserve file order, not equal calendar periods; no causal mechanism inferred.')
    development = pd.read_csv(root/'development_metrics.csv')
    fig, ax = plt.subplots(figsize=(11,6)); fig.subplots_adjust(right=.73,bottom=.18)
    palette=['#27659B','#C78B26','#5F6B76','#9B4775']
    leaders = pd.read_csv(root/'leaderboard.csv').name.head(3).tolist()
    shown = list(dict.fromkeys(leaders+['prior']))
    for color,name in zip(palette,shown):
        part = development[development.name==name]
        ax.plot(part.fold,part.lift_at_10pct,marker='o',lw=1.5,color=color,label=name)
    ax.axhline(1,color='black',ls='--',lw=1)
    ax.set(xticks=[1,2,3],xlabel='Successive evaluation window',ylabel='Cumulative lift at 10% contact budget',title='Ranking stability: leading candidates and baseline')
    ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',frameon=False,fontsize=9); ax.grid(axis='y',alpha=.15)
    save(fig,'03_backtest_stability','Source: development_metrics.csv | Top three by development AP ratio + baseline. Overlapping histories; ranges are not confidence intervals.')
    pred = pd.read_csv(ROOT/'outputs/research/final_predictions.csv')
    fig, ax = plt.subplots(figsize=(8,6)); fig.subplots_adjust(bottom=.2)
    for col,color,label,marker in [('raw_probability',GOLD,'Raw model','s'),('probability',BLUE,'Selected calibration policy','o')]:
        observed,mean = calibration_curve(pred.actual,pred[col],n_bins=8,strategy='quantile')
        ax.plot(mean,observed,marker=marker,color=color,label=label)
    ax.plot([0,1],[0,1],'--',color='#253342',label='Perfect calibration')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed subscription rate',title='Calibration on the existing final window')
    ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(frameon=False)
    save(fig,'04_calibration_comparison','Source: outputs/research/final_predictions.csv | Quantile bins; descriptive rates, no uncertainty intervals. Final outcomes previously inspected.')
    budgets = pd.read_csv(root/'contact_budget_scenarios.csv')
    fig, ax = plt.subplots(figsize=(8,5.8)); fig.subplots_adjust(bottom=.2)
    ax.bar([f'{x:.0%}' for x in budgets.budget],budgets.capture,color=BLUE,width=.55)
    ax.plot(range(len(budgets)),budgets.budget,'o--',color='#253342',label='Random-selection expectation')
    ax.set(ylim=(0,1),xlabel='Contact budget',ylabel='Share of subscribers captured',title='Subscriber capture by contact budget')
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(frameon=False)
    save(fig,'05_contact_budget','Source: contact_budget_scenarios.csv | Expected random tie-breaking. Retrospective capture is not causal treatment uplift.')

if __name__ == '__main__':
    build()
