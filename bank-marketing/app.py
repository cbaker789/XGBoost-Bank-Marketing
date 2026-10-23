"""Local, read-only portfolio demonstration. Run: python -m streamlit run app.py"""
import json
import numpy as np
import pandas as pd
import streamlit as st
from research import ROOT, budget_metrics, infer
import joblib

st.set_page_config(page_title='Bank Marketing | Temporal ML Study', layout='wide')
st.title('Customer targeting under temporal distribution shift')
st.caption('Reproducible research on 41,188 banking records • UCI Bank Marketing')
st.warning('Exploratory portfolio demonstration. Evaluation outcomes were previously inspected. Results are retrospective; monetary scenarios are hypothetical.')
result_dir = ROOT/'results'
if not (result_dir/'final_evaluation.json').exists():
    st.info('Run python run_experiments.py to generate the benchmark first.')
    st.stop()
final = json.loads((result_dir/'final_evaluation.json').read_text())
model = final['selected_model']
cols = st.columns(4)
for col,label,value in zip(cols,['Selected model','ROC AUC','Lift at 10%','Evaluation customers'],[model['name'],f"{model['roc_auc']:.3f}",f"{model['lift_at_10pct']:.2f}×",f"{model['n']:,}"]):
    col.metric(label,value)
benchmark, economics, explore, methodology = st.tabs(['Model evidence','Contact budget scenario','Score sample customers','Method and limitations'])
with benchmark:
    st.subheader('Development comparison')
    st.dataframe(pd.read_csv(result_dir/'leaderboard.csv'),hide_index=True)
    st.image(str(result_dir/'figures/03_backtest_stability.png'))
    st.image(str(result_dir/'figures/04_calibration_comparison.png'))
with economics:
    budget = st.slider('Share of customers to contact',1,100,10)/100
    cost = st.number_input('Hypothetical cost per contact ($)',min_value=0.,value=2.)
    value = st.number_input('Hypothetical value per observed subscription ($)',min_value=0.,value=100.)
    pred = pd.read_csv(ROOT/'outputs/research/final_predictions.csv')
    result = budget_metrics(pred.actual,pred.probability,budget)
    c = st.columns(3)
    c[0].metric('Contacts',f"{result['contacts']:,}")
    c[1].metric('Expected subscribers selected',f"{result['expected_subscribers']:.1f}")
    c[2].metric('Subscribers captured',f"{result['capture']:.1%}")
    st.metric('Hypothetical net response value',f"${result['expected_subscribers']*value-result['contacts']*cost:,.0f}")
    st.caption('Expected selection averages over score ties. Value = observed subscribers selected × assumed value − contact cost. This is not actual profit or incremental revenue; contacting may not cause the observed subscription.')
with explore:
    st.write('Score a few original dataset records with the saved model and frozen policy. These records may have appeared in development; they are not new validation evidence.')
    row = st.number_input('Source row (zero-based)',min_value=0,max_value=41187,value=35009)
    if st.button('Score five sample customers'):
        frame = pd.read_csv(ROOT/'data/raw/bank-additional-full.csv',sep=';').iloc[int(row):int(row)+5]
        bundle = joblib.load(ROOT/'outputs/research/selected_bundle.joblib')
        p = infer(bundle,frame.drop(columns='y'))
        table = frame[['age','job','education','contact']].copy()
        table['predicted_probability'] = p
        table['policy_contact'] = p>=bundle['threshold']
        st.dataframe(table)
        st.caption(f"Policy threshold: {bundle['threshold']:.3f}. Duration and outcome are excluded from inference features.")
with methodology:
    st.markdown((result_dir/'FINDINGS.md').read_text())
    st.subheader('Model card')
    st.markdown((ROOT/'docs/MODEL_CARD.md').read_text())
