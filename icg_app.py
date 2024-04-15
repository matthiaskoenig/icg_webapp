from typing import Dict

import numpy as np
import streamlit as st


from settings import icg_model_path


import sampling
from simulation import simulate_samples, calculate_pk, load_model

# from classification import classification
# from visualization import figure_boxplot, figure_histograms

# np.random.seed(42)

'''
# Physiological based pharmacokinetics model of indocyanine green
'''
col1, col2 = st.columns(2)
col1.image(
    image="./model/icg_body.png",
)
col2.caption(
    "Model overview: A: PBPK model. The whole-body model for ICG consists "
    "of venous blood, arterial blood, lung, liver, gastrointestinal tract, "
    "and rest compartment (accounting for organs not modeled in detail). "
    "The systemic blood circulation connects these compartments. "
    "B: Liver model. ICG is taken up into the liver tissue (hepatocytes) "
    "via OATP1B3. The transport was modeled as competitively inhibited by "
    "plasma bilirubin. Hepatic ICG is excreted in the bile from where it is "
    "subsequently excreted in the feces. No metabolism of ICG occurs in "
    "the liver."
)
'''
## Personalized prediction of survival after hepatectomy
'''
# --- Input data ---
n_samples = 100
f_cirrhosis = 0.0
st.markdown("### Patient")

col1, col2, col3 = st.columns(3)
col1.markdown("**Liver disease**")
cpt = col1.radio("Liver cirrhosis", ('No cirrhosis', 'Mild (CPT A)', 'Moderate (CPT B)', 'Severe (CPT C)'))

col2.markdown("**Anthropometric parameters**")
bodyweight = col2.slider(
    "Body weight [kg]",
    value=75, min_value=30, max_value=140, step=1
)
age = col2.slider(
    "Age [yr]",
    value=55, min_value=18, max_value=84, step=1
)

col3.markdown("**Liver parameters**")
liver_volume = col3.number_input(
    "Liver volume [ml]",
    min_value=500.0, max_value=3000.0,
    value=np.NaN
)
liver_bloodflow = col3.number_input(
    "Hepatic blood flow [ml/min]",
    min_value=500.0, max_value=3000.0,
    value=np.NaN
)
if cpt == 'No cirrhosis':
    f_cirrhosis = 0.0
elif cpt == 'Mild (CPT A)':
    f_cirrhosis = 0.41
elif cpt == 'Moderate (CPT B)':
    f_cirrhosis = 0.72
elif cpt == 'Severe (CPT C)':
    f_cirrhosis = 0.82

simulator = load_model(icg_model_path)
resection_rates = np.linspace(0.1, 0.9, num=9)

samples = sampling.samples_for_individual(
    bodyweight=bodyweight,
    age=age,
    liver_bloodflow=liver_bloodflow,
    liver_volume=liver_volume,
    f_cirrhosis=f_cirrhosis,
    n=n_samples,
    resection_rates=resection_rates
)

#
# #@st.cache
# def simulate_and_classify(samples):
#     """Simulate and classify the subject."""
#     data = samples.copy()
#     xres, data = simulate_samples(data, simulator)
#     data = calculate_pk(samples=data, xres=xres)
#     data = classification(samples=data)
#     return data
#
# data = simulate_and_classify(samples)
#
# st.markdown("### Personalized predictions")
# # figure_histograms
# fig_histograms = figure_histograms(data)
# col1, col2, col3 = st.columns(3)
#
# col1.pyplot(fig=fig_histograms["FOATP1B3"], clear_figure=False)
# col2.pyplot(fig=fig_histograms["LIVVOL"], clear_figure=False)
# col3.pyplot(fig=fig_histograms["LIVBF"], clear_figure=False)
#
#
# # figure boxplots
# fig_boxplots = figure_boxplot(data)
#
# col1, col2 = st.columns(2)
# col1.pyplot(fig=fig_boxplots["postop_r15_model"], clear_figure=False, bbox_inches="tight")
# col2.pyplot(fig=fig_boxplots["y_score"], clear_figure=False)


# --- References ---
'''
## References
**Prediction of survival after hepatectomy using a physiologically based pharmacokinetic model of indocyanine green 
liver function tests**  
*Adrian Köller, Jan Grzegorzewski, Michael Tautenhahn, Matthias König*  
bioRxiv 2021.06.15.448411; doi: https://doi.org/10.1101/2021.06.15.448411,  
[Submitted to Frontiers Physiology 2021-06-30]

**Physiologically based modeling of the effect of physiological and anthropometric variability on indocyanine green 
based liver function tests**  
*Adrian Köller, Jan Grzegorzewski and Matthias König*  
bioRxiv 2021.08.11.455999; doi: https://doi.org/10.1101/2021.08.11.455999  
[Submitted to Frontiers Physiology 2021-08-12]
'''
st.markdown("## Disclaimer")
st.caption("The software is provided **AS IS**, without warranty of any kind, express or implied, "
           "including but not limited to the warranties of merchantability, "
           "fitness for a particular purpose and noninfringement. In no event shall the "
           "authors or copyright holders be liable for any claim, damages or other liability, "
           "whether in an action of contract, tort or otherwise, arising from, out of or in "
           "connection with the software or the use or other dealings in the software. "
           ""
           "This software is a research proof-of-principle and not fit for any clinical application."
           )
