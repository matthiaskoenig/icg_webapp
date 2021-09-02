from typing import Dict

import numpy as np
import streamlit as st
import xarray
import pandas as pd
import altair as alt
from matplotlib import pyplot as plt

from sbmlsim.simulation import Timecourse, TimecourseSim
from sbmlsim.simulator import SimulatorSerial as Simulator

from pathlib import Path

from classification import classification, figure
from icg_simulation import simulate_samples, calculate_pk
from sampling import samples_for_individual

st.sidebar.title('Settings')
st.sidebar.markdown("## Integration")
endtime = st.sidebar.number_input("End time [min]", value=20, min_value=1,
                                  max_value=200, step=10)
steps = st.sidebar.number_input("Steps", value=100, min_value=2, max_value=300,
                                step=10)

st.sidebar.markdown("## Parameters")
icg_dose = st.sidebar.slider("ICG Dose [mg]", value=10, min_value=0, max_value=200,
                             step=5)

'''
# Physiological based pharmacokinetics model of indocyanine green
Personalized prediction of survival after hepatectomy.
'''

resection_rates = np.linspace(0.1, 0.9, num=9)

samples = samples_for_individual(
    bodyweight=75,
    age=55,
    f_cirrhosis=0.4,
    n=200,
    resection_rates=resection_rates,
)

xres, samples = simulate_samples(samples)

print("-" * 80)
print(xres)
samples = calculate_pk(samples=samples, xres=xres)
print(samples.head())

# classification
samples = classification(samples=samples,)


# figure boxplots
figure(samples)


# st.image(
#     image="./model/icg_body.png",
#     width=600,
#     caption="Model overview: A: PBPK model. The whole-body model for ICG consists "
#             "of venous blood, arterial blood, lung, liver, gastrointestinal tract, "
#             "and rest compartment (accounting for organs not modeled in detail). "
#             "The systemic blood circulation connects these compartments. "
#             "B: Liver model. ICG is taken up into the liver tissue (hepatocytes) "
#             "via OATP1B3. The transport was modeled as competitively inhibited by "
#             "plasma bilirubin. Hepatic ICG is excreted in the bile from where it is "
#             "subsequently excreted in the feces. No metabolism of ICG occurs in "
#             "the liver.")


'''
## References
**Physiologically based modeling of the effect of physiological and anthropometric variability on indocyanine green based liver function tests**  
*Adrian Köller, Jan Grzegorzewski and Matthias König*  
bioRxiv 2021.08.11.455999; doi: https://doi.org/10.1101/2021.08.11.455999  
[Submitted to Frontiers Physiology 2021-08-12]

**Prediction of survival after hepatectomy using a physiologically based pharmacokinetic model of indocyanine green liver function tests**  
*Adrian Köller, Jan Grzegorzewski, Michael Tautenhahn, Matthias König*  
bioRxiv 2021.06.15.448411; doi: https://doi.org/10.1101/2021.06.15.448411,  
[Submitted to Frontiers Physiology 2021-06-30]
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
