# Prediction of LC50 value using Quantitative structure–activity relationship models (QSAR models)

# Notebook link
https://hariprasath-ai.github.io/LC50_Project/Notebook_html.html

# Description:
Thousands of chemical substances for which no ecological toxicity data are available
can benefit from QSAR modelling to help prioritise testing. One of the data set
encompassing in vivo test data on fish for hundreds of chemical substances using the
ECOTOX database of the US Environmental Protection Agency, you can check that
dataset through this link: ECOTOX Database and additional data from ECHA. We can
utilise this to develop QSAR models that could forecast two sorts of end points: acute
LC50 (median lethal concentration) and points of departure akin to the NOEC (no
observed effect concentration) for any period (the “LC50” and “NOEC” models,
respectively). Study factors, such as species and exposure route, were incorporated as
features in these models to allow for the simultaneous use of many data types. To
maximise generalizability to other species, a novel way of substituting taxonomic
categories for species dummy variables was introduced.
The goal here is to build an end-to-end automated Machine Learning model that
predicts the LC50 value, the concentration of a compound that causes 50% lethality of
fish in a test batch over a duration of 96 hours, using 6 given molecular descriptors.

# Dataset link:
https://archive.ics.uci.edu/dataset/504/qsar+fish+toxicity

# Additional Information

Data set containing values for 6 attributes (molecular descriptors) of 908 chemicals used to predict quantitative acute aquatic toxicity towards the fish Pimephales promelas (fathead minnow).

This dataset was used to develop quantitative regression QSAR models to predict acute aquatic toxicity towards the fish Pimephales promelas (fathead minnow) on a set of 908 chemicals. LC50 data, which is the concentration that causes death in 50% of test fish over a test duration of 96 hours, was used as model response. The model comprised 6 molecular descriptors: MLOGP (molecular properties), CIC0 (information indices), GATS1i (2D autocorrelations), NdssC (atom-type counts), NdsCH ((atom-type counts), SM1_Dz(Z) (2D matrix-based descriptors). Details can be found in the quoted reference: M. Cassotti, D. Ballabio, R. Todeschini, V. Consonni. A similarity-based QSAR model for predicting acute toxicity towards the fathead minnow (Pimephales promelas), SAR and QSAR in Environmental Research (2015), 26, 217-243; doi: 10.1080/1062936X.2015.1018938

6 molecular descriptors and 1 quantitative experimental response:
1) CIC0
2) SM1_Dz(Z)
3) GATS1i
4) NdsCH
5) NdssC
6) MLOGP
7) quantitative response, LC50 [-LOG(mol/L)]
