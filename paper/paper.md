---
title: 'tell:  a Python package to model future total electricity loads.'
tags:
  - Python
  - Electricity loads
  - MultiSector Dynamics
authors:
  - name: Casey McGrath
    orcid: 0000-0002-8808-8312
    affiliation: 1
  - name: Casey Burleyson
    affiliation: 1
    orchid: 0000-0001-6218-9361
  - name: Chris R. Vernon
    orcid: 0000-0002-3406-6214
    affiliation: 1
affiliations:
 - name: Pacific Northwest National Laboratory, Richland, WA., USA
   index: 1
date: 20 January 2021
bibliography: paper.bib
---

# Statement of need
 Forecasting changes in electricity loads in response to anthropogenic and natural stressors is necessary for promoting energy system resilience. Given the pressures of aging infrastructure and the increasing integration of renewables, accurate load forecasts are critical for maintaining a stable grid and as a basis for long-term planning. Within the past two decades there have been rapid advances in both short-term (minutes to hours ahead) and long-term (months to years ahead) probabilistic load forecasting approaches. The general structure of these types of models are, understandably, quite different. Short- and medium-term load models most commonly relate meteorology and day-of-week parameters to loads. Longer-term models also use meteorology/climate as explanatory variables, but typically require bringing in “macro” variables like the decadal evolution of population, number of customers, or economic indicators. 

![](ISNE_graphic.png) 
*Figure 1. a. ISO New England (ISNE) observed and predicted hourly electricity demand, as the training (2016-2019) and test period (2019-2020). b. Annual population for New England 2019. c. Fraction of ISNE load as weighted by annual population (state scaling factor), with detailed hourly electricity demand for selected counties.*

# Summary
The Total ELectricity Load (TELL) model provides a framework that integrates aspects of both short- and long-term predictions of electricity demand in a coherent and scalable way. TELL takes as input gridded hourly time-series of meteorology and uses the temporal variations in weather to predict hourly profiles of total electricity demand for every county in the lower 48 United States using a multilayer perceptron (MLP) approach. Hourly predictions from TELL are then scaled to match the annual state-level total electricity loads predicted by the U.S. version of the Global Change Analysis Model (GCAM-USA). GCAM-USA is designed to capture the long-term co-evolution of the human-Earth system. Using this unique approach allows TELL to reflect both changes in the shape of the load profile due to variations in weather and climate and the long-term evolution of energy demand due to changes in population, technology, and economics. TELL is unique from other probabilistic load forecasting models in that it features an explicit spatial component that allows us to relate predicted loads to where they would occur spatially within a grid operations model.

The TELL model will generate predictions of hourly total electricity load for every county in the Continental United States (CONUS). Predictions from TELL will be scaled to match the annual state-level total electricity loads predicted by the U.S. version of the Global Change Analysis Model [GCAM-USA; @iyer2017measuring; @iyer2019improving].

`tell` was implemented to...  

`tell` is offers the following features...

`tell` can be accessed on GitHub (https://github.com/IMMM-SFA/tell). We provide an walkthrough of some key functionality in a step-by-step tutorial in our website here: [Tutorial](link to the readthedocs.io that will have the tutorial).

# Acknowledgements
This research was supported in part by the U.S. Department of Energy, Office of Science, as part of research in MultiSector Dynamics, Earth and Environmental System Modeling Program. The Pacific Northwest National Laboratory is operated for DOE by Battelle Memorial Institute under contract DE-AC05-76RL01830. The views and opinions expressed in this paper are those of the authors alone.

# References
