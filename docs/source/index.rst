.. TELL documentation master file, created by
   sphinx-quickstart on Fri Feb 18 07:50:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


:notoc:

.. module:: tell

************************
TELL documentation
************************

**Date**: |today| **Version**: |version|

**Useful links**:
`Source Repository <https://github.com/immm-sfa/tell>`__ |
`Issues & Ideas <https://github.com/immm-sfa/tell/issues>`__

`tell` is an open-source Python package for predicting future Total ELectricity Loads (TELL) in the Lower 48 United States.

`tell` was created to:
   1. Predict future hourly electricity demand by blending aspects of short-term (minutes to hours ahead) and long term (months to years ahead) load forecasting methods.
   2. Include an explicit spatial component that allows users to relate the predicted loads to where they would occur spatially within a grid operations model.
   3. Be implemented as a nationwide model that can be applied uniformly in any of the three grid interconnections.

.. panels::
    :card: + intro-card text-center
    :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 d-flex

    >>>
    :img-top: _static/cognitive.svg

    Getting started
    ^^^

    New to `tell`? Get familiar with what `tell` is all about.

    +++
    .. link-button:: README
            :type: ref
            :text: Getting started
            :classes: btn-block btn-secondary stretched-link

    >>>
    :img-top: _static/education.svg

    Quickstarter
    ^^^

    Follow along with this Jupyter notebook to learn the ropes of `tell`.

    +++
    .. link-button:: tell_quickstarter
            :type: ref
            :text: Quickstarter
            :classes: btn-block btn-secondary stretched-link

    >>>
    :img-top: _static/soccer.svg

    User Guide
    ^^^

    The user guide provides in-depth information on the key concepts of `tell`.

    +++
    .. link-button:: user_guide
            :type: ref
            :text: User Guide
            :classes: btn-block btn-secondary stretched-link

    >>>
    :img-top: _static/api.svg

    API reference
    ^^^

    A detailed description of the `tell` API.

    +++
    .. link-button:: modules
            :type: ref
            :text: API
            :classes: btn-block btn-secondary stretched-link


.. toctree::
    :maxdepth: 1
    :hidden:
    :titlesonly:

    Getting Started<README.md>
    Quickstarter Notebook<tell_quickstarter.ipynb>
    User Guide<user_guide>
    API Reference<modules>
    Contributing<contributing>
    License<license>
    Acknowledgement<acknowledgement>
