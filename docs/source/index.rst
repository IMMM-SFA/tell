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

`tell` was created to:

  1. Project the short- and long-term evolution of hourly electricity demand in response to changes in weather and climate.
  2. Work at a spatial resolution adequate for input to a unit commitment/economic dispatch (UC/ED) model.
  3. Maintain consistency with the long-term growth and evolution of annual state-level electricity demand projected by an economically driven human-Earth system model.

`tell` is targeted for users that are interested in grid stress modeling. This could include those whose business needs require
accurate load forecasting (electric utilities, regulatory commissions, industrial and big commercial companies, etc.),
researchers interested in energy system transitions, and many others in need of a framework for blending short and long-term
load models with an explicit spatial resolution component.

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
    Quickstarter Notebook<tell_quickstarter.rst>
    User Guide<user_guide>
    API Reference<modules>
    Contributing<contributing>
    License<license>
    Acknowledgement<acknowledgement>
