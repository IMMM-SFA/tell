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

.. grid:: 4
    :gutter: 3

    .. grid-item-card:: Getting started
        :img-top: _static/index_info.svg
        :class-img-top: sd-px-4, sd-py-4
        :text-align: center

        New to `tell`? Get familiar with what `tell` is all about.

        +++
        .. button-ref:: README
                :click-parent:
                :color: primary
                :expand: 

                Getting started

    .. grid-item-card:: Quickstarter
        :img-top: _static/index_quickstarter.svg
        :class-img-top: sd-px-4, sd-py-4
        :text-align: center

        Follow along with this Jupyter notebook to learn the ropes of `tell`.

        +++
        .. button-ref:: tell_quickstarter
                :click-parent:
                :color: primary
                :expand: 

                Quickstarter

    .. grid-item-card:: User Guide
        :img-top: _static/index_user_guide.svg
        :class-img-top: sd-px-4, sd-py-4
        :text-align: center

        The user guide provides in-depth information on the key concepts of `tell`.

        +++
        .. button-ref:: user_guide
                :click-parent:
                :color: primary
                :expand: 

                User Guide

    .. grid-item-card:: API reference
        :img-top: _static/index_api.svg
        :class-img-top: sd-px-4, sd-py-4
        :text-align: center

        A detailed description of the `tell` API.

        +++
        .. button-ref:: modules
                :click-parent:
                :color: primary
                :expand: 

                API    


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
