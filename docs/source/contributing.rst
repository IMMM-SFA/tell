Contributing to **tell**
========================

Whether you find a typo in the documentation, find a bug, or want to develop functionality that you think will make **tell** more robust, you are welcome to contribute!


Opening issues
______________

If you find a bug or would like to contribute a new feature, please `open an issue <https://github.com/IMMM-SFA/tell/issues>`_.

[Issues]: https://github.com/IMMM-SFA/tell/issues
[new Issue]: https://github.com/IMMM-SFA/tell/issues/new/choose
[new Pull Request]: https://github.com/IMMM-SFA/tell/compare
[IM3]: https://im3.pnnl.gov/

Contribution workflow (bug or feature)
______________

__I found a bug!__
* Check if the bug has already been reported by searching the existing GitHub [Issues]. If you find a match, consider adding additional details to the existing ticket.
* Open a [new Issue], being sure to include a clear title and description along with as much detail as possible; code samples or log messages demonstrating the bug are quite helpful.

__I fixed a bug!__
* Open a [new Pull Request] with the fix. Ensure the description clearly outlines the bug and the solution. Include the Issue number if applicable.

__I created a new feature!__
* Consider opening a [new Issue] to describe use cases for the new feature. This will offer a platform for discussion and critique.
* Then, open a [new Pull Request] with clear documentation of the methodology. Be sure to include new unit tests if appropriate.


Contribution workflow (pull request)
_____________________

The following is the recommended workflow for contributing to **tell**:

1. `Fork the tell repository <https://github.com/IMMM-SFA/tell/fork>`_ and then clone it locally:

  .. code-block:: bash

    git clone https://github.com/<your-user-name>/tell


2. Create a branch for your changes

  .. code-block:: bash

    git checkout -b bug/some-bug

    # or

    git checkout -b feature/some-feature

3. Add your recommended changes and ensure all tests pass, then commit your changes

  .. code-block:: bash

    git commit -m '<my short message>'

4. Push your changes to the remote

  .. code-block:: bash

    git push origin <my-branch-name>

5. Submit a pull request with the following information:

  - **Purpose**:  The reason for your pull request in short
  - **Summary**:  A description of the environment you are using (OS, Python version, etc.), logic, any caveats, and a summary of changes that were made.