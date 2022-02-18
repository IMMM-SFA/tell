Contributing to **tell**
========================

Whether you find a typo in the documentation, find a bug, or want to develop functionality that you think will make **tell** more robust, you are welcome to contribute!


Opening issues
______________

If you find a bug or would like to contribute a new feature, please `open an issue <https://github.com/IMMM-SFA/tell/issues>`_.


Contribution workflow
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