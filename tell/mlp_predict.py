import os
import warnings

import joblib
import sklearn


def load_model(model_file: str) -> object:
    """Pickle model to file using joblib.  Version of scikit-learn is included in the file name as a compatible
    version is required to reload the data safely.

    :param model_file:                  Full path with filename an extension to the joblib pickled model file.
    :type model_file:                   str

    :return:                            Model as an object.

    """

    # get version of scikit-learn and compare with the model from file to ensure compatibility
    sk_model_version = os.path.splitext(model_file)[0].split('-')[-1]

    # get version of scikit-learn being used during runtime
    sk_run_version = sklearn.__version__

    if sk_model_version != sk_run_version:
        msg = f"WARNING: Incompatible scikit-learn version for saved model ({sk_model_version}) and current version ({sk_run_version})."
        warnings.warn(msg)

    # load model from
    return joblib.load(model_file)
