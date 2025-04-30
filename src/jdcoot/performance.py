from .models.continuous_partial_coot import continuous_partial_coot
from .models.continuous_partial_jdcoot import continuous_partial_jdcoot
from .models.continuous_partial_reference import continuous_partial_reference
from .models.continuous_semisupervised_coot import continuous_semisupervised_coot
from .models.continuous_semisupervised_jdcoot import continuous_semisupervised_jdcoot
from .models.continuous_semisupervised_reference import continuous_semisupervised_reference
from .models.continuous_unsupervised_coot import continuous_unsupervised_coot
from .models.continuous_unsupervised_jdcoot import continuous_unsupervised_jdcoot
from .models.discrete_partial_coot import discrete_partial_coot
from .models.discrete_partial_jdcoot import discrete_partial_jdcoot
from .models.discrete_partial_reference import discrete_partial_reference
from .models.discrete_semisupervised_coot import discrete_semisupervised_coot
from .models.discrete_semisupervised_jdcoot import discrete_semisupervised_jdcoot
from .models.discrete_semisupervised_reference import discrete_semisupervised_reference
from .models.discrete_unsupervised_coot import discrete_unsupervised_coot
from .models.discrete_unsupervised_jdcoot import discrete_unsupervised_jdcoot

models = dict()
models[("continuous", "unsupervised", "coot")] = continuous_unsupervised_coot
models[("continuous", "unsupervised", "jdcoot")] = continuous_unsupervised_jdcoot
models[("continuous", "semisupervised", "coot")] = continuous_semisupervised_coot
models[("continuous", "semisupervised", "jdcoot")] = continuous_semisupervised_jdcoot
models[("continuous", "semisupervised", "reference")] = continuous_semisupervised_reference
models[("continuous", "partial", "coot")] = continuous_partial_coot             
models[("continuous", "partial", "jdcoot")] = continuous_partial_jdcoot
models[("continuous", "partial", "reference")] = continuous_partial_reference
models[("discrete", "unsupervised", "coot")] = discrete_unsupervised_coot
models[("discrete", "unsupervised", "jdcoot")]= discrete_unsupervised_jdcoot
models[("discrete", "semisupervised", "coot")] = discrete_semisupervised_coot
models[("discrete", "semisupervised", "jdcoot")] = discrete_semisupervised_jdcoot
models[("discrete", "semisupervised", "reference")] = discrete_semisupervised_reference
models[("discrete", "partial", "coot")] = discrete_partial_coot
models[("discrete", "partial", "jdcoot")] = discrete_partial_jdcoot
models[("discrete", "partial", "reference")] = discrete_partial_reference

variable_types = ["continuous", "discrete"]

learning_methods = ["unsupervised", "semisupervised", "partial"]

recoding_methods = ["coot", "jdcoot", "reference"]
