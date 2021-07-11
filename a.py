from typing_extensions import Literal
import utils.data as data_util
import models.bases as base_models

data_util.load(.7, .3)
test = base_models.Test()
print(test.test())