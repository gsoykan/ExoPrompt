import pickle
from typing import Any


# Sources to compare
# https://stackoverflow.com/questions/30329726/fastest-save-and-load-options-for-a-numpy-array
class PickleHelper:
    test_batch = "test_batch.obj"
    exoprompt_old_world_timeseries_batch = "exoprompt_old_world_timeseries_batch.obj"
    exoprompt_new_world_timeseries_batch = "exoprompt_new_world_timeseries_batch.obj"
    exoprompt_gt_timeseries_batch = "exoprompt_gt_timeseries_batch.obj"

    @staticmethod
    def save_object(filename: str, object: Any):
        filehandler = open(filename, "wb")
        pickle.dump(object, filehandler)
        filehandler.close()

    @staticmethod
    def load_object(filename: str):
        filehandler = open(filename, "rb")
        object = pickle.load(filehandler)
        filehandler.close()
        return object
