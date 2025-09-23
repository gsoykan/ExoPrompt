import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from src.data.greenlight_utils.read_climate_model_gt_data import (
    read_climate_model_simulation_csv_data,
)
from src.utils.greenlight_scaler import GreenlightScaler
from src.utils.timefeatures import time_features


class GreenlightGtTimeSeriesDataset(Dataset):
    # TODO: @gsoykan - try to update frequency - to minutes or 5 minutes
    """
    Main Difference from Custom dataset will be it will take 20 (or more) features
    and outputs only 3 (t, co2, vp)
    """

    def __init__(
        self,
        root_path,
        flag="train",
        size=(96, 48, 96),
        features="M",
        data_path="greenlight_gt_timeseries.csv",
        target="OT",  # redundant in our context
        scale=True,
        timeenc=1,
        freq="t",
        # s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
        output_feature_idx: Optional[Tuple] = (7, 8, 9),
        use_greenlight_scaler: bool = True,
        gl_simulation_csv_path: Optional[str] = None,
        return_all_output_features: bool = False,
        train_test_split_rate: Tuple[int, int] = (0.7, 0.2),
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        assert (
            (train_test_split_rate[0] + train_test_split_rate[1]) <= 1.0
        ), "Train and test rate sum should be lower than 1, so that validation has value"
        self.train_test_split_rate = train_test_split_rate

        self.features = features
        self.target = target
        self.scale = scale
        self.use_greenlight_scaler = use_greenlight_scaler
        self.timeenc = timeenc
        self.freq = freq
        self.output_feature_idx = output_feature_idx
        self.gl_simulation_csv_path = gl_simulation_csv_path
        self.return_all_output_features = return_all_output_features

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = (
            GreenlightScaler() if self.use_greenlight_scaler else StandardScaler()
        )
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        add_sim_data = self.gl_simulation_csv_path is not None
        if add_sim_data:
            _, indoor_sim_df, _, crop_df, aux_states_df = (
                read_climate_model_simulation_csv_data(self.gl_simulation_csv_path)
            )
            output_raw_simulation_df = indoor_sim_df
            output_raw_simulation_df.drop(columns=["time"], inplace=True)

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        cols = list(df_raw.columns)
        if self.target in cols:
            cols.remove(self.target)
        cols.remove("date")

        if self.target in cols:
            df_raw = df_raw[["date"] + cols + [self.target]]
        else:
            df_raw = df_raw[["date"] + cols]

        num_train = int(len(df_raw) * self.train_test_split_rate[0])
        num_test = int(len(df_raw) * self.train_test_split_rate[1])
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # for testing on all dataset (special case)
        if self.train_test_split_rate == (0, 1.0):
            if border1 == 0 and border2 == 0:
                border2 = self.seq_len + self.pred_len
            elif border1 < 0 and border2 == 0:
                border1 = 0
                border2 = self.seq_len + self.pred_len
            elif border1 < 0 < border2:
                border1 = 0

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        df_data = df_data.astype("float32")
        if add_sim_data:
            output_raw_simulation_df = output_raw_simulation_df.astype("float32")

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            if isinstance(self.scaler, StandardScaler):
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
                # TODO: @gsoykan - handle scaling of simulation values
            else:
                df_data = self.scaler.transform(df_data)
                data = df_data.values
                if add_sim_data:
                    df_sim_data = self.scaler.transform(
                        output_raw_simulation_df, is_only_output=True
                    )
                    sim_data = df_sim_data.values
        else:
            data = df_data.values
            if add_sim_data:
                sim_data = output_raw_simulation_df.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)
        data_stamp = data_stamp.astype("float32")

        self.data_y_all = None
        self.data_x = data[border1:border2]
        if self.output_feature_idx is None:
            self.data_y = data[border1:border2]
        else:
            self.data_y = data[border1:border2, self.output_feature_idx]
            if self.return_all_output_features:
                self.data_y_all = data[border1:border2]

        self.data_y_sim = None
        if add_sim_data:
            self.data_y_sim = sim_data[border1:border2]

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        instance = {
            "seq_x": seq_x,
            "seq_y": seq_y,
            "seq_x_mark": seq_x_mark,
            "seq_y_mark": seq_y_mark,
        }

        # TODO: a question should the model know about the x_sim too?
        if self.data_y_sim is not None:
            seq_x_sim = self.data_y_sim[s_begin:s_end]
            seq_y_sim = self.data_y_sim[r_begin:r_end]
            instance = {
                **instance,
                "seq_x_sim": seq_x_sim,
                "seq_y_sim": seq_y_sim,
            }

        if self.data_y_all is not None:
            instance["seq_y_all"] = self.data_y_all[r_begin:r_end]

        return instance

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        data_last_dim = data.shape[-1]
        train_last_dim = self.data_x.shape[-1]
        if data_last_dim != train_last_dim and (
            data_last_dim == len(self.output_feature_idx)
            if self.output_feature_idx is not None
            else False
        ):
            # it means it is test data so fill the rest with zero
            filler = np.zeros((*data.shape[:-1], train_last_dim), dtype=data.dtype)
            filler[:, self.output_feature_idx] = data
            inverse_filler = self.scaler.inverse_transform(filler)
            inverse_data = inverse_filler[:, self.output_feature_idx]
            return inverse_data
        else:
            return self.scaler.inverse_transform(data)


if __name__ == "__main__":
    gt_dataset = GreenlightGtTimeSeriesDataset(
        root_path="/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/physinet/data/from_david_by_gurkan",
        flag="train",
        features="M",
        data_path="greenlight_gt_timeseries.csv",
        gl_simulation_csv_path="/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/physinet/data/from_david/Simulation data/CSV output/climateModel_hps_manuscriptParams.csv",
    )
    sample_instance = gt_dataset[0]
    print(sample_instance)
