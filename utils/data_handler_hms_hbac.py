import os

import joblib
import keras_cv
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import tensorflow as tf

from config_hms_hbac import Config


class DataHandler():
    def __init__(self, config: Config):
        self.base_path = "/kaggle/input/hms-harmful-brain-activity-classification"
        self.spec_dir_path = "hms-hbac"
        self.config = config
        self.train_df = None
        self.test_df = None

    def __build_augmenter(self, dim=[400, 300]):
        self.augmenters = [
            keras_cv.layers.MixUp(alpha=2.0),
            keras_cv.layers.RandomCutout(height_factor=(1.0, 1.0),
                                         width_factor=(0.06, 0.1)), # freq-masking
            keras_cv.layers.RandomCutout(height_factor=(0.06, 0.1),
                                         width_factor=(1.0, 1.0)), # time-masking
        ]

        def augment(img, label):
            data = {"images":img, "labels":label}
            for augmenter in self.augmenters:
                if tf.random.uniform([]) < 0.5:
                    data = augmenter(data, training=True)
            return data["images"], data["labels"]

        self.augmenter = augment

    def __build_decoder(self, with_labels=True, target_size=[400, 300], dtype=32):
        def decode_signal(path, offset=None):
            # Read .npy files and process the signal
            file_bytes = tf.io.read_file(path)
            sig = tf.io.decode_raw(file_bytes, tf.float32)
            sig = sig[1024//dtype:]  # Remove header tag
            sig = tf.reshape(sig, [400, -1])

            # Extract labeled subsample from full spectrogram using "offset"
            if offset is not None: 
                offset = offset // 2  # Only odd values are given
                sig = sig[:, offset:offset+300]

                # Pad spectrogram to ensure the same input shape of [400, 300]
                pad_size = tf.math.maximum(0, 300 - tf.shape(sig)[1])
                sig = tf.pad(sig, [[0, 0], [0, pad_size]])
                sig = tf.reshape(sig, [400, 300])

            # Log spectrogram 
            sig = tf.clip_by_value(sig, tf.math.exp(-4.0), tf.math.exp(8.0)) # avoid 0 in log
            sig = tf.math.log(sig)

            # Normalize spectrogram
            sig -= tf.math.reduce_mean(sig)
            sig /= tf.math.reduce_std(sig) + 1e-6

            # Mono channel to 3 channels to use "ImageNet" weights
            sig = tf.tile(sig[..., None], [1, 1, 3])
            return sig

        def decode_label(label):
            label = tf.one_hot(label, self.config.num_classes)
            label = tf.cast(label, tf.float32)
            label = tf.reshape(label, [self.config.num_classes])
            return label

        def decode_with_labels(path, offset=None, label=None):
            sig = decode_signal(path, offset)
            label = decode_label(label)
            return (sig, label)

        self.decoder = decode_with_labels if with_labels else decode_signal

    def __get_dataset(self, paths, offsets=None, labels=None, batch_size=32, cache=True,
                      augment=False, repeat=True, shuffle=1024, 
                      cache_dir="", drop_remainder=False):
        if cache_dir != "" and cache is True:
            os.makedirs(cache_dir, exist_ok=True)

        self.__build_augmenter()
        self.__build_decoder(labels is not None)

        AUTO = tf.data.experimental.AUTOTUNE
        slices = (paths, offsets) if labels is None else (paths, offsets, labels)

        ds = tf.data.Dataset.from_tensor_slices(slices)
        ds = ds.map(self.decoder, num_parallel_calls=AUTO)
        ds = ds.cache(cache_dir) if cache else ds
        ds = ds.repeat() if repeat else ds
        if shuffle: 
            ds = ds.shuffle(shuffle, seed=self.config.seed)
            opt = tf.data.Options()
            opt.experimental_deterministic = False
            ds = ds.with_options(opt)
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
        ds = ds.map(self.augmenter, num_parallel_calls=AUTO) if augment else ds
        ds = ds.prefetch(AUTO)
        return ds
        
    def __make_train_and_test_dirs(self) -> tuple[str, str]:
        """Make directories for storing train and test dataframes."""
        self.train_dir_path = self.spec_dir_path + '/train_spectrograms'
        os.makedirs(self.train_dir, exist_ok=True)
        self.test_dir_path = self.spec_dir_path + '/test_spectrograms'
        os.makedirs(self.test_dir_path, exist_ok=True)

    def __save_spec_as_npy(self, spec_id, split="train"):
        """Save spectrogram data, which is originally formatted as parquet, as npy."""
        spec_path = f"{self.base_path}/{split}_spectrograms/{spec_id}.parquet"
        spec = pd.read_parquet(spec_path)
        spec = spec.fillna(0).values[:, 1:].T # fill NaN values with 0, transpose for (Time, Freq) -> (Freq, Time)
        spec = spec.astype("float32")
        np.save(f"{self.spec_dir_path}/{split}_spectrograms/{spec_id}.npy", spec)

    def __save_spec_as_npy_parallelly(self, df, name):
        """Parallelly save spectrogram data as npy."""
        # Get unique spec_ids of train and valid data
        spec_ids = df["spectrogram_id"].unique()

        # Parallelize the processing using joblib for training data
        _ = joblib.Parallel(n_jobs=-1, backend="loky")(
            joblib.delayed(process_spec)(spec_id, name)
            for spec_id in tqdm(spec_ids, total=len(spec_ids))
        )

    def __set_train_df(self):
        """Set train dataframe."""
        train_df = pd.read_csv(f'{self.base_path}/train.csv')
        train_df['eeg_path'] = f'{self.base_path}/train_eegs/' + train_df['eeg_id'].astype(str) + '.parquet'
        train_df['spec_path'] = f'{self.base_path}/train_spectrograms/' + train_df['spectrogram_id'].astype(str) + '.parquet'
        train_df['spec2_path'] = f'{self.spec_dir_path}/train_spectrograms/' + train_df['spectrogram_id'].astype(str) + '.npy'
        train_df['class_name'] = train_df.expert_consensus.copy()
        train_df['class_label'] = train_df.expert_consensus.map(self.config.name2label)
        self.train_df = train_df

    def __set_test_df(self):
        """Set test dataframe."""
        test_df = pd.read_csv(f'{self.base_path}/test.csv')
        test_df['eeg_path'] = f'{self.base_path}/test_eegs/' + test_df['eeg_id'].astype(str) + '.parquet'
        test_df['spec_path'] = f'{self.base_path}/test_spectrograms/' + test_df['spectrogram_id'].astype(str) + '.parquet'
        test_df['spec2_path'] = f'{self.spec_dir_path}/test_spectrograms/' + test_df['spectrogram_id'].astype(str) + '.npy'
        self.test_df = test_df

    def __set_train_and_test_df(self):
        """Set train and test dataframes."""
        self.__set_train_df()
        self.__set_test_df()
    
    def __set_train_and_validation_ds(self):
        """Set train and validation datasets."""
        if self.train_df is None:
            return
        
        sample_df = self.train_df.groupby("spectrogram_id").head(1).reset_index(drop=True)
        train_df = sample_df[sample_df.fold != self.config.fold]
        valid_df = sample_df[sample_df.fold == self.config.fold]

        # Train
        train_paths = train_df.spec2_path.values
        train_offsets = train_df.spectrogram_label_offset_seconds.values.astype(int)
        train_labels = train_df.class_label.values
        self.train_ds = self.__get_dataset(train_paths, train_offsets, train_labels, batch_size=self.config.batch_size,
                                         repeat=True, shuffle=True, augment=True, cache=True)

        # Valid
        valid_paths = valid_df.spec2_path.values
        valid_offsets = valid_df.spectrogram_label_offset_seconds.values.astype(int)
        valid_labels = valid_df.class_label.values
        self.valid_ds = self.__get_dataset(valid_paths, valid_offsets, valid_labels, batch_size=self.config.batch_size,
                                         repeat=False, shuffle=False, augment=False, cache=True)

    def __split_train_df(self):
        """Split train dataframe."""
        if self.train_df is None:
            return

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.config.seed)

        self.train_df["fold"] = -1
        self.train_df.reset_index(drop=True, inplace=True)
        for fold, (train_idx, valid_idx) in enumerate(
            sgkf.split(self.train_df, y=self.train_df["class_label"], groups=self.train_df["patient_id"])
        ):
            self.train_df.loc[valid_idx, "fold"] = fold

    def save_spec_train_and_test_as_npy_parallelly(self):
        """Parallelly save train and test spectrogram data as npy."""
        self.set_train_and_test_df()
        
        self.__save_spec_as_npy_parallelly(self.train_df, "train")
        self.__save_spec_as_npy_parallelly(self.test_df, "test")
    
    def run(self, use_npy: bool):
        self.__set_train_and_test_df()
        self.__split_train_df()
        self.__set_train_and_validation_ds()