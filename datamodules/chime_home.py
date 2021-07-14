import csv
import os
import pickle
import wave

import numpy as np
import librosa
import scipy
from tqdm import tqdm
from torch.utils.data import Dataset

from datamodules.utils import FeatureNormalizer


class ChimeHomeDatamodule(object):
    @property
    def folds(self):
        return range(1, self.evaluation_folds + 1)

    def __init__(self,
                 data_root_path: str,
                 evaluation_folds: int = 5):

        self.data_root_path = data_root_path
        self.meta_file = os.path.join(data_root_path, 'chime_home', 'meta.txt')
        self.evaluation_setup_path = os.path.join(data_root_path, 'chime_home', 'evaluation_setup')
        self.evaluation_folds = evaluation_folds
        self.feature_path = os.path.join(data_root_path, 'chime_home', 'features')
        self.feature_normalizer_path = os.path.join(data_root_path, 'chime_home', 'feature_normalizers')

        self.files = None
        self.meta_data = None
        self.evaluation_data_train = {}
        self.evaluation_data_test = {}
        self.audio_extensions = {'wav', 'flac'}
        self.sampling_rate = '16kHz'
        self.package_list = [
            {
                'remote_package':
                    'https://archive.org/download/chime-home/chime_home.tar.gz',
                'local_package':
                    os.path.join(data_root_path, 'chime_home.tar.gz'),
                'local_audio_path':
                    os.path.join(data_root_path, 'chime_home', 'chunks'),
                'development_chunks_refined_csv':
                    os.path.join(data_root_path, 'chime_home', 'development_chunks_refined.csv'),
                'development_chunks_refined_crossval_csv':
                    os.path.join(data_root_path, 'chime_home', 'development_chunks_refined_crossval_dcase2016.csv'),
            },
        ]

    def fetch(self):
        print('-' * 40, '\t1. Make metadata\t', '-' * 40)
        self.make_dataset_metadata()
        print('Done!')

        print('-' * 40, '\t2. Extract and save features\t', '-' * 40)
        self.extract_and_save_features()

        print('-' * 40, '\t3. Normalize\t', '-' * 40)
        self.normalize_features()

    @property
    def audio_tags(self):
        tags = []
        for item in self.meta:
            if 'tags' in item:
                for tag in item['tags']:
                    if tag and tag not in tags:
                        tags.append(tag)
        tags.sort()
        return tags

    @property
    def audio_files(self):
        """
        Get all audio files in the dataset, use only file from CHime-Home-refined set.
        :return: file list with absolute paths
        """
        if self.files is None:
            refined_files = []
            with open(self.package_list[0]['development_chunks_refined_csv'], 'rt') as f:
                for row in csv.reader(f, delimiter=','):
                    refined_files.append(row[1])
            if 'evaluation_chunks_refined_csv' in self.package_list[0].keys():
                with open(self.package_list[0]['evaluation_chunks_refined_csv'], 'rt') as f:
                    for row in csv.reader(f, delimiter=','):
                        refined_files.append(row[1])

            self.files = []
            for file in self.package_list:
                path = file['local_audio_path']
                if path:
                    l = os.listdir(path)
                    p = path.replace(self.data_root_path + os.path.sep, '')
                    for f in l:
                        file_name, file_extension = os.path.splitext(f)
                        file_name, sampling_rate = os.path.splitext(file_name)
                        if file_extension[1:] in self.audio_extensions \
                                and file_name in refined_files \
                                and sampling_rate[1:] in self.sampling_rate:
                            self.files.append(os.path.abspath(os.path.join(path, f)))

            self.files.sort()
        return self.files

    @staticmethod
    def read_chunk_meta(meta_filename):
        if os.path.isfile(meta_filename):
            meta_file_handle = open(meta_filename, 'rt')
            try:
                meta_file_reader = csv.reader(meta_file_handle, delimiter=',')
                data = {}
                for meta_file_row in meta_file_reader:
                    data[meta_file_row[0]] = meta_file_row[1]
            finally:
                meta_file_handle.close()
            return data
        else:
            return None

    @staticmethod
    def tag_code_to_tag_label(tag):
        label_map = {'c': 'child speech',
                     'm': 'adult male speech',
                     'f': 'adult female speech',
                     'v': 'video game/tv',
                     'p': 'percussive sound',
                     'b': 'broadband noise',
                     'o': 'other',
                     'S': 'silence/background',
                     'U': 'unidentifiable'}
        if tag in label_map:
            return label_map[tag]
        else:
            return None

    @property
    def meta(self):
        if self.meta_data is None:
            self.meta_data = []
            meta_id = 0
            if os.path.isfile(self.meta_file):
                f = open(self.meta_file, 'rt')
                try:
                    reader = csv.reader(f, delimiter='\t')
                    for row in reader:
                        if len(row) == 2:
                            # Scene meta
                            self.meta_data.append({'file': row[0], 'scene_label': row[1].rstrip().strip()})
                        elif len(row) == 4:
                            # Audio tagging meta
                            self.meta_data.append(
                                {'file': row[0], 'scene_label': row[1].rstrip().strip(), 'tag_string': row[2],
                                 'tags': row[3].split(';')})
                        elif len(row) == 6:
                            # Event meta
                            self.meta_data.append({'file': row[0],
                                                   'scene_label': row[1].rstrip().strip(),
                                                   'event_onset': float(row[2]),
                                                   'event_offset': float(row[3]),
                                                   'event_label': row[4],
                                                   'event_type': row[5],
                                                   'id': meta_id})
                        meta_id += 1
                finally:
                    f.close()
            else:
                raise IOError("Meta file missing [%s]" % self.meta_file)

        return self.meta_data

    def file_meta(self, file):
        file = self.absolute_to_relative(file)
        file_meta = []
        for item in self.meta:
            if item['file'] == file:
                file_meta.append(item)

        return file_meta

    def relative_to_absolute_path(self, path):
        return os.path.abspath(os.path.join(self.data_root_path, 'chime_home', path))

    def absolute_to_relative(self, path):
        if path.startswith(os.path.abspath(os.path.join(self.data_root_path, 'chime_home'))):
            return os.path.relpath(path, os.path.join(self.data_root_path, 'chime_home'))
        else:
            return path

    def make_dataset_metadata(self):
        # Make legacy dataset compatible with DCASE2016 dataset scheme
        if not os.path.isfile(self.meta_file):
            print('Generating meta file for dataset')

            scene_label = 'home'
            f = open(self.meta_file, 'wt')
            try:
                writer = csv.writer(f, delimiter='\t')
                for file in self.audio_files:
                    raw_path, raw_filename = os.path.split(file)
                    relative_path = self.absolute_to_relative(raw_path)
                    base_filename, file_extension = os.path.splitext(raw_filename)
                    base_filename, sampling_rate = os.path.splitext(base_filename)
                    annotation_filename = os.path.join(raw_path, base_filename + '.csv')
                    meta_data = self.read_chunk_meta(annotation_filename)
                    tags = []

                    for i, tag in enumerate(meta_data['majorityvote']):
                        if tag is not 'S' and tag is not 'U':
                            tags.append(tag)
                    tags = ';'.join(tags)
                    writer.writerow(
                        (os.path.join(relative_path, raw_filename), scene_label, meta_data['majorityvote'], tags))
            finally:
                f.close()

        all_folds_found = False
        if not all_folds_found:
            if not os.path.isdir(self.evaluation_setup_path):
                os.makedirs(self.evaluation_setup_path)

            files, fold_assignments = [], []
            with open(self.package_list[0]['development_chunks_refined_crossval_csv'], 'rt') as f:
                for row in csv.reader(f, delimiter=','):
                    files.append(self.relative_to_absolute_path(
                        os.path.join('chunks', row[1] + '.' + self.sampling_rate + '.wav')))
                    fold_assignments.append(int(row[2]))
            files = np.array(files)
            fold_assignments = np.array(fold_assignments)

            for fold in np.unique(fold_assignments):
                train_files = files[fold_assignments != fold]
                test_files = files[fold_assignments == fold]

                with open(os.path.join(self.evaluation_setup_path, 'fold' + str(fold + 1) + '_train.txt'), 'wt') as f:
                    writer = csv.writer(f, delimiter='\t')
                    for file in train_files:
                        raw_path, raw_filename = os.path.split(file)
                        relative_path = self.absolute_to_relative(raw_path)
                        item = self.file_meta(file)[0]
                        writer.writerow([
                            os.path.join(relative_path, raw_filename),
                            item['scene_label'],
                            item['tag_string'],
                            ';'.join(item['tags'])
                        ])

                with open(os.path.join(self.evaluation_setup_path, 'fold' + str(fold + 1) + '_test.txt'), 'wt') as f:
                    writer = csv.writer(f, delimiter='\t')
                    for file in test_files:
                        raw_path, raw_filename = os.path.split(file)
                        relative_path = self.absolute_to_relative(raw_path)
                        writer.writerow([os.path.join(relative_path, raw_filename)])

                with open(os.path.join(self.evaluation_setup_path,
                                       'fold' + str(fold + 1) + '_evaluate.txt'), 'wt') as f:
                    writer = csv.writer(f, delimiter='\t')
                    for file in test_files:
                        raw_path, raw_filename = os.path.split(file)
                        relative_path = self.absolute_to_relative(raw_path)
                        item = self.file_meta(file)[0]
                        writer.writerow(
                            [os.path.join(relative_path, raw_filename), item['scene_label'], item['tag_string'],
                             ';'.join(item['tags'])])

    def get_train_data_path(self, fold=0):
        if fold not in self.evaluation_data_train:
            self.evaluation_data_train[fold] = []
            if fold > 0:
                with open(os.path.join(self.evaluation_setup_path, 'fold' + str(fold) + '_train.txt'), 'rt') as f:
                    for row in csv.reader(f, delimiter='\t'):
                        if len(row) == 2:
                            # Scene meta
                            self.evaluation_data_train[fold].append({
                                'file': self.relative_to_absolute_path(row[0]),
                                'scene_label': row[1]
                            })
                        elif len(row) == 4:
                            # Audio tagging meta
                            self.evaluation_data_train[fold].append({
                                'file': self.relative_to_absolute_path(row[0]),
                                'scene_label': row[1],
                                'tag_string': row[2],
                                'tags': row[3].split(';')
                            })
                        elif len(row) == 5:
                            # Event meta
                            self.evaluation_data_train[fold].append({
                                'file': self.relative_to_absolute_path(row[0]),
                                'scene_label': row[1],
                                'event_onset': float(row[2]),
                                'event_offset': float(row[3]),
                                'event_label': row[4]
                            })
            else:
                data = []
                for item in self.meta:
                    if 'event_label' in item:
                        data.append({'file': self.relative_to_absolute_path(item['file']),
                                     'scene_label': item['scene_label'],
                                     'event_onset': item['event_onset'],
                                     'event_offset': item['event_offset'],
                                     'event_label': item['event_label'],
                                     })
                    else:
                        data.append({'file': self.relative_to_absolute_path(item['file']),
                                     'scene_label': item['scene_label']
                                     })
                self.evaluation_data_train[0] = data

        return self.evaluation_data_train[fold]

    def get_test_data_path(self, fold=0):
        if fold not in self.evaluation_data_test:
            self.evaluation_data_test[fold] = []
            if fold > 0:
                with open(os.path.join(self.evaluation_setup_path, 'fold' + str(fold) + '_test.txt'), 'rt') as f:
                    for row in csv.reader(f, delimiter='\t'):
                        self.evaluation_data_test[fold].append({'file': self.relative_to_absolute_path(row[0])})
            else:
                data = []
                files = []
                for item in self.meta:
                    if self.relative_to_absolute_path(item['file']) not in files:
                        data.append({'file': self.relative_to_absolute_path(item['file'])})
                        files.append(self.relative_to_absolute_path(item['file']))

                self.evaluation_data_test[fold] = data

        return self.evaluation_data_test[fold]

    @staticmethod
    def get_feature_filename(audio_file, path, extension='pkl'):
        return os.path.join(path, os.path.splitext(audio_file)[0] + '.' + extension)

    @staticmethod
    def load_audio(filename, mono=True, fs=44100):
        file_base, file_extension = os.path.splitext(filename)
        if file_extension == '.wav':
            audio_file = wave.open(filename)

            # Audio info
            sample_rate = audio_file.getframerate()
            sample_width = audio_file.getsampwidth()
            number_of_channels = audio_file.getnchannels()
            number_of_frames = audio_file.getnframes()

            # Read raw bytes
            data = audio_file.readframes(number_of_frames)
            audio_file.close()

            # Convert bytes based on sample_width
            num_samples, remainder = divmod(len(data), sample_width * number_of_channels)
            if remainder > 0:
                raise ValueError('The length of data is not a multiple of sample size * number of channels.')
            if sample_width > 4:
                raise ValueError('Sample size cannot be bigger than 4 bytes.')

            if sample_width == 3:
                # 24 bit audio
                a = np.empty((num_samples, number_of_channels, 4), dtype=np.uint8)
                raw_bytes = np.frombuffer(data, dtype=np.uint8)
                a[:, :, :sample_width] = raw_bytes.reshape(-1, number_of_channels, sample_width)
                a[:, :, sample_width:] = (a[:, :, sample_width - 1:sample_width] >> 7) * 255
                array = a.view('<i4').reshape(a.shape[:-1]).T
            else:
                # 8 bit samples are stored as unsigned ints; others as signed ints.
                dt_char = 'u' if sample_width == 1 else 'i'
                a = np.frombuffer(data, dtype='<%s%d' % (dt_char, sample_width))
                array = a.reshape(-1, number_of_channels).T

            if mono:
                # Down-mix audio
                array = np.mean(array, axis=0)

            # Convert int values into float
            array = array / float(2 ** (sample_width * 8 - 1) + 1)

            if (fs != sample_rate):
                array = librosa.core.resample(array, sample_rate, fs)
                sample_rate = fs

            return array, sample_rate

        elif file_extension == '.flac':
            array, sample_rate = librosa.load(filename, sr=fs, mono=mono)

            return array, sample_rate

        return None, None

    @staticmethod
    def extract_feature(y=None,
                        fs=None,
                        statistics=True,
                        include_mfcc0=True,
                        include_delta=True,
                        include_acceleration=True,
                        mfcc_params=None,
                        delta_params=None,
                        acceleration_params=None):

        # Extract features, Mel Frequency Cepstral Coefficients
        eps = np.spacing(1)

        # Default mfcc configurations
        if mfcc_params is None:
            mfcc_params = {
                'window': 'hamming_asymmetric',  # [hann_asymmetric, hamming_asymmetric]
                'win_length': int(0.064 * fs),
                'hop_length': int(0.01 * fs),
                'n_mfcc': 20,  # Number of MFCC coefficients
                'n_mels': 64,  # Number of MEL bands used
                'n_fft': 1024,  # FFT length
                'fmin': 0,  # Minimum frequency when constructing MEL bands
                'fmax': fs // 2,  # Maximum frequency when constructing MEL band
                'htk': False,
            }

        # Windowing function
        if mfcc_params['window'] == 'hamming_asymmetric':
            window = scipy.signal.hamming(mfcc_params['n_fft'], sym=False)
        elif mfcc_params['window'] == 'hamming_symmetric':
            window = scipy.signal.hamming(mfcc_params['n_fft'], sym=True)
        elif mfcc_params['window'] == 'hann_asymmetric':
            window = scipy.signal.hann(mfcc_params['n_fft'], sym=False)
        elif mfcc_params['window'] == 'hann_symmetric':
            window = scipy.signal.hann(mfcc_params['n_fft'], sym=True)
        else:
            window = None

        # Calculate Static Coefficients
        magnitude_spectrogram = np.abs(librosa.stft(y + eps,
                                                    n_fft=mfcc_params['n_fft'],
                                                    win_length=mfcc_params['win_length'],
                                                    hop_length=mfcc_params['hop_length'],
                                                    window=window)) ** 2

        mel_basis = librosa.filters.mel(sr=fs,
                                        n_fft=mfcc_params['n_fft'],
                                        n_mels=mfcc_params['n_mels'],
                                        fmin=mfcc_params['fmin'],
                                        fmax=mfcc_params['fmax'],
                                        htk=mfcc_params['htk'])

        mel_spectrum = np.dot(mel_basis, magnitude_spectrogram)

        mfcc = librosa.feature.mfcc(
            S=librosa.amplitude_to_db(mel_spectrum),
            n_mfcc=mfcc_params['n_mfcc']
        )

        # Collect the feature matrix
        feature_matrix = mfcc
        if include_delta:
            if delta_params is None:
                delta_params = {
                    'width': 9,
                }

            # Delta coefficients
            mfcc_delta = librosa.feature.delta(mfcc, **delta_params)

            # Add Delta Coefficients to feature matrix
            feature_matrix = np.vstack((feature_matrix, mfcc_delta))

        if include_acceleration:
            if acceleration_params is None:
                acceleration_params = {
                    'width': 9,
                }

            # Acceleration coefficients (aka delta)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2, **acceleration_params)

            # Add Acceleration Coefficients to feature matrix
            feature_matrix = np.vstack((feature_matrix, mfcc_delta2))

        if not include_mfcc0:
            # Omit mfcc0
            feature_matrix = feature_matrix[1:, :]

        # Collect into data structure
        if statistics:
            return {
                'feat': feature_matrix,
                'stat': {
                    'mean': np.mean(feature_matrix, axis=0),
                    'std': np.std(feature_matrix, axis=0),
                    'N': feature_matrix.shape[0],
                    'S1': np.sum(feature_matrix, axis=0),
                    'S2': np.sum(feature_matrix ** 2, axis=0),
                }
            }
        else:
            return {'feat': feature_matrix}

    @staticmethod
    def save_data(file, data):
        pickle.dump(data, open(file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def extract_and_save_features(self,
                                  sample_rate: int = 16000,
                                  overwrite: bool = False,
                                  include_mfcc0: bool = False,
                                  include_delta: bool = False,
                                  include_acceleration: bool = False):
        if not os.path.isdir(self.feature_path):
            os.makedirs(self.feature_path)

        files = []
        for fold in self.folds:
            for item in self.get_train_data_path(fold):
                if item['file'] not in files:
                    files.append(item['file'])
            for item in self.get_test_data_path(fold):
                if item['file'] not in files:
                    files.append(item['file'])
        files = sorted(files)

        for file_id, audio_filename in enumerate(tqdm(files, desc='Feature Extraction')):
            # Get feature filename
            current_feature_file = self.get_feature_filename(
                audio_file=os.path.split(audio_filename)[1],
                path=self.feature_path
            )

            if not os.path.isfile(current_feature_file) or overwrite:
                # Load audio
                if os.path.isfile(self.relative_to_absolute_path(audio_filename)):
                    y, fs = self.load_audio(
                        filename=self.relative_to_absolute_path(audio_filename),
                        mono=True,
                        fs=sample_rate
                    )
                else:
                    raise IOError("Audio file not found [%s]" % audio_filename)

            # Extract features
            feature_data = self.extract_feature(y=y,
                                                fs=fs,
                                                include_mfcc0=include_mfcc0,
                                                include_delta=include_delta,
                                                include_acceleration=include_acceleration)

            # Save
            self.save_data(current_feature_file, feature_data)

    @staticmethod
    def get_feature_normalizer_filename(fold, path, extension='pkl'):
        return os.path.join(path, 'scale_fold' + str(fold) + '.' + extension)

    @staticmethod
    def load_data(file):
        return pickle.load(open(file, "rb"))

    def normalize_features(self,
                           overwrite=False):

        # Check that target path exists, create if not
        if not os.path.isdir(self.feature_normalizer_path):
            os.makedirs(self.feature_normalizer_path)

        for fold in self.folds:
            current_normalizer_file = self.get_feature_normalizer_filename(
                fold=fold,
                path=self.feature_normalizer_path
            )

            files = []
            if not os.path.isfile(current_normalizer_file) or overwrite:
                # Initialize statistics
                for item_id, item in enumerate(self.get_train_data_path(fold)):
                    if item['file'] not in files:
                        files.append(item['file'])

                file_count = len(files)
                normalizer = FeatureNormalizer()

                for file_id, audio_filename in enumerate(tqdm(files, desc="fold {} Normalizer".format(fold))):

                    # Load features
                    feature_filename = self.get_feature_filename(
                        audio_file=os.path.split(audio_filename)[1],
                        path=self.feature_path
                    )

                    if os.path.isfile(feature_filename):
                        feature_data = self.load_data(feature_filename)['stat']
                    else:
                        raise IOError("Features missing [%s]" % audio_filename)

                    # Accumulate statistics
                    normalizer.accumulate(feature_data)

                # Calculate normalization factors
                normalizer.finalize()

                # Save
                self.save_data(current_normalizer_file, normalizer)


class ChimeDataset(Dataset):
    chime_home_labels = [
        'c',  # Child speech
        'm',  # Adult male speech
        'f',  # Adult female speech
        'v',  # Video game / TV
        'p',  # Percussive sounds, e.g.crash, bang, knock, footsteps
        'b',  # Broadband noise, e.g.household appliances
        'o',  # Other identifiable sounds
        '',  # Silence
    ]

    def relative_to_absolute_path(self, path):
        return os.path.abspath(os.path.join(self.data_root_path, 'chime_home', path))

    @staticmethod
    def get_feature_filename(audio_file, path, extension='pkl'):
        return os.path.join(path, os.path.splitext(audio_file)[0] + '.' + extension)

    @staticmethod
    def get_feature_normalizer_filename(fold, path, extension='pkl'):
        return os.path.join(path, 'scale_fold' + str(fold) + '.' + extension)

    @staticmethod
    def load_data(file):
        return pickle.load(open(file, "rb"))

    @property
    def num_classes(self):
        return self._num_classes

    def __init__(self,
                 fold: int,
                 data_root_path: str,
                 data_split: str):
        assert data_split in ['train', 'evaluate']

        self.data_root_path = data_root_path
        self.evaluation_setup_path = os.path.join(data_root_path, 'chime_home', 'evaluation_setup')
        self.feature_path = os.path.join(data_root_path, 'chime_home', 'features')

        feature_normalizer_path = os.path.join(data_root_path, 'chime_home', 'feature_normalizers')
        feature_normalizer_filename = self.get_feature_normalizer_filename(fold=fold, path=feature_normalizer_path)
        if os.path.isfile(feature_normalizer_filename):
            self.normalizer = self.load_data(feature_normalizer_filename)
        else:
            raise IOError("Feature normalizer missing [%s]" % feature_normalizer_filename)

        self.evaluation_data_train = []
        with open(os.path.join(self.evaluation_setup_path,
                               'fold' + str(fold) + '_{}.txt'.format(data_split)), 'rt') as f:
            for row in csv.reader(f, delimiter='\t'):
                # Audio tagging meta
                self.evaluation_data_train.append({
                    'file': self.relative_to_absolute_path(row[0]),
                    'scene_label': row[1],
                    'tag_string': row[2],
                    'tags': row[3].split(';')
                })

        self._num_classes = len(self.chime_home_labels)

    def __len__(self):
        return len(self.evaluation_data_train)

    def __getitem__(self, idx):
        data = self.evaluation_data_train[idx]
        labels = data['tags']

        # Load features
        feature_filename = self.get_feature_filename(audio_file=os.path.split(data['file'])[1],
                                                     path=self.feature_path)
        if os.path.isfile(feature_filename):
            feature_data = self.load_data(feature_filename)['feat']
        else:
            raise IOError("Features missing [%s]" % feature_filename)

        # Normalize features
        feature_data = self.normalizer.normalize(feature_data)
        labels = [self.chime_home_labels.index(label) for label in labels]
        encoded_labels = np.eye(self.num_classes)[labels]
        encoded_labels = np.sum(encoded_labels, axis=0)
        return feature_data, encoded_labels
