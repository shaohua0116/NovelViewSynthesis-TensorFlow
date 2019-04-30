import requests
import os.path as osp
from util import log


def download_file_from_google_drive(id, destination):
    """
    This code is partially borrowed from
    https://gist.github.com/charlesreid1/4f3d676b33b95fce83af08e4ec261822
    """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def check_dataset(file_path, dataset_name):
    assert dataset_name in ['car', 'chair', 'kitti', 'synthia']

    if not osp.isfile(file_path):
        log.warn('The {} dataset is not found. '
                 'Downloading the dataset now...'.format(dataset_name))

        if dataset_name == 'car':
            download_file_from_google_drive('1vrZURHH5irKrxPFuw6e9mZ3wh2RqzFC9',
                                            './datasets/shapenet/data_car.hdf5')
        elif dataset_name == 'chair':
            download_file_from_google_drive('1-IbmdJqi37JozGuDJ42IzOFG_ZNAksni',
                                            './datasets/shapenet/data_chair.hdf5')
        elif dataset_name == 'kitti':
            download_file_from_google_drive('1LT3WoHxdCycu4jTxCGc1vGYpdRWridFH',
                                            './datasets/kitti/data_kitti.hdf5')
        elif dataset_name == 'synthia':
            download_file_from_google_drive('1Fxv5r7oeG0PHgR42S5pHNvyl2pJN739H',
                                            './datasets/synthia/data_synthia.hdf5')
        else:
            raise NotImplementedError
    else:
        log.warn('Found {} dataset at {}'.format(dataset_name, file_path))
