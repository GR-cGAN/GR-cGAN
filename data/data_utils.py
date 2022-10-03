import os
import logging
import numpy as np
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

logger = logging.getLogger('GAN.Data')


def get_data_path(raw_data_path, name, zip_url=None):
    """ Check if the dataset exists and returns the path.
    Args:
        raw_data_path: (string) path of directory containing the dataset
        name: (string) name of the file
        zip_url: (string) optional - url link to download the dataset
    Returns:
        csv_path: (string) path to the dataset
    """
    csv_path = os.path.join(raw_data_path, name)
    file_exists = os.path.exists(csv_path)
    if not file_exists:
        if zip_url is None:
            logger.info(f'Error: {name} not found!')
        else:
            logger.info(f'Dataset not found! Downloading...')
            with urlopen(zip_url) as zip_resp:
                with ZipFile(BytesIO(zip_resp.read())) as z_file:
                    z_file.extractall(raw_data_path)
    return csv_path


def MinMaxScaler(data):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data
