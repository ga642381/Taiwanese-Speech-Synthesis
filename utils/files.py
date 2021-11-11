from pathlib import Path
from typing import Union

def get_files(path: Union[str, Path], extension='.wav'):
    '''
        Get all files under the path
        Arguments:
            path: absolute/relative path to search for files
            extension: file extension
    '''
    if isinstance(path, str): path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))
