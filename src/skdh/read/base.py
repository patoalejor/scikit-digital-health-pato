"""
Base process for file IO

Lukas Adamowicz
Copyright (c) 2021. Pfizer Inc. All rights reserved.
"""
from pathlib import Path
import functools
from warnings import warn


def check_input_file(extension, ext_message="File extension [{}] does not match expected [{}]"):
    """
    Check the input file for existence and suffix.

    Parameters
    ----------
    extension : str
        Expected file suffix, eg '.abc'.
    ext_message : str, optional
        Message to print if the suffix does not match. Should take 2 format arguments
        ('{}'), the first for the actual file suffix, and the second for the
        expected suffix.
    """
    def decorator_check_input_file(func):
        @functools.wraps(func)
        def wrapper_check_input_file(self, file=None, **kwargs):
            # check if the file is provided
            if file is None:
                raise ValueError("`file` must not be None.")

            # make a path instance for ease of use
            pfile = Path(file)
            # make sure the file exists
            if not pfile.exists():
                raise FileNotFoundError(f"File {file} does not exist.")

            # check that the file matches the expected extension
            if pfile.suffix != extension:
                if self.ext_error == 'warn':
                    warn(ext_message.format(pfile.suffix, extension), UserWarning)
                elif self.ext_error == 'raise':
                    raise ValueError(ext_message.format(pfile.suffix, extension))
                elif self.ext_error == 'skip':
                    kwargs.update({'file': file})
                    return (kwargs, None) if self._in_pipeline else kwargs

            return func(self, file=file, **kwargs)

        return wrapper_check_input_file

    return decorator_check_input_file
