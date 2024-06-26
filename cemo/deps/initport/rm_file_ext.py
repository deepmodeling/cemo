import os


def rm_file_ext(file_name: str) -> str:
    """
    Remove the extension from a file name.

    Args:
        file_name: The name of a file.

    Returns:
        The file name without the extension.
    """
    return os.path.splitext(file_name)[0]
