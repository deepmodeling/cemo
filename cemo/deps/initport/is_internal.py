def is_internal(file_name: str) -> bool:
    """
    Check whether a file name starts with an underscore.

    Args:
        file_name: The name of a file.
    
    Returns:
        True if the file name starts with an underscore.
    """
    return file_name.startswith("_")
