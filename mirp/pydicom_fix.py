import logging as logger

from pydicom.charset import default_encoding
from pydicom.dataset import Dataset
from pydicom.filereader import data_element_generator
from pydicom.tag import BaseTag, Tag


def read_dataset(fp, is_implicit_VR, is_little_endian, bytelength=None,
                 stop_when=None, defer_size=None,
                 parent_encoding=default_encoding, specific_tags=None):
    """Return a Dataset instance containing the next dataset in the file.

    Parameters
    ----------
    fp : an opened file object
    is_implicit_VR : boolean
        True if file transfer syntax is implicit VR.
    is_little_endian : boolean
        True if file has little endian transfer syntax.
    bytelength : int, None, optional
        None to read until end of file or ItemDeliterTag, else
        a fixed number of bytes to read
    stop_when : None, optional
        optional call_back function which can terminate reading.
        See help for data_element_generator for details
    defer_size : int, None, optional
        Size to avoid loading large elements in memory.
        See ``dcmread`` for more parameter info.
    parent_encoding :
        optional encoding to use as a default in case
        a Specific Character Set (0008,0005) isn't specified
    specific_tags : list or None
        See ``dcmread`` for parameter info.

    Returns
    -------
    a Dataset instance

    See Also
    --------
    pydicom.dataset.Dataset
        A collection (dictionary) of Dicom `DataElement` instances.
    """
    raw_data_elements = dict()
    fpStart = fp.tell()
    de_gen = data_element_generator(fp, is_implicit_VR, is_little_endian,
                                    stop_when, defer_size, parent_encoding,
                                    specific_tags)
    try:
        while (bytelength is None) or (fp.tell() - fpStart < bytelength):
            raw_data_element = next(de_gen)
            # Read data elements. Stop on some errors, but return what was read
            tag = raw_data_element.tag
            # Check for ItemDelimiterTag --dataset is an item in a sequence
            if tag == BaseTag(0xFFFEE00D):
                break
            # Check if list element has length is 0
            if not is_implicit_VR and hasattr(raw_data_element, "length") and raw_data_element.tag < Tag(0x004,0x000):
                if raw_data_element.length == 0:
                    # rewind file pointer
                    fp.seek(raw_data_element.value_tell - 8)
                    break

            raw_data_elements[tag] = raw_data_element
    except StopIteration:
        pass
    except EOFError as details:
        # XXX is this error visible enough to user code with just logging?
        logger.error(str(details) + " in file " +
                     getattr(fp, "name", "<no filename>"))
    except NotImplementedError as details:
        logger.error(details)

    return Dataset(raw_data_elements)
