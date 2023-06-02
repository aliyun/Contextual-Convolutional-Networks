import os

from .parser_image_folder import ParserImageFolder
from .parser_image_tar import ParserImageTar
from .parser_image_in_tar import ParserImageInTar
from .parser_image_folder_with_name2cls import ParserImageFolderWithName2Cls


def create_parser(name, root, split='train', **kwargs):
    name = name.lower()
    name = name.split('/', 2)
    prefix = ''
    if len(name) > 1:
        prefix = name[0]
    name = name[-1]

    # FIXME improve the selection right now just tfds prefix or fallback path, will need options to
    # explicitly select other options shortly
    if prefix == 'tfds':
        from .parser_tfds import ParserTfds  # defer tensorflow import
        parser = ParserTfds(root, name, split=split, **kwargs)
    elif 'name2cls' in prefix  or 'name2cls' in name:
        assert os.path.exists(root)
        parser = ParserImageFolderWithName2Cls(root, **kwargs)
    else:
        assert os.path.exists(root)
        # default fallback path (backwards compat), use image tar if root is a .tar file, otherwise image folder
        # FIXME support split here, in parser?
        if os.path.isfile(root) and os.path.splitext(root)[1] == '.tar':
            parser = ParserImageInTar(root, **kwargs)
        else:
            parser = ParserImageFolder(root, **kwargs)
    return parser
