""" A dataset parser that reads images from folders

Folders are scannerd recursively to find image files. Labels are based
on the folder hierarchy, just leaf folders by default.

Hacked together by / Copyright 2020 Ross Wightman
"""
import os

from timm.utils.misc import natural_key

from .parser import Parser
from .class_map import load_class_map
from .constants import IMG_EXTENSIONS


def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, name_to_cls=None, leaf_name_only=True, sort=True):
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False, followlinks=True):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types: # FIXME: add our parser
                filenames.append(os.path.join(root, f))
                labels.append(label)

    if name_to_cls is not None:
        # replace the raw labels with specified labels in name_to_cls dict
        new_filenames = []
        new_labels = []
        lens = len(filenames)
        prefix = '/'.join(filenames[0].split('/')[:-2])
        for i in range(lens):
            key = filenames[i].split('/')[-1]
            if key in name_to_cls.keys(): # keep the images in label text only
                new_filenames.append(filenames[i])
                new_labels.append(name_to_cls[key])
        filenames = new_filenames
        labels = new_labels

    if class_to_idx is not None:
        images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
    else:
        images_and_targets = [(f, l) for f, l in zip(filenames, labels)]

    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    
    return images_and_targets, class_to_idx


def load_name2cls_map(map_or_filename):
    if isinstance(map_or_filename, dict):
        assert dict, 'name2cls_map dict must be non-empty'
        return map_or_filename
    name2cls_map_path = map_or_filename
    assert os.path.exists(name2cls_map_path), 'Cannot locate specified class map file (%s)' % map_or_filename

    name2cls_map_ext = os.path.splitext(map_or_filename)[-1].lower()
    if name2cls_map_ext == '.txt':
        name_to_cls_idx = dict()
        with open(name2cls_map_path) as f:
            for l in f.readlines():
                name, idx = l.strip().split(' ')
                # remove all prefix
                name = name.split('/')[-1]
                name_to_cls_idx[name] = int(idx) 
    else:
        assert False, f'Unsupported name2cls map file extension ({name2cls_map_ext}).'
    return name_to_cls_idx

class ParserImageFolderWithName2Cls(Parser):

    def __init__(
            self,
            root,
            class_map=''):
        super().__init__()

        self.root = root
        class_to_idx = None
        # TODO: load image_map
        name_to_cls = load_name2cls_map(root+'.txt')
        # if class_map:
        #     class_to_idx = load_class_map(class_map, root)
        self.samples, self.class_to_idx = find_images_and_targets(root, name_to_cls=name_to_cls)
        if len(self.samples) == 0:
            raise RuntimeError(
                f'Found 0 images in subfolders of {root}. Supported image extensions are {", ".join(IMG_EXTENSIONS)}')

    def __getitem__(self, index):
        path, target = self.samples[index]
        return open(path, 'rb'), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename
