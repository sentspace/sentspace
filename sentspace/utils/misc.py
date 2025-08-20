import concurrent.futures
import hashlib
from functools import partial
from itertools import chain
from time import time

# import seaborn as sns
import nltk

# import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from tqdm import tqdm


_START_TIME = time()


def START_TIME():
    return _START_TIME


# download NLTK data if not already downloaded
def download_nltk_resources():
    for category, nltk_resource in [
        ("taggers", "averaged_perceptron_tagger"),
        ("corpora", "wordnet"),
        ("tokenizers", "punkt"),
    ]:
        try:
            nltk.data.find(category + "/" + nltk_resource)
        except LookupError as e:
            try:
                nltk.download(nltk_resource)
            except FileExistsError:
                pass


def md5(fname_or_raw, raw=False) -> str:
    """generates md5sum of the contents of fname
    fname (str): path to file whose md5sum we want
    """
    hash_md5 = hashlib.md5()
    if raw:
        chunk = fname_or_raw.encode("utf-8")
        hash_md5.update(chunk)
        return hash_md5.hexdigest()
    with open(fname_or_raw, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def sha1(ob):
    ob_repr = repr(ob)
    hash_object = hashlib.sha1()
    hash_object.update(ob_repr.encode("utf-8"))
    return hash_object.hexdigest()


def parallelize(
    function, *iterables, wrap_tqdm=True, desc="", max_workers=None, **kwargs
):
    """parallelizes a function by calling it on the supplied iterables and (static) kwargs.
       optionally wraps in tqdm for progress visualization

    Args:
        function ([type]): [description]
        wrap_tqdm (bool, optional): [description]. Defaults to True.
        desc ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    partialfn = partial(function, **kwargs)
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        if wrap_tqdm:
            return [
                *tqdm(
                    executor.map(partialfn, *iterables),
                    total=len(iterables[0]),
                    desc="[parallelized] " + desc,
                )
            ]
        return executor.map(partialfn, *iterables)


# this might be data-dependent
def load_passage_labels(filename):
    """
    Given .mat file, load and return list of passage no. for each sentence
    """
    labelsPassages = sio.loadmat(filename)
    lP = labelsPassages["labelsPassageForEachSentence"]
    return lP.flatten()


# this might be data-dependent
def load_passage_categories(filename):
    """
    Given .mat file, load and return list of passage category labels
    """
    labelsPassages = sio.loadmat(filename)
    lP = labelsPassages["keyPassageCategory"]
    return list(np.hstack(lP[0]))


def get_passage_labels(f1g, lplst):
    """
    Given list of passage no. for each sentence, return list of passage no. (for each word)
    """
    lplst_word = []
    for i, sentence in enumerate(f1g):
        for word in sentence:
            lplst_word.append(lplst[i])
    return lplst_word


# this might be data-dependent
def load_passage_category(filename):
    """
    Given .mat file, return category no. for each passage
    """
    labelsPassageCategory = sio.loadmat(filename)
    lPC = labelsPassageCategory["labelsPassageCategory"]
    lPC = np.hsplit(lPC, 1)
    lpclst = np.array(lPC).tolist()
    lpclst = lpclst[0]
    lpclst = list(chain.from_iterable(lpclst))  # Accessing the nested lists
    return lpclst


def merge_lists(list_a, list_b, feature=""):
    """Input: Two lists with potentially missing values.
    Return: If list 1 contains NA vals, the NA val is replaced by the value in list 2 (either numerical val or np.nan again)
    """
    merged = []

    for val1, val2 in zip(list_a, list_b):
        merged += [val2 if np.isnan(val1) else val1]

    return merged


def sizeof_fmt(num, suffix="B"):
    """
    This function can be used to print out how big a file is
    """
    """ by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified"""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, "Yi", suffix)
