import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sentspace.utils
from sentspace.utils import io, text
from sentspace.utils.caching import cache_to_disk, cache_to_mem
from sentspace.utils.utils import Word, merge_lists, wordnet
from tqdm import tqdm


# --------- GloVe
def lowercase(f1g):
    """
    Return lowercase version of input (assume input is a list of token lists)
    """
    return [[token.lower() for token in sent] for sent in f1g]


def get_sent_version(version, df):
    """
    Return a list of sentences as lists of tokens given dataframe & version of token to use
    Options for version: 'raw', 'cleaned', 'lemmatized'
    """
    ref = {'raw': 'Word', 'cleaned': 'Word cleaned', 'lemmatized': 'Word lemma'}
    version = ref[version]
    f1g = []
    for i in df['Sentence no.'].unique():
        f1g.append(list(df[df['Sentence no.'] == i].sort_values('Word no. within sentence')[version]))
    return f1g


def get_vocab(token_list):
    """
    Return set of unique tokens in input (assume input is a list of tokens)
    """
    return set(t for t in token_list)


def download_embeddings(which='glove.840B.300d.txt'):
    raise NotImplementedError()
    if 'glove' in which:
        url = 'https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip'


@cache_to_mem
def load_embeddings(emb_file: str = 'glove.840B.300d.txt',
                    data_dir: Path = None,
                    vocab: tuple = ()):
    """
    Read through the embedding file to find embeddings for target words in vocab
    Return dict mapping word to embedding (numpy array)
    """
    try:
        data_dir = Path(data_dir)
    except TypeError:
        data_dir = Path(__file__).parent / '..' / '..' / '.feature_database/'
    
    vocab = set(vocab)
    OOV = set(vocab)

    io.log(f"loading embeddings from {emb_file} for vocab of size {len(vocab)}")
    w2v = {}
    with (data_dir / emb_file).open('r') as f:
        total_lines = sum(1 for _ in tqdm(f, desc=f'counting # of lines in {data_dir/emb_file}'))
    with (data_dir / emb_file).open('r') as f:
        for line in tqdm(f, total=total_lines, desc=f'loading {len(vocab)} embeddings'):
            token, *emb = line.split(' ')
            if token in vocab:
                # print(f'found {token}!')
                w2v[token] = np.asarray(emb, dtype=float)
                OOV.remove(token)
    
    io.log(f"---done--- loading embeddings from {emb_file}. OOV count: {len(OOV)}/{len(vocab)}")
    io.log(f"           a selection of up to 100 OOV tokens: {[*OOV][:100]}")

    return w2v


def get_word_embeds(token_list, w2v, which='glove', dims=300, return_NA_words=False, save=False, save_path=False):
    """[summary]

    Args:
        tokenized ([type]): [description]
        w2v ([type]): [description]
        embedding (str, optional): [description]. Defaults to 'glove'.
        dims (int, optional): [description]. Defaults to 300.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """    
    #
    """
    Return dataframe of each word, sentence no., and its glove embedding
    If embedding does not exist for a word, fill cells with np.nan
    Parameters:
        f1g: list of sentences as lists of tokens
        w2v: dict mapping word to embedding
        return_NA_words: optionally return unique words that are NA
        save: whether to save results
        save_path: path to save, support .csv & .mat files
    """

    embeddings = []
    
    OOV_words = set()
    for token in token_list:
        if token in w2v[which]:
            embeddings.append(w2v[which][token])
        else:
            embeddings.append([np.nan]*dims)
            OOV_words.add(token)
    
    return embeddings

    flat_token_list = sentspace.utils.text.get_flat_tokens(f1g)
    flat_sentence_num = sentspace.utils.text.get_flat_sentence_num(f1g)
    df = pd.DataFrame(glove_embed)
    df.insert(0, 'Sentence no.', flat_sentence_num)
    df.insert(0, 'Word', flat_token_list)

    print(f'Number of words with NA glove embedding: {len(NA_words)},',
          f'{len(NA_words)/len(flat_token_list)*100:.2f}%')
    print('Example NA words:', NA_words[:5])
    print('-'*79)

    if save:
        suffix = save_path.rsplit('.', -1)[1]
        if suffix == 'csv':
            df.to_csv(save_path, index=False)
        elif suffix == 'mat':
            sio.savemat(save_path, {'glove_word': df})
        else:
            raise ValueError('File type not supported!')

    if return_NA_words:
        return df, set(NA_words)
    else:
        return df



def pool_sentence_embeds(tokens, token_embeddings, filters=[lambda i, x: True],
                         which='glove'):
    """pools embeddings of an entire sentence (given as a list of embeddings)
       using averaging, maxpooling, minpooling, etc., after applying all the
       provided filters as functions (such as content words only).

    Args:
        token_embeddings (list[np.array]): [description]
        filters (list[function[(idx, token) -> bool]], optional): [description]. Defaults to [lambda x: True].
            filters should be functions that map token to bool (e.g. is_content_word(...))
            only tokens that satisfy all filters are retained.

    Returns:
        dict: averaging method -> averaged embedding
    """                         

    """
    Return dataframe of each sentence no. and its sentence embedding
    from averaging embeddings of words in a sentence (ignore NAs)
    Parameters:
        df: dataframe, output of get_glove_word()
        content_only: if True, use content words only
        is_content_lst: list, values 1 if token is content word, 0 otherwise
        save: whether to save results
        save_path: path to save, support .csv & .mat files
    """
    # if content_only:
    #     df = df[np.array(is_content_lst) == 1]

    all_pooled = {}
    for which in token_embeddings:
        filtered_embeds = [e for i, (t, e) in enumerate(zip(tokens, token_embeddings[which]))
                           if all(fn(i, t) for fn in filters)]
        filtered_embeds = np.array(filtered_embeds, dtype=np.float32)
        filtered_embeds = filtered_embeds[~np.isnan(filtered_embeds[:,0])]
        
        if len(filtered_embeds) == 0:
            warnings.warn(f'all embeddings for current sentence ({tokens}) are NaN', ValueError)
            filtered_embeds = np.zeros((1,))

        pooled = {
            'pooled_'+which+'_median': np.median(filtered_embeds, axis=0).reshape(-1).tolist(),
            'pooled_'+which+'_mean': filtered_embeds.mean(axis=0).reshape(-1).tolist(),
            'pooled_'+which+'_max': filtered_embeds.max(axis=0).reshape(-1).tolist(),
            'pooled_'+which+'_min': filtered_embeds.min(axis=0).reshape(-1).tolist(),
        }

        all_pooled.update(pooled)

    return all_pooled

    sent_vectors = df.drop(columns=['Word']).groupby('Sentence no.').mean()  # ignores nans

    na_frac = len(df.dropna())/len(df)
    print(f'Fraction of words used for sentence embeddings: {na_frac*100:.2f}%')
    print('-'*79)

    if save:
        suffix = save_path.rsplit('.', -1)[1]
        if suffix == 'csv':
            sent_vectors.to_csv(save_path)
        elif suffix == 'mat':
            sio.savemat(save_path, {'glove_sent': sent_vectors})
        else:
            raise ValueError('File type not supported!')
    return sent_vectors


def compile_results_for_glove_only(wordlst, wordlst_l, wordlst_lem,
                                   taglst, is_content_lst, setlst,
                                   snlst, wordlen):
    """
    Return dataframe: each row is a word & its various associated values
    """
    result = pd.DataFrame({'Word': wordlst})
    result['Word cleaned'] = wordlst_l
    result['Word lemma'] = wordlst_lem

    result['POS'] = taglst
    result['Content/function'] = is_content_lst
    result['Set no.'] = setlst
    result['Sentence no.'] = snlst
    result['Specific topic'] = ['']*len(wordlst)
    result['Word length'] = wordlen

    return result
