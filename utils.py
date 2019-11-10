from pathlib import Path
import random
import re
from typing import List, Union, Tuple
import zipfile

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import torch
from tqdm.notebook import tqdm
import wget


def download_wikitext(local_path: Path) -> None:
    if local_path.exists(): return
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
    print(f'Downloading {url}')
    print(f'Saving to {local_path}')
    list(local_path.parents)[0].mkdir(exist_ok=True, parents=True)
    wget.download(
        url,
        str(local_path)
    )


def get_token_string(zip_path, mode='train'):
    assert mode in ('train', 'valid', 'test')
    with zipfile.ZipFile(zip_path) as zfo:
        dirname = zfo.namelist()[0]
        path = Path(f'{dirname}') / f'wiki.{mode}.tokens'
        tokens = zfo.open(str(path)).read().decode()
        return tokens


def clean_token_string(text: str) -> str:
    text = text.lower()
    text = re.sub('\n( =){1,}.*?( =){1,} \n', '', text)
    text = re.sub(r'[ \n]{1,}', ' ', text)
    return text


def tokenize(text: str) -> List[str]:
    return text.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in set(stopwords.words('english'))]


def prepare_data(zip_path, mode, sampling_rate: float = 1., seed: int = None):
    random.seed(seed)
    token_string = get_token_string(zip_path, mode)
    token_string = clean_token_string(token_string)
    sents = [
        sent for sent in sent_tokenize(token_string)
        if random.random() < sampling_rate
        ]
    tokens = [tokenize(sent) for sent in tqdm(sents)]
    return tokens


def load_glove(path: Union[Path, str]) -> Tuple[List[str], torch.Tensor]:
    tokens = []
    embs = []
    with open(path) as fo:
        for line in tqdm(fo):
            token, *emb = line.split()
            emb = list(map(float, emb))
            tokens.append(token)
            embs.append(emb)
    return tokens, torch.tensor(embs)