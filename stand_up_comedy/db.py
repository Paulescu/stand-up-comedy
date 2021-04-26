from pathlib import Path
import json
import random
from typing import (Dict, List)
import os
from pdb import set_trace as stop

# from stand_up_comedy.constants import DATA_DIR

DATA_DIR = os.getenv('DATA_DIR')
print('DATA_DIR: ', DATA_DIR)


def id2path(
    id: str,
    collection: str = 'raw',
) -> Path:
    return Path(DATA_DIR) / collection / f'{id}.json'


def save_document(
    data: Dict,
    overwrite: bool = True,
    collection: str = 'raw',
):
    """"""
    id = data['id']

    if (not overwrite) and exists_document(id):
        return

    file = Path(DATA_DIR) / collection / f'{id}.json'
    with open(file, "w") as f:
        json.dump(data, f)


def load_document(
    id: str,
    collection: str = 'raw',
) -> Dict:
    """"""
    file = Path(DATA_DIR) / collection / f'{id}.json'
    with open(file) as f:
        return json.load(f)


def update_document(
    data: Dict,
):
    """"""
    if not exists_document(data['id']):
        save_document(data)

    else:
        document = load_document(data['id'])

        for key, value in data.items():
            document[key] = value

        save_document(document)


def delete_document(id: str):

    os.remove(id2path(id))
    print(f'Removed {id}')


def exists_document(
    id: str,
    collection: str = 'raw',
) -> bool:
    file = Path(DATA_DIR) / collection / f'{id}.json'
    return file.exists()


def get_ids_without_transcript(keyword: str = None) -> List[str]:
    """"""
    ids = get_ids(keyword=keyword)
    no_transcript_ids = []
    for id in ids:
        document = load_document(id)
        if ('transcript' not in document) or document['transcript'].startswith('ERROR'):
            no_transcript_ids.append(id)

    return no_transcript_ids


def get_transcript(id: str) -> str:
    """"""
    return load_document(id)['transcript']


def get_ids_with_transcript(keyword: str = None) -> List[str]:
    """"""
    ids = get_ids(keyword=keyword)
    with_transcript_ids = []
    for id in ids:
        document = load_document(id)
        if ('transcript' in document) and (not document['transcript'].startswith('ERROR')):
            with_transcript_ids.append(id)

    return with_transcript_ids


def get_ids(
    keyword: str = None,
    collection: str = 'raw',
) -> List[str]:
    """Returns list of all video ids in the database"""
    all_ids = [x.name.split('.')[-2] for x in (Path(DATA_DIR) / collection).glob('*.json')]
    output_ids = []
    for id in all_ids:
        document = load_document(id)

        if not keyword:
            output_ids.append(id)

        elif keyword and (document.get('keyword', '') == keyword):
            output_ids.append(id)


    return output_ids


def get_stats() -> Dict:
    """"""
    n_ids_with_transcript = len(get_ids_with_transcript())

    stats = {
        'id': n_ids_with_transcript,
        'keyword_distribution': get_keyword_distribution()
    }

    return stats


def get_random_id(
    keyword: str = None
) -> str:
    """"""
    return random.choice(get_ids_with_transcript(keyword=keyword))


def delete_documents_by_keyword(keyword: str):
    ids = get_ids(keyword=keyword)
    for id in ids:
        delete_document(id)


def get_keyword_distribution() -> Dict:
    """"""
    ids = get_ids_with_transcript()
    stats = {}
    for id in ids:
        document = load_document(id)
        kw = document['keyword']
        stats[kw] = stats.get(kw, 0) + 1

    stats = {k: v for k, v in sorted(stats.items(), key=lambda item: item[1])}
    # stats = sorted(stats, key=stats.get, reverse=True)
    return stats


def get_ml_dataset() -> List[Dict]:
    """"""
    dataset = []
    ids = get_ids_with_transcript()
    for id in ids:
        document = load_document(id)
        dataset.append({
            'keyword': document['keyword'],
            'text': document['transcript'],
        })
    return dataset