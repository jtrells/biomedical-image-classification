import pandas as pd
from pathlib import Path
from os import getenv
from dotenv import load_dotenv
from pymongo import MongoClient
from argparse import ArgumentParser
from tqdm import tqdm


def populate_with_dataset(conn_uri, parquet_path, is_curated=False):
    ''' 
        Insert a starting dataset into a new db before any AL 
        calculation. Make sure the DB_CONN is in a .env file.

        Params:
        -------------------
        parquet_path: Location of parquet file
    '''
    path = Path(parquet_path)
    df = pd.read_parquet(path)
    classifier = parquet_path.split("_")[1]

    client = MongoClient(conn_uri)
    images = client['vil'].images

    count = 0
    new_images = []
    images_to_update = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        existing_image = images.find_one({'path': row['img_path']})
        if existing_image:            
            existing_image['classifiers'].append(classifier)
            images.replace_one(
                {'path': row['img_path']}, 
                {'classifiers': existing_image['classifiers']})
        else:
            image = {
                'img': row['img'],
                'path': row['img_path'],
                'src': row['source'],
                'cap': row['caption'],
                'gtr': row['label'],
                'pred': None,
                'set': row['split_set'],
                # 'hits': row['hit'],
                'feats': row['features'].tolist(),
                'probs': None,
                'alms': None,
                'allc': None,
                'alen': None,
                'classifiers': [classifier],
                'doc': row['img_path'].split('//')[-3] if is_curated else None,
                'crop': row['needsCropping'] if is_curated else None,
                'comp': row['isCompound'] if is_curated else None,
                'ovcrop': row['isOvercropped'] if is_curated else None,
                'ovfrag': row['isOverfragmented'] if is_curated else None,
                'bbox': [row['x0'], row['y0'], row['x1'], row['y1']] if is_curated else None,
            }

            # image_id = images.insert_one(image)
            new_images.append(image)
        count += 1
    results_insert = images.insert_many(new_images)
    print(f'{count} images inserted from {parquet_path}')


if __name__ == '__main__':
    parser = ArgumentParser(description="populate database with images")
    parser.add_argument('--parquet_path', type=str, required=True)
    parser.add_argument('--is_curated', type=bool, default=False,
                        help='dataset curated from publications')
    args = parser.parse_args()

    load_dotenv()
    conn_uri = getenv("DB_CONN")
    populate_with_dataset(conn_uri, args.parquet_path,
                          is_curated=args.is_curated)
