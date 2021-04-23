from tqdm.auto import tqdm

from stand_up_comedy.data import add_features
import stand_up_comedy.db as db

video_ids = db.get_ids_with_transcript()
for id in tqdm(video_ids):
    add_features(id)