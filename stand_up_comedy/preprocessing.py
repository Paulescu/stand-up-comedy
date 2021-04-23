import stand_up_comedy.db as db


def preprocess_data():
    """
    Removes artifacts from the transcripts
    """
    ids = db.get_ids_with_transcript()
    print(f'{len(ids)} ids to process..')



preprocess_data()
