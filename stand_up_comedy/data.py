from time import sleep
import re
from pdb import set_trace as stop

from tqdm.auto import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

import stand_up_comedy.db as db
# from stand_up_comedy.constants import (DATA_DIR, CHROME_DRIVER_PATH)
from stand_up_comedy.exceptions import (InvalidURL)




def scrape_youtube_video_ids(keyword: str):
    """"""
    options = Options()
    options.headless = False

    driver = webdriver.Chrome(executable_path=CHROME_DRIVER_PATH,
                              options=options)

    base_url = 'https://youtube.com'
    driver.get(f'{base_url}/search?q={keyword}')

    def scroll_down_webpage(driver, max_scrolls: int = None):

        if not max_scrolls:
            max_scrolls = 9999

        n_scrolls = 0
        while True:
            scroll_height = 2000
            document_height_before = driver.execute_script("return document.documentElement.scrollHeight")
            driver.execute_script(f"window.scrollTo(0, {document_height_before + scroll_height});")
            sleep(1.5)
            document_height_after = driver.execute_script("return document.documentElement.scrollHeight")
            if document_height_after == document_height_before:
                break

            n_scrolls += 1
            if n_scrolls >= max_scrolls:
                break
    scroll_down_webpage(driver, max_scrolls=3)

    elements = driver.find_elements_by_xpath('//*[@id="thumbnail"]')
    links = [e.get_attribute('href') for e in elements]
    print(f'{len(links)} links scrapped.')
    driver.quit()

    # save scrapped data to DB
    n_valid_video_ids = 0
    for l in links:
        try:
            video_id = get_youtube_video_id_from_url(l)
            n_valid_video_ids += 1

            data = {
                'id': video_id,
                'url': l,
                'keyword': keyword,
            }
            db.save_document(data, overwrite=False)

        except InvalidURL:
            continue

    print(f'{n_valid_video_ids} valid video ids')


def get_youtube_video_id_from_url(
    url: str
) -> str:
    """"""
    url_prefix = 'https://www.youtube.com/watch?v='

    try:
        if url.startswith(url_prefix):
            return url.split(url_prefix)[1]
    except:
        raise InvalidURL


def get_youtube_video_transcript(
    video_id: str,
) -> str:
    """"""
    transcript = YouTubeTranscriptApi.get_transcript(video_id,
                                                     languages=['en-US', 'en'])
    utterances = [p['text'] for p in transcript]
    return ' '.join(utterances)


def scrape_youtube_video_transcripts():
    """
    Scrapes youtube videos using the given 'keyword', fetches their transcript
    (if possible) and saves it locally
    """
    ids = db.get_ids_without_transcript()

    # stop()

    for id in tqdm(ids):
        try:
            transcript = get_youtube_video_transcript(video_id=id)

            db.update_document({
                'id': id,
                'transcript': transcript
            })

        except (
                TranscriptsDisabled,
                # NoTranscriptFound
                ):
            print(f'Transcript is disabled for video_id={id}.')
            continue

        except Exception as e:
            print(e)


def scrape_youtube_video_transcript(video_id: str):
    """
    Scrapes youtube videos using the given 'keyword', fetches their transcript
    (if possible) and saves it locally
    """
    try:
        transcript = get_youtube_video_transcript(video_id=video_id)

        db.update_document({
            'id': video_id,
            'transcript': transcript
        })

    except TranscriptsDisabled:
        print(f'Transcript is disabled for video_id={video_id}.')

        db.update_document({
            'id': video_id,
            'transcript': 'ERROR: Transcript disabled'
        })

    except Exception as e:
        print(e)


def scrape_youtube_video_metadata(video_id: str):
    """
    https://www.youtube.com/watch?v=i_5xPDX-erE
    """
    raise Exception('TODO')


def add_features(video_id: str):
    """"""
    # try:
    document = db.load_document(video_id)
    # except:
    #     stop()

    num_applauses = len(re.findall(r'\b(Applause|Laughter)\b', document['transcript']))
    num_words = len(re.findall(r'\w+', document['transcript']))

    db.update_document({
        'id': document['id'],
        'num_applauses': num_applauses,
        'num_words': num_words,
        'words_per_applause': num_words/num_applauses if num_applauses > 0 else 0,
    })

    # stop()


if __name__ == '__main__':

    import sys
    keyword = sys.argv[1]

    scrape_youtube_video_ids(keyword=keyword)

    video_ids = db.get_ids_without_transcript(keyword=keyword)
    print(len(video_ids))
    for id in video_ids:
        scrape_youtube_video_transcript(id)

    # video_ids = db.get_ids_with_transcript()
    # for id in tqdm(video_ids):
    #     add_features(id)