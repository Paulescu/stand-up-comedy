import json

def pprint(x):
    print(json.dumps(x, indent=4, sort_keys=True))


def play_video(video_id):
    from IPython.display import HTML

    # Youtube
    return HTML(
        f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')


