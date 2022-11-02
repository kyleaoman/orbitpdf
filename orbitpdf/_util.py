from datetime import datetime


def _log(*logmsgs):

    T = datetime.now()
    timer = " [{0:02d}:{1:02d}:{2:02d}]".format(T.hour, T.minute, T.second)
    print(*((timer,) + logmsgs))

    return
