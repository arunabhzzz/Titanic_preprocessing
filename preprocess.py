def check(row):
    return 0


def commontitle(title):
    common=['Mr','Miss','Master']
    for titlemaybe in common:
        if titlemaybe in title:
            return titlemaybe
    return "rare"


    