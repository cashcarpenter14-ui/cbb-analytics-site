def get_tournament_status(row):
    rating = row.get("Overall_Rating", 0)
    power = row.get("Power_Rating", 0)

    if rating >= 18:
        return "Lock"
    elif rating >= 12:
        return "Likely In"
    elif rating >= 7:
        return "Bubble"
    elif rating >= 3:
        return "Work To Do"
    else:
        return "Likely Out"


def get_projected_seed(row):
    rating = row.get("Overall_Rating", 0)

    if rating >= 24:
        return "1–2 Seed"
    elif rating >= 18:
        return "3–5 Seed"
    elif rating >= 12:
        return "6–9 Seed"
    elif rating >= 7:
        return "10–12 Seed / Bubble"
    else:
        return "Outside Field"