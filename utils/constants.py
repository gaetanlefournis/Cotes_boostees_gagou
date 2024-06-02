# Description: Constants used in the project

# URL of the retrieving website
URL_WEPARI_WINAMAX = "https://wepari.fr/indexh.html"
URL_WEPARI_PSEL = "https://wepari.fr/indexpsel.html"
URL_BOOSTED_ODDS_WINAMAX = "https://www.winamax.fr/paris-sportifs/sports/100000"
URL_BOOSTED_ODDS_PSEL = "https://www.enligne.parionssport.fdj.fr/cotes-boostees"

# List of possible sports on Winamax
LIST_SPORTS = ["Football", "Basketball", "Tennis", "Athlétisme", "Automobile", "Badminton", "Baseball", "Biathlon", "Boxe", "Cyclisme", "Football américain", "Football australien", "Formule 1", "Golf", "Handball", "Hockey sur glace", "JO", "Judo", "MMA", "Moto", "Natation", "Pétanque", "Rugby à XV", "Rugby à XIII", "Rugby à 7", "Ski alpin", "Ski de fond", "Snooker", "Tennis de table", "Volley-ball", "Water-polo"]

DICO_SPORTS = {
    'f': 'Football',
    'b': 'Basketball',
    't': 'Tennis',
    'a': 'Athlétisme',
    'au': 'Automobile',
    'bm': 'Badminton',
    'ba': 'Baseball',
    'bi': 'Biathlon',
    'bo': 'Boxe',
    'c': 'Cyclisme',
    'fa': 'Football américain',
    'fas': 'Football australien',
    'f1': 'Formule 1',
    'g': 'Golf',
    'h': 'Handball',
    'hg': 'Hockey sur glace',
    'jo': 'JO',
    'ju': 'Judo',
    'm': 'MMA',
    'mo': 'Moto',
    'n': 'Natation',
    'p': 'Pétanque',
    'r': 'Rugby à XV',
    'rx': 'Rugby à XIII',
    'rr': 'Rugby à 7',
    'sa': 'Ski alpin',
    'sf': 'Ski de fond',
    'sn': 'Snooker',
    'tt': 'Tennis de table',
    'v': 'Volley-ball',
    'wp': 'Water-polo'
}

CONDITIONS_ON_SPORTS = {
    "winamax": 
        {
        "silver" : 
            {
            "Football": [2.8, 21], 
            "Basketball": [2.5, 12], 
            "Tennis":[3.7, 22], 
            "Baseball":[2.5, 9], 
            "Biathlon":[10, 9], 
            "Football américain":[4.4, 19],
            "Formule 1":[4.4, 11],
            "Handball":[2.7, 7],
            "Hockey sur glace":[2.7, 14],
            "MMA":[2.5, 8],
            "Rugby à XV":[5.4, 21],
            "Rugby à XIII":[2.6, 12],
            "Volley-ball":[3.5, 11],
            },
        "gold" : 
            {
            "Football": [4.9, 0],
            "Basketball": [4.9, 0],
            "Tennis":[3.2, 0],
            "Rugby à XV":[10, 0],
            "MMA":[10, 0],
            "Handball":[10, 0],
            }
        },
    "PSEL":
        {
        "silver" : 
            {
            "Football": [2.6, 19], 
            "Basketball": [2.9, 20], 
            "Tennis":[3.1, 16],
            },
        "gold" : 
            {
            "Football": [2.6, 0],
            "Tennis": [3.8, 0],
            }
        }
    }