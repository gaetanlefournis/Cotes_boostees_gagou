# Description: Constants used in the project

# URL of the retrieving website
URL_WEPARI_WINAMAX = "https://wepari.fr/indexh.html"
URL_WEPARI_PSEL = "https://wepari.fr/indexpsel.html"
URL_BOOSTED_ODDS_WINAMAX = "https://www.winamax.fr/paris-sportifs/sports/100000"
URL_BOOSTED_ODDS_PSEL = "https://www.enligne.parionssport.fdj.fr/cotes-boostees"

# List of possible sports on Winamax
LIST_SPORTS = ["Football", "Basketball", "Tennis", "Athlétisme", "Automobile", "Aviron", "Badminton", "Baseball", "Beach-volley", "Biathlon", "Boxe", "Canoé-kayak", "Cyclisme", "Escalade", "Escrime", "Equitation", "Football américain", "Football australien", "Formule 1", "Golf", "Haltérophilie", "Handball", "Hockey sur gazon", "Hockey sur glace", "JO", "Judo", "Lutte", "MMA", "Moto", "Natation", "Pentathlon moderne", "Pétanque", "Rugby à XV", "Rugby à XIII", "Rugby à 7", "Ski alpin", "Ski de fond", "Snooker", "Taekwondo", "Tennis de table", "Tir", "Tir à l arc", "Triathlon", "Voile", "Volley-ball", "Water-polo"]

DICO_SPORTS = {
    'f': 'Football',
    'b': 'Basketball',
    't': 'Tennis',
    'a': 'Athlétisme',
    'au': 'Automobile',
    'av': 'Aviron',
    'bm': 'Badminton',
    'ba': 'Baseball',
    'bv': 'Beach-volley',
    'bi': 'Biathlon',
    'bo': 'Boxe',
    'ck': 'Canoé-kayak',
    'c': 'Cyclisme',
    'esca': 'Escalade',
    'escr': 'Escrime',
    'eq': 'Equitation',
    'fa': 'Football américain',
    'fas': 'Football australien',
    'f1': 'Formule 1',
    'g': 'Golf',
    'ha': 'Haltérophilie',
    'h': 'Handball',
    'hga': 'Hockey sur gazon',
    'hg': 'Hockey sur glace',
    'jo': 'JO',
    'ju': 'Judo',
    'lu': 'Lutte',
    'm': 'MMA',
    'mo': 'Moto',
    'n': 'Natation',
    'pe': 'Pentathlon moderne',
    'p': 'Pétanque',
    'r': 'Rugby à XV',
    'rx': 'Rugby à XIII',
    'rr': 'Rugby à 7',
    'sa': 'Ski alpin',
    'sf': 'Ski de fond',
    'sn': 'Snooker',
    'ta': 'Taekwondo',
    'tt': 'Tennis de table',
    'ti': 'Tir',
    'tir': 'Tir à l arc',
    'tr': 'Triathlon',
    'vo': 'Voile',
    'v': 'Volley-ball',
    'wp': 'Water-polo'
}

SPORTS_LOGO = {
    "⚽": "Football",
    "🏀": "Basketball",
    "🏈": "Football américain",
    "⚾": "Baseball",
    "🎾": "Tennis",
    "🏐": "Volley-ball",
    "🏉": "Rugby à XV",
    "🏒": "Hockey sur glace",
    "🥊": "MMA",
    "🤾": "Handball",
}


CONDITIONS_ON_SPORTS = {
    "winamax": 
        {
        "silver" : 
            {
            #"Football": [2.8, 21], 
            "Basketball": [2.5, 12], 
            "Tennis":[3.7, 22], 
            "Baseball":[2.5, 9], 
            "Biathlon":[10, 14], 
            "Football américain":[4.4, 19],
            "Formule 1":[4.4, 12],
            "Handball":[3.1, 16],
            "Hockey sur glace":[2.7, 14],
            "MMA":[2.5, 8],
            "Rugby à XV":[5.4, 20],
            "Rugby à XIII":[2.6, 12],
            "Volley-ball":[3.5, 13],
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
            # "Football": [2.6, 19], 
            # "Basketball": [2.9, 20], 
            # "Tennis":[4.1, 16],
            "Rugby à XIII":[3.4, 17]
            },
        "gold" : 
            {
            "Football": [2.3, 19],
            "Tennis": [3.8, 0],
            }
        }
    }