import csv
from addict import Dict
from playaid.constants import PARAMS_LABELS


# Ontology of this data
ONTOLOGY = {
    # animations/file names shared across characters
    "all": {
        "Jab": {
            "param_string": ["attack_1"],
            "raw_animations": ["c00attack1"],
            "advantage_state": "neutral",
        },
        "DashAttack": {
            "param_string": ["attack_dash"],
            "raw_animations": ["c00attackdash"],
            "advantage_state": "neutral",
        },
        "ForwardTilt": {
            "param_string": ["attack_s3"],
            "raw_animations": ["c01attacks"],
            "advantage_state": "neutral",
        },
        "DownTilt": {
            "param_string": ["attack_lw3"],
            "raw_animations": ["c02attacklw"],
            "advantage_state": "neutral",
        },
        "UpTilt": {
            "param_string": ["attack_hi3"],
            "raw_animations": ["c02attackhi"],
            "advantage_state": "neutral",
        },
        "ForwardSmash": {
            "param_string": ["attack_s4"],
            "raw_animations": ["c03attacks4"],
            "advantage_state": "neutral",
        },
        "DownSmash": {
            "param_string": ["attack_lw4"],
            "raw_animations": ["c04attacklw"],
            "advantage_state": "neutral",
        },
        "UpSmash": {
            "param_string": ["attack_hi4"],
            "raw_animations": ["c04attackhi"],
            "advantage_state": "neutral",
        },
        "NeutralSpecial": {
            "param_string": ["special_n", "special_air_n"],
            "raw_animations": ["d00special"],
            "advantage_state": "neutral",
        },
        "ForwardSpecial": {
            "param_string": ["special_s", "special_air_s"],
            "raw_animations": ["d01special"],
            "advantage_state": "neutral",
        },
        "DownSpecial": {
            "param_string": ["special_lw", "special_air_lw"],
            "raw_animations": ["d03special"],
            "advantage_state": "neutral",
        },
        "UpSpecial": {
            "param_string": ["special_hi", "special_air_hi"],
            "raw_animations": ["d02special"],
            "advantage_state": "neutral",
        },
        "NeutralAir": {
            "param_string": ["attack_air_n"],
            "raw_animations": ["c05attackairn"],
            "advantage_state": "neutral",
        },
        "ForwardAir": {
            "param_string": ["attack_air_f"],
            "raw_animations": ["c05attackairf"],
            "advantage_state": "neutral",
        },
        "BackAir": {
            "param_string": ["attack_air_b"],
            "raw_animations": ["c05attackairb"],
            "advantage_state": "neutral",
        },
        "DownAir": {
            "param_string": ["attack_air_lw"],
            "raw_animations": ["c05attackairlw"],
            "advantage_state": "neutral",
        },
        "UpAir": {
            "param_string": ["attack_air_hi"],
            "raw_animations": ["c05attackairhi"],
            "advantage_state": "neutral",
        },
        "ZAir": {
            "param_string": ["air_catch"],
            "raw_animations": [],
            "advantage_state": "neutral",
        },
        "Grab": {
            "param_string": ["catch"],
            "raw_animations": ["e00catch", "e00catchdash", "e00catchpull"],
            "advantage_state": "neutral",
        },
        "GrabRelease": {
            "param_string": ["grabrelease"],
            "raw_animations": ["e00catchcut"],
            "advantage_state": "neutral",
        },
        "Parry": {
            "param_string": ["just_shield_off"],
            "raw_animations": [],
            "advantage_state": "neutral",
        },
        "Pummel": {
            "param_string": ["pummel"],
            "raw_animations": ["e00catchattack"],
            "advantage_state": "neutral",
        },
        "ForwardThrow": {
            "param_string": ["throw_f", "throw_f_f"],
            "raw_animations": ["e01throwf"],
            "advantage_state": "neutral",
        },
        "BackThrow": {
            "param_string": ["throw_b", "throw_f_b"],
            "raw_animations": ["e01throwb"],
            "advantage_state": "neutral",
        },
        "DownThrow": {
            "param_string": ["throw_lw", "throw_f_lw"],
            "raw_animations": ["e01throwlw"],
            "advantage_state": "neutral",
        },
        "UpThrow": {
            "param_string": ["throw_hi", "throw_f_hi"],
            "raw_animations": ["e01throwhi"],
            "advantage_state": "neutral",
        },
        "Jump": {
            "param_string": ["jump"],
            "raw_animations": ["a03jump"],
            "advantage_state": "neutral",
        },
        "ShortHop": {
            "param_string": ["jump_f_mini", "jump_b_mini"],
            "raw_animations": [],
            "advantage_state": "neutral",
        },
        "Fall": {
            "param_string": ["fall"],
            "raw_animations": [
                "a04damagefall",
                "a04fall",
                "a04fallaerial",
                "a04fallaerialb",
                "a04fallaerialf",
                "a04fallb",
                "a04fallf",
                "a04runfalll",
                "a04runfallr",
                "a04walkfalll",
                "a04walkfallr",
                "a05landingheavy",
                "a05landinglight",
                "a06stepfall",
            ],
            "advantage_state": "neutral",
        },
        "SpecialFall": {
            "param_string": ["specialfall"],
            "raw_animations": [
                "a04fallspecial",
                "a04landingfallspecial",
            ],
            "advantage_state": "neutral",
        },
        # "DoubleJump": {
        # "param_string": ['doublejump'],
        # "raw_animations": [], "advantage_state": "neutral"},
        "Shield": {
            "param_string": ["guard_on", "guard_damage"],
            "raw_animations": [
                "b00guard",
                "b00justshieldoff",
                "b01guarddamage",
            ],
            "advantage_state": "neutral",
        },
        "ShieldStun": {
            "param_string": [],
            "raw_animations": [],
            "advantage_state": "neutral",
        },
        "ShieldDrop": {
            "param_string": ["guard_off"],
            "raw_animations": ["b01guardoff"],
            "advantage_state": "neutral",
        },
        "Damaged": {
            "param_string": ["damage", "wall_damage", "thrown"],
            "raw_animations": ["f00damage"],
            "advantage_state": "disadvantage",
        },
        "Wait": {
            "param_string": ["wait"],
            "raw_animations": ["a00wait", "wait"],
            "advantage_state": "neutral",
        },
        "Walk": {
            "param_string": ["walk"],
            "raw_animations": ["a01walk"],
            "advantage_state": "neutral",
        },
        "Squat": {
            "param_string": ["squat"],
            "raw_animations": ["a05squat"],
            "advantage_state": "neutral",
        },
        "Dash": {
            "param_string": ["dash"],
            "raw_animations": ["a02dash"],
            "advantage_state": "neutral",
        },
        "Run": {
            "param_string": ["run"],
            "raw_animations": ["a02run"],
            "advantage_state": "neutral",
        },
        "Turn": {
            "param_string": ["turn"],
            "raw_animations": ["a01turn", "a02turn"],
            "advantage_state": "neutral",
        },
        "PlatformDrop": {
            "param_string": ["pass", "platform_drop"],
            "raw_animations": [],
            "advantage_state": "neutral",
        },
        # "DirectionalAirDodge": {
        #     "param_string": ["directionalairdodge"],
        #     "raw_animations": ["b02escapeair"],
        #     "advantage_state": "disadvantage",
        # },
        "AirDodge": {
            # TODO - figure out how to distinguish between the two'neutralairdodge'!!
            "param_string": ["escape_air"],
            "raw_animations": ["b02escapeair"],
            "advantage_state": "disadvantage",
        },
        "Roll": {
            "param_string": ["escape_b", "escape_f"],
            "raw_animations": [
                "b02escapeb",
                "b02escapef",
                "f03downbacku",
                "f03downforwardu",
            ],
            "advantage_state": "neutral",
        },
        "SpotDodge": {
            "param_string": ["escape"],
            "raw_animations": ["b02escapen"],
            "advantage_state": "neutral",
        },
        # TECH OPTIONS
        "DownWait": {
            "param_string": ["down_wait"],
            "raw_animations": ["f04downwaitd"],
            "advantage_state": "disadvantage",
            "option_group": "tech",
        },
        "MissedTech": {
            "param_string": ["down_bound"],
            "raw_animations": ["f04downwaitd"],
            "advantage_state": "disadvantage",
            "option_group": "tech",
        },
        "TechInPlace": {
            "param_string": ["passive"],
            "raw_animations": ["f05passive"],
            "advantage_state": "disadvantage",
            "option_group": "tech",
        },
        "TechRoll": {
            "param_string": ["tech_roll"],
            "raw_animations": ["f05passivestandb", "f05passivestandf"],
            "advantage_state": "disadvantage",
            "option_group": "tech",
        },
        "NormalGetUp": {
            "param_string": ["normalgetup"],
            "raw_animations": ["f04downstandd"],
            "advantage_state": "disadvantage",
            "option_group": "tech",
        },
        "GetUpAttack": {
            "param_string": ["slip_attack", "down_attack"],
            "raw_animations": ["f04downstandd"],
            "advantage_state": "disadvantage",
            "option_group": "tech",
        },
        "Taunt": {
            "param_string": ["appeal"],
            "raw_animations": [""],
            "advantage_state": "neutral",
        },
        # LEDGE OPTIONS
        "LedgeHang": {
            "param_string": ["cliff_wait"],
            "raw_animations": ["g01cliffwait"],
            "advantage_state": "disadvantage",
            "option_group": "ledge",
        },
        "LedgeAttack": {
            "param_string": ["cliff_attack"],
            "raw_animations": ["g02cliffattack"],
            "advantage_state": "disadvantage",
            "option_group": "ledge",
        },
        "LedgeNormalGetUp": {
            "param_string": ["cliff_climb"],
            "raw_animations": ["g02cliffclimb"],
            "advantage_state": "disadvantage",
            "option_group": "ledge",
        },
        "LedgeRoll": {
            "param_string": ["cliff_escape"],
            "raw_animations": ["g02cliffescape"],
            "advantage_state": "disadvantage",
            "option_group": "ledge",
        },
        "LedgeJump": {
            "param_string": ["cliff_jump"],
            "raw_animations": ["g02cliffjump"],
            "advantage_state": "disadvantage",
            "option_group": "ledge",
        },
        "LedgeGrab": {
            "param_string": ["cliff_catch"],
            "raw_animations": [],
            "advantage_state": "disadvantage",
            "option_group": "ledge",
        },
        "ItemPickup": {
            "param_string": ["item_light_get"],
            "raw_animations": [],
            "advantage_state": "neutral",
        },
        "ItemThrow": {
            "param_string": ["item_light_throw"],
            "raw_animations": [],
            "advantage_state": "neutral",
        },
        # "Grabbed": {
        # "param_string": ['grabbed'],
        # "raw_animations": [], "advantage_state": "disadvantage"},
        # "GrabItem": {
        # "param_string": ['grabitem'],
        # "raw_animations": [], "advantage_state": "neutral"},
        # "ThrowItem": {
        # "param_string": ['throwitem'],
        # "raw_animations": [], "advantage_state": "neutral"},
        # "Crawl": {
        # "param_string": ['crawl'],
        # "raw_animations": [], "advantage_state": "neutral"},
        "Slip": {
            "param_string": ["slip"],
            "raw_animations": [],
            "advantage_state": "disadvantage",
        },
        "Landing": {
            "param_string": ["landing"],
            "raw_animations": [
                "a04landingfallspecial",
                "a05landing",
                "a05landing",
                "c05landing",
            ],
            "advantage_state": "neutral",
        },
        "Undefined": {
            "param_string": ["undefined"],
            "raw_animations": [],
            "advantage_state": "neutral",
        },
        "Grabbed": {
            "param_string": ["caught"],
            "raw_animations": ["e00catch", "e00catchdash", "e00catchpull"],
            "advantage_state": "neutral",
        },
    }
}

FIGHTER_ENUM_TO_NAME = {
    0: "Mario",
    1: "Donkey Kong",
    2: "Link",
    3: "Samus",
    4: "Dark Samus",
    5: "Yoshi",
    6: "Kirby",
    7: "Fox",
    8: "Pikachu",
    9: "Luigi",
    10: "Ness",
    11: "Captain Falcon",
    12: "Jigglypuff",
    13: "Peach",
    14: "Daisy",
    15: "Bowser",
    16: "Ice Climbers",
    17: "Sheik",
    18: "Zelda",
    19: "Dr. Mario",
    20: "Falco",
    21: "Marth",
    22: "Lucina",
    23: "Young Link",
    24: "Ganondorf",
    25: "Mewtwo",
    26: "Roy",
    27: "Chrom",
    28: "Game & Watch",
    29: "Meta Knight",
    30: "Pit",
    31: "Dark Pit",
    32: "Zero Suit Samus",
    33: "Wario",
    34: "Snake",
    35: "Ike",
    36: "Pokemon Trainer - Squirtle",
    37: "Pokemon Trainer - Ivysaur",
    38: "Pokemon Trainer - Charizard",
    39: "Diddy Kong",
    40: "Lucas",
    41: "Sonic",
    42: "King Dedede",
    43: "Olimar",
    44: "Lucario",  #
    45: "R.O.B.",
    46: "Toon Link",
    47: "Wolf",  #
    48: "Villager",
    49: "Mega Man",
    50: "Wii-Fit Trainer",
    51: "Rosalina & Luma",
    52: "Little Mac",
    53: "Greninja",
    54: "Palutena",
    55: "Pac-Man",
    56: "Robin",
    57: "Shulk",
    58: "Bowser Jr.",
    59: "Duck Hunt",
    60: "Ryu",
    61: "Ken",
    62: "Cloud",
    63: "Corrin",
    64: "Bayonetta",
    65: "Inkling",
    66: "Ridley",
    67: "Simon",
    68: "Richter",
    69: "King K. Rool",
    70: "Isabelle",
    71: "Incineroar",
    # These are probably the miis.
    72: "??",  # ? MASTER
    73: "??",  # ? TANTAN
    74: "??",  # ? PICKEL
    75: "??",  # ? EDGE
    76: "??",  # ? MIIFIGHTER
    77: "??",  # ? MIISWORDSMAN
    78: "??",  # ? MIIGUNNER
    79: "??",  # ? SAMUSD
    80: "??",  # ? DAISY
    81: "Piranha Plant",
    82: "Joker",
    83: "Hero",
    84: "Banjo & Kazooie",
    85: "Terry",
    86: "Byleth",
    87: "Min Min",
    88: "Steve",
    89: "Sephiroth",
    # These may need to be swapped.
    90: "Pyra",
    91: "Mythra",
    92: "Kazuya",
    93: "Sora",
}


FIGHTER_NAME_TO_ENUM = {v: k for k, v in FIGHTER_ENUM_TO_NAME.items()}

STAGE_ENUM_TO_DATA = {
    0: {
        "name": "BATTLEFIELD",
        "fov": 50,
    },
    3: {
        "name": "FINAL_DESTINATION",
        "fov": 50,
    },
    44: {
        "name": "YOSHI_ISLAND",
        "fov": 50,
    },
    51: {
        "name": "FOUNTAIN_OF_DREAMS",
        "fov": 50,
    },
    86: {
        "name": "YOSHI_ISLAND_OMEGA",
        "fov": 50,
    },
    89: {
        "name": "HOLLOW_BASTION",
        "fov": 50,
    },
    95: {
        "name": "TOWN_AND_CITY",
        "fov": 30,
    },
    107: {
        "name": "POKEMON_STADIUM_2",
        "fov": 50,
    },
    118: {
        "name": "NEW_PORK_CITY",
        "fov": 50,
    },
    242: {
        "name": "KALOS",
        "fov": 50,
    },
    257: {
        "name": "SMASHVILLE",
        "fov": 50,
    },
    268: {
        "name": "PILOT_WINGS",
        "fov": 50,
    },
    293: {
        "name": "UMBRA_CLOCK_TOWER",
        "fov": 50,
    },
    295: {
        "name": "UMBRA_CLOCK_TOWER",
        "fov": 50,
    },
    330: {
        "name": "MEMENTOS",
        "fov": 50,
    },
    347: {
        "name": "SMALL_BATTLEFIELD",
        "fov": 50,
    },
    351: {
        "name": "NORTHERN_CAVE",
        "fov": 50,
    },
    361: {
        "name": "HOLLOW_BASTION",
        "fov": 50,
    },
}

# Stores the data in params_labels which maps hex values of actions to the
# string representation of the action.
HEX_TO_ACTION = {}
with open(PARAMS_LABELS) as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        HEX_TO_ACTION[row[0]] = row[1]

ANIM_FILE_TO_ANIMATION = {}
for fighter in ONTOLOGY:
    for anim in ONTOLOGY[fighter]:
        for animation_file in ONTOLOGY[fighter][anim]["raw_animations"]:
            ANIM_FILE_TO_ANIMATION[animation_file] = anim

PARAM_STRING_TO_ANIMATION = {}
for fighter in ONTOLOGY:
    for anim in ONTOLOGY[fighter]:
        for animation_file in ONTOLOGY[fighter][anim]["param_string"]:
            PARAM_STRING_TO_ANIMATION[animation_file] = anim

MOVE_TO_CLASS_ID = {}
MOVE_TO_ADVANTAGE_STATE = {}
class_id = 0
for fighter in ONTOLOGY:
    for move in ONTOLOGY[fighter]:
        if move not in MOVE_TO_CLASS_ID:
            MOVE_TO_CLASS_ID[move] = class_id
            MOVE_TO_ADVANTAGE_STATE[move] = ONTOLOGY[fighter][move]["advantage_state"]
            class_id += 1

# Classes are 1-indexed to match AVA format.
ONE_INDEXED_MOVE_TO_CLASS_ID = {}
class_id = 1
for fighter in ONTOLOGY:
    for move in ONTOLOGY[fighter]:
        if move not in ONE_INDEXED_MOVE_TO_CLASS_ID:
            ONE_INDEXED_MOVE_TO_CLASS_ID[move] = class_id
            class_id += 1


TRAINED_ACTIONS_2_17 = [
    "Jab",
    "DashAttack",
    "ForwardTilt",
    "DownTilt",
    "UpTilt",
    "ForwardSmash",
    "DownSmash",
    "UpSmash",
    "NeutralSpecial",
    "ForwardSpecial",
    "DownSpecial",
    "UpSpecial",
    "NeutralAir",
    "ForwardAir",
    "BackAir",
    "DownAir",
    "UpAir",
    "Grab",
    "GrabRelease",
    "Pummel",
    "ForwardThrow",
    "BackThrow",
    "DownThrow",
    "UpThrow",
    "Jump",
    "Fall",
    "SpecialFall",
    "Shield",
    "Wait",
    "Walk",
    "Squat",
    "Dash",
    "Run",
    "Turn",
    "AirDodge",
    "Roll",
    "SpotDodge",
    "DownWait",
    "TechInPlace",
    "TechRoll",
    "NormalGetUp",
    "LedgeHang",
    "LedgeAttack",
    "LedgeNormalGetUp",
    "LedgeRoll",
    "LedgeJump",
]

STATUS_ENUM_TO_STRING = {
    0: "FIGHTER_STATUS_KIND_WAIT",
    1: "FIGHTER_STATUS_KIND_WALK",
    3: "FIGHTER_STATUS_KIND_DASH",
    4: "FIGHTER_STATUS_KIND_RUN",
    5: "FIGHTER_STATUS_KIND_RUN_BRAKE",
    6: "FIGHTER_STATUS_KIND_TURN",
    7: "FIGHTER_STATUS_KIND_TURN_DASH",
    8: "FIGHTER_STATUS_KIND_TURN_RUN",
    10: "FIGHTER_STATUS_KIND_JUMP_SQUAT",
    11: "FIGHTER_STATUS_KIND_JUMP",
    12: "FIGHTER_STATUS_KIND_JUMP_AERIAL",
    14: "FIGHTER_STATUS_KIND_FALL",
    15: "FIGHTER_STATUS_KIND_FALL_AERIAL",
    16: "FIGHTER_STATUS_KIND_FALL_SPECIAL",
    17: "FIGHTER_STATUS_KIND_SQUAT",
    18: "FIGHTER_STATUS_KIND_SQUAT_WAIT",
    21: "FIGHTER_STATUS_KIND_SQUAT_RV",
    22: "FIGHTER_STATUS_KIND_LANDING",
    23: "FIGHTER_STATUS_KIND_LANDING_LIGHT",
    24: "FIGHTER_STATUS_KIND_LANDING_ATTACK_AIR",
    25: "FIGHTER_STATUS_KIND_LANDING_FALL_SPECIAL",
    27: "FIGHTER_STATUS_KIND_GUARD_ON",
    28: "FIGHTER_STATUS_KIND_GUARD",
    29: "FIGHTER_STATUS_KIND_GUARD_OFF",
    30: "FIGHTER_STATUS_KIND_GUARD_DAMAGE",
    31: "FIGHTER_STATUS_KIND_ESCAPE",
    32: "FIGHTER_STATUS_KIND_ESCAPE_F",
    33: "FIGHTER_STATUS_KIND_ESCAPE_B",
    34: "FIGHTER_STATUS_KIND_ESCAPE_AIR",
    39: "FIGHTER_STATUS_KIND_ATTACK",
    41: "FIGHTER_STATUS_KIND_ATTACK_DASH",
    42: "FIGHTER_STATUS_KIND_ATTACK_S3",
    43: "FIGHTER_STATUS_KIND_ATTACK_HI3",
    44: "FIGHTER_STATUS_KIND_ATTACK_LW3",
    45: "FIGHTER_STATUS_KIND_ATTACK_S4_START",
    47: "FIGHTER_STATUS_KIND_ATTACK_S4",
    48: "FIGHTER_STATUS_KIND_ATTACK_LW4_START",
    50: "FIGHTER_STATUS_KIND_ATTACK_LW4",
    51: "FIGHTER_STATUS_KIND_ATTACK_HI4_START",
    52: "FIGHTER_STATUS_KIND_ATTACK_HI4_HOLD",
    53: "FIGHTER_STATUS_KIND_ATTACK_HI4",
    54: "FIGHTER_STATUS_KIND_ATTACK_AIR",
    55: "FIGHTER_STATUS_KIND_CATCH",
    56: "FIGHTER_STATUS_KIND_CATCH_PULL",
    57: "FIGHTER_STATUS_KIND_CATCH_DASH",
    58: "FIGHTER_STATUS_KIND_CATCH_DASH_PULL",
    61: "FIGHTER_STATUS_KIND_CATCH_ATTACK",
    64: "FIGHTER_STATUS_KIND_THROW",
    65: "FIGHTER_STATUS_KIND_CAPTURE_PULLED",
    67: "FIGHTER_STATUS_KIND_CAPTURE_DAMAGE",
    70: "FIGHTER_STATUS_KIND_THROWN",
    71: "FIGHTER_STATUS_KIND_DAMAGE",
    72: "FIGHTER_STATUS_KIND_DAMAGE_AIR",
    73: "FIGHTER_STATUS_KIND_DAMAGE_FLY",
    74: "FIGHTER_STATUS_KIND_DAMAGE_FLY_ROLL",
    76: "FIGHTER_STATUS_KIND_DAMAGE_FLY_REFLECT_LR",
    78: "FIGHTER_STATUS_KIND_DAMAGE_FLY_REFLECT_D",
    79: "FIGHTER_STATUS_KIND_DAMAGE_FALL",
    103: "FIGHTER_STATUS_KIND_PASSIVE",
    104: "FIGHTER_STATUS_KIND_PASSIVE_FB",
    105: "FIGHTER_STATUS_KIND_PASSIVE_WALL",
    116: "FIGHTER_STATUS_KIND_PASS",
    117: "FIGHTER_STATUS_KIND_CLIFF_CATCH_MOVE",
    118: "FIGHTER_STATUS_KIND_CLIFF_CATCH",
    119: "FIGHTER_STATUS_KIND_CLIFF_WAIT",
    120: "FIGHTER_STATUS_KIND_CLIFF_ATTACK",
    121: "FIGHTER_STATUS_KIND_CLIFF_CLIMB",
    122: "FIGHTER_STATUS_KIND_CLIFF_ESCAPE",
    123: "FIGHTER_STATUS_KIND_CLIFF_JUMP1",
    124: "FIGHTER_STATUS_KIND_CLIFF_JUMP2",
    134: "FIGHTER_STATUS_KIND_SLIP",
    135: "FIGHTER_STATUS_KIND_SLIP_DAMAGE",
    136: "FIGHTER_STATUS_KIND_SLIP_WAIT",
    137: "FIGHTER_STATUS_KIND_SLIP_STAND",
    138: "FIGHTER_STATUS_KIND_SLIP_STAND_ATTACK",
    141: "FIGHTER_STATUS_KIND_ITEM_LIGHT_PICKUP",
    143: "FIGHTER_STATUS_KIND_ITEM_THROW",
    144: "FIGHTER_STATUS_KIND_ITEM_THROW_DASH",
    181: "FIGHTER_STATUS_KIND_DEAD",
    182: "FIGHTER_STATUS_KIND_REBIRTH",
    240: "FIGHTER_STATUS_KIND_CLUNG_DIDDY",
    241: "FIGHTER_STATUS_KIND_CLUNG_DAMAGE_DIDDY",
    249: "FIGHTER_STATUS_KIND_AIR_LASSO_REACH",
    250: "FIGHTER_STATUS_KIND_AIR_LASSO_HANG",
    251: "FIGHTER_STATUS_KIND_AIR_LASSO_REWIND",
    436: "FIGHTER_STATUS_KIND_CAPTURE_MASTER_SWORD",
    470: "FIGHTER_STATUS_KIND_STANDBY",
    476: "FIGHTER_STATUS_KIND_COMMON_NUM",
    477: "FIGHTER_STATUS_KIND_SPECIAL_S",
    478: "FIGHTER_STATUS_KIND_SPECIAL_HI",
    479: "FIGHTER_STATUS_KIND_SPECIAL_LW",
}

FIGHTER_SPECIAL_NAME_MAP = {
    "R.O.B.": {
        "Robo Beam": "NeutralSpecial",
        "Arm Rotor": "SideSpecial",
        "Robo Burner": "SideSpecial",
    }
}

FIGHTER_STATUS_ENUM_TO_STRING = {
    "Diddy Kong": {
        481: "FIGHTER_DIDDY_STATUS_KIND_SPECIAL_N_CHARGE",
        482: "FIGHTER_DIDDY_STATUS_KIND_SPECIAL_N_SHOOT",
        483: "FIGHTER_DIDDY_STATUS_KIND_SPECIAL_N_DANGER",
        485: "FIGHTER_DIDDY_STATUS_KIND_SPECIAL_S_JUMP",
        488: "FIGHTER_DIDDY_STATUS_KIND_SPECIAL_S_STICK",
        489: "FIGHTER_DIDDY_STATUS_KIND_SPECIAL_S_STICK_ATTACK",
        490: "FIGHTER_DIDDY_STATUS_KIND_SPECIAL_S_STICK_ATTACK2",
        493: "FIGHTER_DIDDY_STATUS_KIND_SPECIAL_S_FLIP_LANDING",
        494: "FIGHTER_DIDDY_STATUS_KIND_SPECIAL_S_FLIP_FALL",
        496: "FIGHTER_DIDDY_STATUS_KIND_SPECIAL_HI_CHARGE",
        498: "FIGHTER_DIDDY_STATUS_KIND_SPECIAL_HI_UPPER",
        501: "FIGHTER_DIDDY_STATUS_KIND_SPECIAL_HI_FALL_ROLL",
    },
    "byleth": {
        485: "FIGHTER_MASTER_STATUS_KIND_SPECIAL_N_CANCEL",
        487: "FIGHTER_MASTER_STATUS_KIND_SPECIAL_S_FRONT",
        488: "FIGHTER_MASTER_STATUS_KIND_SPECIAL_S_FRONT_DASH",
        489: "FIGHTER_MASTER_STATUS_KIND_SPECIAL_S_LANDING",
        490: "FIGHTER_MASTER_STATUS_KIND_SPECIAL_HI_HIT",
        491: "FIGHTER_MASTER_STATUS_KIND_SPECIAL_HI_OVERTAKE",
        495: "FIGHTER_MASTER_STATUS_KIND_SPECIAL_LW_LANDING_1",
        497: "FIGHTER_MASTER_STATUS_KIND_SPECIAL_LW_HIT",
    },
}
