ANCHORS = ["World", "Head", "Torso", "Arm"]
TASKS = ["Key", "Visual", "Controls"]
CONDITIONS = ["Stationary", "Semi-Stationary", "Moving"]

# Column name normalization helpers
COLUMN_ALIASES = {
    "participant id": "participant_id",
    "age": "age",
    "gender": "gender",
    "trial condition": "condition",
    "experience with xr devices": "xr_experience",
    "experience with videogames": "game_experience",
    "start time": "start_time",
    "completion time": "completion_time",
}

# Fallbacks for anchor typos found in the CSV
ANCHOR_FIX = {
    "worl": "World",
    "world": "World",
    "head": "Head",
    "tors": "Torso",
    "torso": "Torso",
    "arm": "Arm",
}

# Comment fields that hold free text
GLOBAL_COMMENT_KEYS = [
    "any further thoughts on preferences? did you change your mind since the planning phase?",
    "what could be improved for the anchoring / positioning to work better in this scenario?",
    "instead of moving and anchoring the interface, imagine you could talk to it. how would you tell the interface to behave?",
    "other comments",
]

# Theme seed keywords (lowercased)
THEME_KEYWORDS = {
    "visibility": ["see", "visible", "fov", "field of view", "look"],
    "obstruction": ["obstruct", "clutter", "block", "annoy", "tiring"],
    "stability": ["stable", "fixed", "follow", "move", "tracking"],
    "interaction": ["click", "button", "interact", "grab", "press"],
    "convenience": ["easy", "convenient", "handy", "close", "reach"],
}

