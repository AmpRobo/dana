"""Spoken-style commands with their tool calls for examples/tools/desktop.json, from templates.

No Gemini needed: every tool has hand-written phrasings and value pools, sampled and
deduplicated, plus about a tenth "do two things at once" commands. The output is the
--data JSONL that scripts/voice_finetune.py takes.

    python scripts/desktop_commands.py --out data/voice/desktop_commands.jsonl
    python scripts/voice_finetune.py examples/tools/desktop.json --data data/voice/desktop_commands.jsonl --tts kokoro
"""
import argparse
import json
import os
import random
import re
import sys

NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
             10: "ten", 15: "fifteen", 20: "twenty", 25: "twenty five", 30: "thirty", 40: "forty", 45: "forty five",
             50: "fifty", 60: "sixty", 70: "seventy", 75: "seventy five", 80: "eighty", 90: "ninety",
             100: "one hundred", 0: "zero", 35: "thirty five", 65: "sixty five", 85: "eighty five", 12: "twelve"}

POLITE = ["Please ", "Can you ", "Could you ", "Hey, ", "Hey, can you ", "Go ahead and ", "Would you ",
          "I want to ", "I'd like to ", "Quickly ", "Okay, "]
TAIL = [" please", " for me", " now", " right now", " real quick", " thanks"]


def number(rng, n, words=0.25):
    return NUM_WORDS[n] if n in NUM_WORDS and rng.random() < words else str(n)


def finish(rng, template, raw=False):
    """A template as a recognizer would print it: optional polite start and tail, capital, stop."""
    text = template
    question = False
    if not raw and rng.random() < 0.45:
        start = rng.choice(POLITE)
        if start.startswith(("I want", "I'd like")) and not re.match(r"[a-z]", text):
            start = ""
        question = start.rstrip(", ").endswith(("you", "Can you", "Could you", "Would you"))
        text = start + text[:1].lower() + text[1:] if start else text
    if not raw and rng.random() < 0.2:
        text += rng.choice(TAIL)
    text = text[:1].upper() + text[1:]
    if text[-1] not in ".?!":
        text += "?" if question else rng.choice([".", ".", ".", ""])
    return text


def pick(rng, *options):
    return rng.choice(options)


APPS = ["Firefox", "Chrome", "Terminal", "Files", "Spotify", "Slack", "VS Code", "Thunderbird", "Calculator",
        "Discord", "Zoom", "Steam", "GIMP", "Blender", "Obsidian", "Telegram", "VLC", "Text Editor", "Calendar",
        "LibreOffice Writer", "Inkscape", "Audacity", "OBS", "Teams", "Signal", "Krita", "PyCharm", "Postman",
        "Notion", "Evolution", "Rhythmbox", "Kdenlive", "Shotwell", "Brave", "Outlook", "Excel", "Word",
        "System Monitor", "Software Center", "Clock", "Weather", "Maps", "Camera", "Photos", "Mail",
        "Safari", "Edge", "Opera", "Vivaldi", "Skype", "WhatsApp", "Element", "Lutris", "Godot", "Unity",
        "Android Studio", "IntelliJ", "Sublime Text", "Vim", "Emacs", "Konsole", "Dolphin", "Okular",
        "Evince", "Transmission", "qBittorrent", "KeePassXC", "Bitwarden", "Anki", "Zotero", "Calibre",
        "Darktable", "Shotcut", "Handbrake", "Wireshark", "VirtualBox", "Docker Desktop", "Figma",
        "Minecraft", "Tidal", "Pocket Casts", "Screenshot tool", "Character Map", "Disks"]
THE_APPS = {"Terminal", "Files", "Calculator", "Text Editor", "Calendar", "Clock", "Weather", "Maps", "Camera",
            "Photos", "Mail", "System Monitor", "Software Center", "Screenshot tool", "Character Map", "Disks"}


def app_say(rng, app):
    """How people name an app: "the terminal", "Terminal", "my Spotify"."""
    if app in THE_APPS and rng.random() < 0.5:
        return pick(rng, f"the {app.lower()}", f"the {app} app", f"the {app.lower()} app")
    return pick(rng, app, app, app, f"my {app}", f"the {app} app")
OUTPUTS = ["External Speakers", "HDMI", "Headphones", "Bluetooth Headphones", "Laptop Speakers", "USB Headset",
           "Monitor Speakers", "Soundbar", "Built-in Speakers", "TV", "DisplayPort", "Desk Speakers"]
OUTPUT_SAY = {"HDMI": ["HDMI", "the HDMI"], "TV": ["the TV"], "DisplayPort": ["DisplayPort"]}
INPUTS = ["External Mic", "Webcam", "USB Mic", "Headset Mic", "Built-in Microphone", "Laptop Mic", "Blue Yeti",
          "Studio Mic", "Webcam Microphone"]
SERVERS = ["Japan", "Tokyo", "US East", "Germany", "London", "Singapore", "Netherlands", "Switzerland", "Canada",
           "Frankfurt", "New York", "Los Angeles", "Sweden", "Australia", "Paris", "Hong Kong", "Seoul", "Chicago"]
PRINTERS = ["office printer", "HP LaserJet", "Brother", "Canon", "Epson", "upstairs printer", "color printer",
            "downstairs printer", "home printer", "printer in the lab"]
DEVICES_BT = ["AirPods", "AirPods Pro", "Keyboard", "Mouse", "Magic Mouse", "Galaxy Buds", "Bose headphones",
              "JBL speaker", "Xbox controller", "Pixel Buds", "Sony headphones", "car stereo", "trackpad",
              "PlayStation controller", "Beats", "smartwatch"]
FOLDERS = ["Downloads", "Documents", "Pictures", "Music", "Videos", "Desktop", "Home", "Projects",
           "Screenshots", "Public", "Templates", "Photos", "Work", "Code"]
FILES = ["quarterly report", "meeting notes", "resume", "budget spreadsheet", "report.pdf", "budget.xlsx",
         "notes.txt", "presentation", "invoice", "tax return", "thesis draft", "shopping list", "lecture slides",
         "contract", "project plan", "travel itinerary", "cover letter", "README", "recipe", "photo album"]
SEARCHES = ["tax documents", "vacation photos", "invoice", "2025 budget", "screenshots from last week", "resume",
            "birthday pictures", "lease agreement", "the slides from Monday", "receipts", "wedding photos",
            "anything called notes", "insurance papers", "PDFs about Python", "the logo", "bank statements",
            "contract", "the video from the party", "spreadsheets", "passport scan"]
SITES = {"GitHub": "github.com", "YouTube": "youtube.com", "Wikipedia": "wikipedia.org", "Gmail": "gmail.com",
         "Reddit": "reddit.com", "Amazon": "amazon.com", "Netflix": "netflix.com",
         "Stack Overflow": "stackoverflow.com", "Twitter": "twitter.com", "Hacker News": "news.ycombinator.com",
         "the BBC": "bbc.com", "Google Maps": "maps.google.com", "LinkedIn": "linkedin.com",
         "Google Drive": "drive.google.com", "the New York Times": "nytimes.com", "eBay": "ebay.com",
         "Facebook": "facebook.com", "Instagram": "instagram.com", "Twitch": "twitch.tv", "IMDb": "imdb.com"}
QUERIES = ["weather in Tokyo", "how to boil an egg", "best pizza near me", "python list comprehension",
           "train times to Osaka", "who won the game last night", "how tall is Mount Fuji", "cheap flights to Seoul",
           "symptoms of the flu", "how to fix a flat tire", "the capital of Australia", "Linux disk usage command",
           "banana bread recipe", "today's exchange rate for yen", "movie times tonight", "how to tie a tie",
           "the meaning of serendipity", "new laptops", "jazz concerts this weekend", "how many ounces in a cup",
           "when does the store close", "rust vs go", "how to center a div", "the latest news"]
TEXTS = ["see you tomorrow", "thanks for the update", "I'll be there in five minutes", "hello, how are you",
         "sounds good to me", "let's meet at noon", "on my way", "happy birthday", "I'm running late",
         "can we talk later", "the meeting is cancelled", "good morning everyone", "please review the document",
         "call me when you get this", "dinner is ready", "best regards", "no problem", "talk soon",
         "I agree with that", "the report is attached"]
CLIPS = ["meeting at 3 pm", "221B Baker Street", "order number 4471", "my phone number 555 0199",
         "the wifi password is sunflower", "see you on Friday", "project deadline is Monday", "room 204",
         "flight JL 61", "pick up milk", "tracking number 9200", "the code is 7731", "Main Street 42",
         "happy new year", "thank you so much"]
MODS = [("control", "ctrl"), ("ctrl", "ctrl"), ("shift", "shift"), ("alt", "alt"), ("super", "super"),
        ("command", "cmd"), ("windows", "super"), ("option", "alt"), ("cmd", "cmd")]
BASE_KEYS = [(c.upper(), c) for c in "abcdefghijklmnopqrstuvwxyz"] + [(str(d), str(d)) for d in range(10)] + \
    [(f"F{n}", f"f{n}") for n in range(1, 13)] + \
    [("escape", "escape"), ("enter", "enter"), ("tab", "tab"), ("space", "space"), ("delete", "delete"),
     ("backspace", "backspace"), ("home", "home"), ("end", "end"), ("page up", "pageup"), ("page down", "pagedown"),
     ("up", "up"), ("down", "down"), ("left", "left"), ("right", "right"), ("plus", "plus"), ("minus", "minus"),
     ("insert", "insert"), ("print screen", "printscreen")]


def random_keys(rng):
    """A shortcut said aloud and as press_keys wants it: ("control shift escape", "ctrl+shift+escape")."""
    n = rng.choice([1, 1, 2, 2, 2])
    mods, said, keys = rng.sample(MODS[:5] if rng.random() < 0.6 else MODS, n), [], []
    seen = set()
    for spoken, key in mods:
        if key not in seen:
            seen.add(key)
            said.append(spoken)
            keys.append(key)
    spoken, key = rng.choice(BASE_KEYS)
    joiner = pick(rng, " ", " ", " plus ")
    return joiner.join(said + [spoken]), "+".join(keys + [key])


KEYS = [("control C", "ctrl+c"), ("control V", "ctrl+v"), ("control Z", "ctrl+z"), ("alt tab", "alt+tab"),
        ("enter", "enter"), ("escape", "escape"), ("control shift T", "ctrl+shift+t"), ("control S", "ctrl+s"),
        ("F5", "f5"), ("the space bar", "space"), ("tab", "tab"), ("control A", "ctrl+a"), ("alt F4", "alt+f4"),
        ("control W", "ctrl+w"), ("control T", "ctrl+t"), ("backspace", "backspace"), ("delete", "delete"),
        ("control F", "ctrl+f"), ("super", "super"), ("control alt T", "ctrl+alt+t"), ("page down", "pagedown"),
        ("control P", "ctrl+p"), ("shift tab", "shift+tab"), ("F11", "f11"), ("control shift N", "ctrl+shift+n")]
KEY_INTENTS = [("Undo that", "ctrl+z"), ("Redo that", "ctrl+shift+z"), ("Select all", "ctrl+a"),
               ("Save this", "ctrl+s"), ("Refresh the page", "f5"), ("Hit enter", "enter"),
               ("Paste it", "ctrl+v"), ("Reopen the tab I just closed", "ctrl+shift+t"),
               ("Find on page", "ctrl+f"), ("Open a new tab", "ctrl+t")]
PAGES = ["display", "sound", "network", "bluetooth", "keyboard", "mouse", "privacy", "power", "notifications",
         "wi-fi", "users", "printers", "appearance", "date and time", "updates", "accessibility"]



# ---------------------------------------------------------------- --wide: many more values
# With small closed lists Dana learns to pick a value it has seen ("Brazil" -> "Brave");
# with --wide most values come from long lists or are made up, so it learns to copy what was said.
WIDE = False
SYLLABLES = ["ka", "lo", "mi", "zen", "tro", "vi", "sa", "no", "ri", "bel", "dor", "fa", "lu", "mar", "pe", "qui",
             "ra", "sol", "ta", "ven", "xo", "yu", "zor", "an", "el", "is", "om", "ul", "bri", "cor", "dex", "flo"]


def made_up(rng, parts=(2, 3)):
    return "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(*parts))).capitalize()


WIDE_SERVERS = """Brazil, Argentina, Chile, Mexico, Peru, Colombia, Spain, Portugal, Italy, France, Belgium, Austria,
Poland, Czechia, Hungary, Romania, Greece, Turkey, Israel, Egypt, Kenya, Nigeria, South Africa, Morocco, India,
Thailand, Vietnam, Malaysia, Indonesia, Philippines, Taiwan, South Korea, China, New Zealand, Ireland, Scotland,
Iceland, Norway, Finland, Denmark, Estonia, Ukraine, Dubai, Qatar, Saudi Arabia, Pakistan, Bangladesh, Sri Lanka,
Nepal, Mongolia, Kazakhstan, Georgia, Armenia, Serbia, Croatia, Slovenia, Bulgaria, Luxembourg, Monaco, Malta,
Cyprus, Lisbon, Madrid, Barcelona, Rome, Milan, Vienna, Prague, Warsaw, Budapest, Athens, Istanbul, Cairo, Nairobi,
Lagos, Cape Town, Mumbai, Delhi, Bangalore, Bangkok, Hanoi, Jakarta, Manila, Taipei, Osaka, Sapporo, Fukuoka,
Shanghai, Beijing, Sydney, Melbourne, Auckland, Dublin, Edinburgh, Oslo, Stockholm, Helsinki, Copenhagen, Amsterdam,
Brussels, Zurich, Geneva, Munich, Berlin, Hamburg, Miami, Dallas, Seattle, Denver, Atlanta, Toronto, Vancouver,
Montreal, Sao Paulo, Buenos Aires, Santiago, Lima, Bogota, Mexico City, US West, US Central, EU West, Asia Pacific,
Oregon, Virginia, Ohio, California, Texas""".replace("\n", " ").split(", ")
BRANDS_AUDIO = ["Sony", "Bose", "JBL", "Sennheiser", "Logitech", "Razer", "HyperX", "SteelSeries", "Audio-Technica",
                "Shure", "Rode", "Focusrite", "Samsung", "LG", "Dell", "Anker", "Marshall", "Jabra", "Corsair",
                "Elgato", "Yamaha", "Creative", "Edifier", "Klipsch", "Sonos", "Beyerdynamic", "AKG", "Behringer"]
OUT_KINDS = ["speakers", "headphones", "headset", "soundbar", "monitor", "TV", "earbuds", "speaker"]
IN_KINDS = ["mic", "microphone", "headset", "webcam", "interface"]
BT_KINDS = ["earbuds", "headphones", "speaker", "keyboard", "mouse", "controller", "watch", "trackpad", "headset"]
PRINTER_BRANDS = ["Ricoh", "Xerox", "Kyocera", "Lexmark", "Samsung", "Dell", "Sharp", "Konica", "Brother", "HP"]
PLACES = ["kitchen", "garage", "basement", "second floor", "lobby", "library", "front office", "studio", "attic",
          "conference room", "reception", "mail room", "den", "classroom", "warehouse"]
FILE_ADJ = ["final", "old", "new", "draft", "signed", "updated", "Q3", "2026", "March", "weekly", "annual", "shared",
            "team", "client", "personal", "scanned", "backup"]
FILE_NOUN = ["budget", "report", "slides", "invoice", "contract", "proposal", "notes", "schedule", "minutes",
             "receipt", "estimate", "roadmap", "photo", "playlist", "essay", "spec", "timesheet", "checklist",
             "brochure", "poster", "manuscript", "itinerary", "statement", "letter"]
FILE_EXT = ["pdf", "docx", "txt", "xlsx", "png", "jpg", "pptx", "csv", "md", "mp4"]
FOLDER_NAMES = ["Downloads", "Documents", "Pictures", "Music", "Videos", "Desktop", "Home", "Projects", "Screenshots",
                "Recordings", "Invoices", "Taxes", "Backups", "School", "Homework", "Archive", "Clients", "Games",
                "Wallpapers", "Podcasts", "Recipes", "Travel", "Receipts", "Drafts", "Exports", "Fonts", "Books",
                "Scans", "Camera Roll", "Dropbox", "OneDrive", "Shared", "Temp", "Notes", "Assets", "Design"]
SEARCH_TOPICS = ["photos from {place}", "{noun} from last {when}", "the {adj} {noun}", "{noun}s about {topic}",
                 "anything with {person} in the name", "{topic} notes", "the {noun} I edited {when2}"]
TOPICS = ["Python", "the garden", "taxes", "the wedding", "marketing", "chemistry", "the move", "Japan", "pricing",
          "the launch", "insurance", "the kids' school", "my car", "hiking", "piano lessons", "the renovation"]
QUERY_TEMPLATES = ["how to {verb} {thing}", "best {thing} {when3}", "what is {topic}", "{topic} news",
                   "price of {thing}", "{thing} near me", "how long to {verb} {thing}", "reviews of {thing}",
                   "history of {topic}", "when was {thing} invented", "{place} opening hours", "is {thing} safe"]
VERBS = ["clean", "fix", "cook", "install", "paint", "repair", "replace", "grow", "store", "freeze", "learn", "plant"]
THINGS = ["a bike chain", "rice", "a laptop battery", "tomatoes", "sourdough", "a ceiling fan", "an air fryer",
          "running shoes", "a mechanical keyboard", "solar panels", "a used car", "wool sweaters", "basil",
          "a dishwasher", "noise cancelling headphones", "an electric toothbrush", "tofu", "a standing desk"]
PEOPLE_W = ["Alex", "Priya", "Tom", "Yuki", "Carlos", "Fatima", "Liam", "Mei", "Olga", "Sam", "Noah", "Aiko"]
SENT_START = ["I'll", "We'll", "Let's", "Please", "Can we", "Don't forget to", "I need to", "We should"]
SENT_VERB = ["meet", "call", "talk", "finish the draft", "leave", "start", "eat", "review it", "head out", "check in"]
SENT_WHEN = ["at 5", "tomorrow", "after lunch", "on Friday", "in ten minutes", "tonight", "next week", "soon",
             "before the meeting", "at noon"]


def wide_device(rng, kinds, brands=BRANDS_AUDIO):
    brand = rng.choice(brands) if rng.random() < 0.75 else made_up(rng)
    return f"{brand} {rng.choice(kinds)}"


def wide_file(rng):
    r = rng.random()
    if r < 0.35:
        return f"{rng.choice(FILE_ADJ)} {rng.choice(FILE_NOUN)}"
    if r < 0.6:
        return f"{rng.choice(FILE_NOUN)}.{rng.choice(FILE_EXT)}"
    if r < 0.8:
        return f"{made_up(rng).lower()} {rng.choice(FILE_NOUN)}"
    return f"{rng.choice(PEOPLE_W)}'s {rng.choice(FILE_NOUN)}"


def wide_search(rng):
    return rng.choice(SEARCH_TOPICS).format(
        place=rng.choice(WIDE_SERVERS), noun=rng.choice(FILE_NOUN), when=pick(rng, "week", "month", "year"),
        adj=rng.choice(FILE_ADJ), topic=rng.choice(TOPICS), person=rng.choice(PEOPLE_W),
        when2=pick(rng, "yesterday", "this morning", "last week"))


def wide_query(rng):
    return rng.choice(QUERY_TEMPLATES).format(
        verb=rng.choice(VERBS), thing=rng.choice(THINGS), topic=rng.choice(TOPICS), place=rng.choice(WIDE_SERVERS),
        when3=pick(rng, "2026", "for beginners", "under 100 dollars", "this year"))


def wide_sentence(rng):
    return f"{rng.choice(SENT_START)} {rng.choice(SENT_VERB)} {rng.choice(SENT_WHEN)}".lower()


def wide_clip(rng):
    r = rng.random()
    if r < 0.3:
        return f"{pick(rng, 'order', 'ticket', 'invoice', 'case', 'booking')} number {rng.randint(100, 99999)}"
    if r < 0.5:
        return f"{rng.randint(1, 999)} {made_up(rng)} {pick(rng, 'Street', 'Avenue', 'Road', 'Lane')}"
    if r < 0.7:
        return f"the {pick(rng, 'code', 'PIN', 'room', 'gate')} is {rng.randint(10, 9999)}"
    return wide_sentence(rng)


def wide_app(rng):
    r = rng.random()
    if r < 0.6:
        return made_up(rng)
    return f"{made_up(rng)} {pick(rng, 'Studio', 'Notes', 'Player', 'Editor', 'Mail', 'Chat', 'Reader', 'Pro')}"


WIDE_EXTRA = {}  # id(pool) -> (share made up or from the long list, function giving one value)


def _wide_setup():
    WIDE_EXTRA.update({
        id(SERVERS): (0.8, lambda rng: rng.choice(WIDE_SERVERS)),
        id(OUTPUTS): (0.6, lambda rng: wide_device(rng, OUT_KINDS)),
        id(INPUTS): (0.6, lambda rng: wide_device(rng, IN_KINDS)),
        id(DEVICES_BT): (0.6, lambda rng: wide_device(rng, BT_KINDS, BRANDS_AUDIO + ["Apple", "Google", "Xbox",
                                                                                  "Nintendo", "Garmin", "Fitbit"])),
        id(PRINTERS): (0.6, lambda rng: pick(rng, f"{rng.choice(PRINTER_BRANDS)} {rng.randint(100, 9900)}",
                                             f"{rng.choice(PLACES)} printer", f"printer in the {rng.choice(PLACES)}",
                                             f"{made_up(rng)} printer")),
        id(APPS): (0.35, wide_app),
        id(FOLDERS): (0.6, lambda rng: rng.choice(FOLDER_NAMES) if rng.random() < 0.7 else made_up(rng)),
        id(FILES): (0.7, wide_file),
        id(SEARCHES): (0.7, wide_search),
        id(QUERIES): (0.7, wide_query),
        id(TEXTS): (0.6, wide_sentence),
        id(CLIPS): (0.7, wide_clip),
    })


def draw(rng, pool):
    share, fn = WIDE_EXTRA.get(id(pool), (0, None))
    return fn(rng) if fn and rng.random() < share else rng.choice(pool)


WIDE_SITES = {"Spotify": "spotify.com", "Duolingo": "duolingo.com", "Etsy": "etsy.com", "Pinterest": "pinterest.com",
              "Medium": "medium.com", "Dropbox": "dropbox.com", "Figma": "figma.com", "Coursera": "coursera.org",
              "Khan Academy": "khanacademy.org", "the Guardian": "theguardian.com", "Bandcamp": "bandcamp.com",
              "Booking": "booking.com", "Airbnb": "airbnb.com", "Zillow": "zillow.com", "Craigslist": "craigslist.org",
              "Quora": "quora.com", "Wolfram Alpha": "wolframalpha.com", "Hugging Face": "huggingface.co",
              "GitLab": "gitlab.com", "Trello": "trello.com", "Canva": "canva.com", "Vimeo": "vimeo.com"}


def draw_site(rng):
    """(name as said, url, what is said): a known site by name, or an address read out."""
    r = rng.random() if WIDE else 1
    if r < 0.35:
        url = f"{made_up(rng).lower()}.{pick(rng, 'com', 'io', 'org', 'net', 'dev', 'app', 'co')}"
        return url, url, url
    if r < 0.6:
        name = rng.choice(list(WIDE_SITES))
    else:
        name = rng.choice(list(SITES))
    url = (WIDE_SITES | SITES)[name]
    return name, url, url if rng.random() < 0.3 else name


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def on_off(rng, thing, enabled, *, extra_on=(), extra_off=()):
    word = "on" if enabled else "off"
    forms = [f"Turn {word} {thing}", f"Turn {thing} {word}", f"Switch {thing} {word}", f"{thing} {word}",
             f"{'Enable' if enabled else 'Disable'} {thing}", f"Can you {'enable' if enabled else 'disable'} {thing}",
             f"Set {thing} {word}"]
    forms += list(extra_on if enabled else extra_off)
    return rng.choice(forms)


def simple(*phrases, raw=()):
    return lambda rng: (finish(rng, rng.choice(phrases)), {}) if (not raw or rng.random() < 0.8) else \
        (finish(rng, rng.choice(raw), raw=True), {})


def g_split(rng):
    side = pick(rng, "left", "right")
    t = pick(rng, f"Snap this window to the {side}", f"Split screen {side}", f"Put this window on the {side} half",
             f"Move the window to the {side} side", f"Snap it {side}", f"Tile this window {side}",
             f"Dock the window to the {side} of the screen", f"Make this window take the {side} half")
    return finish(rng, t), {"position": side}


def g_desktop(rng, move=False):
    choice = pick(rng, "next", "previous", "num", "num")
    if choice == "num":
        n = rng.randint(1, 6)
        value, say = str(n), pick(rng, f"desktop {number(rng, n, 0.3)}", f"workspace {number(rng, n, 0.3)}",
                                  f"virtual desktop {number(rng, n, 0.3)}")
    else:
        value = choice
        say = pick(rng, f"the {choice} desktop", f"the {choice} workspace", f"the {choice} virtual desktop")
    if move:
        t = pick(rng, f"Move this window to {say}", f"Send the window to {say}", f"Put this window on {say}",
                 f"Throw this window over to {say}")
        return finish(rng, t), {"desktop": value}
    t = pick(rng, f"Switch to {say}", f"Go to {say}", f"Take me to {say}", f"Jump to {say}", f"Show {say}")
    if choice != "num" and rng.random() < 0.3:
        t = pick(rng, f"{choice.capitalize()} desktop", f"{choice.capitalize()} workspace")
    return finish(rng, t), {"direction": value}


def g_vpn(rng):
    enabled = rng.random() < 0.6
    if enabled and rng.random() < 0.5:
        server = draw(rng, SERVERS)
        t = pick(rng, f"Connect the VPN to {server}", f"Turn on the VPN with the {server} server",
                 f"Turn on VPN and connect to {server}", f"Use a VPN server in {server}", f"VPN on, {server}",
                 f"Connect to the {server} VPN")
        return finish(rng, t), {"enabled": True, "server": server}
    t = on_off(rng, pick(rng, "the VPN", "VPN", "my VPN"), enabled,
               extra_on=["Connect to the VPN", "Start the VPN"], extra_off=["Disconnect the VPN", "Stop the VPN"])
    return finish(rng, t), {"enabled": enabled}


def g_audio_out(rng):
    dev = draw(rng, OUTPUTS)
    say = rng.choice(OUTPUT_SAY.get(dev, [dev, f"the {dev}", f"my {dev}"]))
    t = pick(rng, f"Switch the audio output to {say}", f"Play sound through {say}", f"Send the audio to {say}",
             f"Use {say} for sound", f"Switch the sound to {say}", f"Output audio to {say}",
             f"Change the speaker to {say}", f"I want sound coming out of {say}")
    return finish(rng, t), {"device_name": dev}


def g_audio_in(rng):
    dev = draw(rng, INPUTS)
    say = pick(rng, dev, f"the {dev}", f"my {dev}")
    t = pick(rng, f"Switch the microphone to {say}", f"Use {say} as the input", f"Change the audio input to {say}",
             f"Record from {say}", f"Set the mic to {say}", f"Switch audio input to {say}")
    return finish(rng, t), {"device_name": dev}


def g_print(rng):
    args, parts = {}, []
    if rng.random() < 0.5:
        n = rng.randint(1, 10)
        args["copies"] = n
        parts.append(f"{number(rng, n, 0.6)} cop{'y' if n == 1 else 'ies'}")
    if rng.random() < 0.5:
        args["printer_name"] = draw(rng, PRINTERS)
    p = args.get("printer_name")
    if parts and p:
        t = pick(rng, f"Print {parts[0]} on the {p}", f"Print {parts[0]} to the {p}", f"Send {parts[0]} to the {p}")
    elif parts:
        t = pick(rng, f"Print {parts[0]}", f"Print {parts[0]} of this", f"I need {parts[0]} printed")
    elif p:
        t = pick(rng, f"Print this on the {p}", f"Send this to the {p}", f"Print the document to the {p}")
    else:
        t = pick(rng, "Print this", "Print the document", "Print this page", "Send this to the printer",
                 "Print it", "Print the current document", "Can I get a printout of this")
    return finish(rng, t), args


def g_screenshot(rng):
    kind = pick(rng, "full", "window", "selection", None)
    t = {"full": ["Take a screenshot of the whole screen", "Screenshot the full screen", "Capture the entire screen",
                  "Grab a full screen screenshot"],
         "window": ["Take a screenshot of this window", "Screenshot just the window", "Capture the current window",
                    "Grab a screenshot of the window"],
         "selection": ["Take a screenshot of a selection", "Let me select an area to screenshot",
                       "Screenshot a region I pick", "Capture part of the screen", "Snip a part of the screen"],
         None: ["Take a screenshot", "Screenshot", "Grab a screenshot", "Capture the screen", "Snap a screenshot"]}[kind]
    return finish(rng, rng.choice(t)), ({"region": kind} if kind else {})


def g_kill(rng):
    app = draw(rng, APPS)
    name = app_say(rng, app) if rng.random() < 0.3 else app
    t = pick(rng, f"Force quit {name}", f"Kill {name}", f"{app} is frozen, kill it", f"Force close {name}",
             f"{app} isn't responding, force quit it", f"End the {app} process", f"Kill the {app} process",
             f"Make {name} quit, it's stuck", f"Close {name}, it's frozen", f"{app} crashed, kill it",
             f"Shut down {name}, it's not responding")
    return finish(rng, t, raw="is" in t.split()[1:3]), {"app_name": app}


def g_open_app(rng):
    app = draw(rng, APPS)
    say = app_say(rng, app)
    t = pick(rng, f"Open {say}", f"Launch {say}", f"Start {say}", f"Fire up {say}", f"Run {say}",
             f"Open up {say}", f"Can you get {say} going", f"I need {say}", f"Start up {say}",
             f"Pull up {say}", f"Get {say} open")
    return finish(rng, t), {"app_name": app}


def g_switch_app(rng):
    app = draw(rng, APPS)
    say = app_say(rng, app)
    t = pick(rng, f"Switch to {say}", f"Go to {say}", f"Bring {say} to the front", f"Show me {say}",
             f"Jump over to {say}", f"Focus {say}", f"Take me back to {say}", f"Bring the {app} window back",
             f"Go back to {say}", f"Switch over to the {app} window", f"Bring up the {app} window I had open")
    return finish(rng, t), {"app_name": app}


def g_monitor(rng):
    d = pick(rng, "left", "right", "next")
    say = {"left": ["the left monitor", "the left screen", "the monitor on the left"],
           "right": ["the right monitor", "the right screen", "the monitor on the right"],
           "next": ["the next monitor", "the other monitor", "the other screen", "the next screen"]}[d]
    t = pick(rng, "Move this window to {}", "Send the window to {}", "Put this on {}", "Throw this window to {}")
    return finish(rng, t.format(rng.choice(say))), {"direction": d}


def g_set_volume(rng):
    n = rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 65, 70, 75, 80, 85, 90, 100, 12, 33, 55])
    s = number(rng, n, 0.35)
    t = pick(rng, f"Set the volume to {s}", f"Volume {s}", f"Set volume to {s} percent", f"Put the volume at {s}",
             f"Change the volume to {s}", f"Make the volume {s}", f"Turn the volume to {s}")
    return finish(rng, t), {"level": n}


def g_adjust_volume(rng):
    up = rng.random() < 0.5
    t = pick(rng, *(["Turn it up", "Volume up", "Louder", "Make it louder", "Turn the volume up",
                     "A bit louder", "Raise the volume", "Turn up the sound", "I can't hear it, turn it up"] if up
                    else ["Turn it down", "Volume down", "Quieter", "Make it quieter", "Turn the volume down",
                          "A bit quieter", "Lower the volume", "Turn down the sound", "It's too loud, turn it down"]))
    return finish(rng, t), {"direction": "up" if up else "down"}


def g_mute(rng, mic=False):
    muted = rng.random() < 0.6
    if mic:
        t = pick(rng, *(["Mute my microphone", "Mute the mic", "Mic off", "Turn off my microphone",
                         "Mute myself", "Stop the mic from picking me up"] if muted else
                        ["Unmute my microphone", "Unmute the mic", "Mic on", "Turn my microphone back on",
                         "Unmute me", "Turn the mic on"]))
    else:
        t = pick(rng, *(["Mute the sound", "Mute", "Mute the audio", "Silence the computer", "Mute the speakers",
                         "Turn the sound off", "Kill the sound"] if muted else
                        ["Unmute", "Unmute the sound", "Turn the sound back on", "Unmute the audio",
                         "Unmute the speakers", "Bring the sound back"]))
    return finish(rng, t), {"muted": muted}


def g_media(rng):
    a = pick(rng, "play", "pause", "next", "previous", "stop")
    t = {"play": ["Play the music", "Resume", "Resume the video", "Play", "Keep playing", "Unpause the music",
                  "Start the music again"],
         "pause": ["Pause", "Pause the music", "Pause the video", "Hold on, pause it", "Pause playback",
                   "Pause the song"],
         "next": ["Next song", "Skip this track", "Skip", "Play the next one", "Next track", "Skip this song"],
         "previous": ["Previous song", "Go back a track", "Play the last song again", "Previous track",
                      "Back to the previous song"],
         "stop": ["Stop the music", "Stop playback", "Stop the video", "Stop playing", "Stop the song"]}[a]
    return finish(rng, rng.choice(t)), {"action": a}


def g_brightness(rng):
    n = rng.choice([0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100, 15, 35, 45, 65, 85])
    s = number(rng, n, 0.35)
    t = pick(rng, f"Set the brightness to {s}", f"Brightness {s}", f"Screen brightness to {s} percent",
             f"Put the brightness at {s}", f"Dim the screen to {s}", f"Change the brightness to {s}",
             f"Make the screen {s} percent bright")
    return finish(rng, t), {"level": n}


def g_toggle(thing_say, on_extra=(), off_extra=()):
    def g(rng):
        enabled = rng.random() < 0.55
        return finish(rng, on_off(rng, rng.choice(thing_say), enabled, extra_on=on_extra, extra_off=off_extra)), \
            {"enabled": enabled}
    return g


def g_power_mode(rng):
    mode = pick(rng, "power saver", "balanced", "performance")
    say = {"power saver": ["power saver mode", "battery saver mode", "power saving mode", "power saver"],
           "balanced": ["balanced mode", "balanced power mode", "the balanced power profile", "balanced"],
           "performance": ["performance mode", "high performance mode", "the performance power profile",
                           "max performance"]}[mode]
    s = rng.choice(say)
    t = pick(rng, f"Switch to {s}", f"Set the power mode to {s}", f"Turn on {s}", f"Use {s}", f"Put it in {s}",
             f"Change power mode to {s}")
    return finish(rng, t), {"mode": mode}


def g_shutdown(rng):
    if rng.random() < 0.4:
        n = rng.choice([1, 2, 5, 10, 15, 20, 30, 45, 60])
        s = number(rng, n, 0.4)
        unit = "minute" if n == 1 else "minutes"
        t = pick(rng, f"Shut down in {s} {unit}", f"Turn off the computer in {s} {unit}",
                 f"Shut the computer down after {s} {unit}", f"Power off in {s} {unit}",
                 f"Schedule a shutdown in {s} {unit}")
        return finish(rng, t), {"delay_minutes": n}
    t = pick(rng, "Shut down the computer", "Shut down", "Turn off the computer", "Power off", "Shut it down",
             "Switch off the PC", "Power down the computer", "Turn the computer off")
    return finish(rng, t), {}


def g_settings(rng):
    if rng.random() < 0.6:
        page = draw(rng, PAGES)
        t = pick(rng, f"Open the {page} settings", f"Open {page} settings", f"Show me the {page} settings",
                 f"Go to {page} in settings", f"Take me to the {page} settings page", f"Open settings for {page}")
        return finish(rng, t), {"page": page}
    t = pick(rng, "Open settings", "Open the system settings", "Show me the settings", "Go to settings",
             "Open the control panel", "Bring up settings")
    return finish(rng, t), {}


def g_bt_device(rng):
    d = draw(rng, DEVICES_BT)
    say = pick(rng, d, f"my {d}", f"the {d}")
    t = pick(rng, f"Connect {say}", f"Connect to {say}", f"Pair {say}", f"Hook up {say}",
             f"Connect {say} over Bluetooth", f"Link {say}")
    return finish(rng, t), {"device_name": d}


def g_folder(rng):
    f = draw(rng, FOLDERS)
    say = pick(rng, f"the {f} folder", f"my {f} folder", f"{f}", f"my {f}")
    t = pick(rng, f"Open {say}", f"Show me {say}", f"Go to {say}", f"Open up {say}", f"Take me to {say}",
             f"Browse {say}")
    return finish(rng, t), {"folder": f}


def g_file(rng):
    f = draw(rng, FILES)
    say = pick(rng, f, f"the {f}", f"my {f}") if "." not in f else f
    t = pick(rng, f"Open {say}", f"Open the file {f}", f"Pull up {say}", f"Show me {say}", f"Open up {say}",
             f"Bring up {say}")
    return finish(rng, t), {"file_name": f}


def g_search_files(rng):
    q = draw(rng, SEARCHES)
    t = pick(rng, f"Search my files for {q}", f"Find {q} on my computer", f"Look for {q} in my files",
             f"Search the computer for {q}", f"Find files with {q}", f"Where are my {q}" if not q.startswith(
                 ("the", "anything", "a ")) else f"Find {q}", f"Locate {q} on this computer")
    return finish(rng, t, raw=t.startswith("Where")), {"query": q}


def g_website(rng):
    name, url, say = draw_site(rng)
    t = pick(rng, f"Open {say}", f"Go to {say}", f"Pull up {say}", f"Take me to {say}", f"Load {say}",
             f"Open {say} in the browser", f"Open the {name} website" if not name.startswith("the") and "." not in name else f"Open {say}")
    return finish(rng, t), {"url": url}


def g_web_search(rng):
    q = draw(rng, QUERIES)
    t = pick(rng, f"Search the web for {q}", f"Google {q}", f"Look up {q}", f"Search for {q} online",
             f"Do a web search for {q}", f"Search online for {q}", f"Look up {q} on the internet")
    return finish(rng, t), {"query": q}


def g_type(rng):
    text = draw(rng, TEXTS)
    t = pick(rng, f"Type {text}", f"Type out {text}", f"Write {text}", f"Type the words {text}",
             f"Type in {text}", f"Enter the text {text}")
    return finish(rng, t), {"text": text}


def g_keys(rng):
    if rng.random() < 0.25:
        t, keys = rng.choice(KEY_INTENTS)
        return finish(rng, t), {"keys": keys}
    say, keys = rng.choice(KEYS) if rng.random() < 0.4 else random_keys(rng)
    t = pick(rng, f"Press {say}", f"Hit {say}", f"Press the {say} keys" if " " in say and not say.startswith("the")
             else f"Press {say}", f"Do {say}", f"Send {say}", f"Push {say}")
    return finish(rng, t), {"keys": keys}


def g_clipboard(rng):
    text = draw(rng, CLIPS)
    t = pick(rng, f"Copy {text} to the clipboard", f"Put {text} on the clipboard", f"Copy this: {text}",
             f"Save {text} to my clipboard", f"Copy the text {text}", f"Add {text} to the clipboard")
    return finish(rng, t), {"text": text}



INTRO_ASK = ["Who are you", "What's your name", "What can you do", "What are you", "Who am I talking to",
             "Are you Dana", "What do you do", "What's your job", "How can you help me", "What can you help me with",
             "What can you do in this car", "Are you the car's assistant", "Are you the assistant in this car",
             "What kind of assistant are you", "Who is this", "What should I call you",
             "Do you have a name", "What are you able to do", "What can I ask you", "Who's speaking"]
INTRO_TELL = ["Introduce yourself", "Tell me about yourself", "Tell me who you are", "Tell me what you can do",
              "Say hello and introduce yourself", "Tell me your name", "Give me a quick introduction",
              "Explain what you can do", "Introduce yourself to my passenger", "Say hi to my friend",
              "Let me know what you can help with", "Tell us a bit about yourself"]
INTRO_START = ["", "", "", "Hey, ", "Hi, ", "Hello, ", "Dana, ", "Hey Dana, ", "Hi Dana, ", "So, ", "Okay, ",
               "Excuse me, ", "Hello there, "]


def g_intro(rng):
    """Questions keep a question mark; requests may take the usual polite start."""
    if rng.random() < 0.6:
        start, ask = rng.choice(INTRO_START), rng.choice(INTRO_ASK)
        t = start + ask[:1].lower() + ask[1:] if start else ask
        return t + rng.choice(["?", "?", "?", "."]), {}
    t = rng.choice(INTRO_TELL)
    if rng.random() < 0.3:
        t = rng.choice(["Hey Dana, ", "Dana, ", "Hi, ", "Hey, "]) + t[:1].lower() + t[1:]
        return t + rng.choice([".", ".", "", " please."]), {}
    return finish(rng, t), {}

PEOPLE = ["mom", "dad", "John", "Sarah", "my boss", "Alex", "Grandma", "Mike", "the dentist", "Emma", "Kenji"]
TIMES = ["5 minutes", "ten minutes", "an hour", "7 am", "noon", "tomorrow morning", "3 pm", "half an hour", "tonight"]
NONE_TEMPLATES = [
    "Remind me to {task} {when}", "Set a reminder to {task}", "Set an alarm for {time}",
    "Set a timer for {time}", "Wake me up at {time}", "Call {person}", "Text {person} that I'm running late",
    "Send a message to {person}", "Email {person} the report", "What time is it", "What's the date today",
    "What day is it", "How are you", "Tell me a joke", "Thank you", "Thanks, that's all",
    "Never mind", "Cancel that", "Forget it", "Good morning", "Good night",
    "What's {a} times {b}", "What's {a} plus {b}", "How do you spell {word}", "Translate {word} into Spanish",
    "Order a pizza", "Book a table for {n} at {time}", "Add {item} to my shopping list", "Order more {item}",
    "Play some music by {artist}", "Play {artist} on Spotify", "Navigate to the airport", "How far is {city}",
    "Call an Uber", "Schedule a meeting with {person} {when}", "Add {task} to my calendar",
    "What's on my calendar {when}", "Read my latest email", "Read me the news", "How's traffic",
    "Turn on the kitchen lights", "Set the thermostat to {a} degrees", "Lock the front door",
    "Start the vacuum", "Feed the cat", "Water the plants", "What's the meaning of life",
    "Do you like music", "Sing a song", "I'm bored", "I love you", "Who won the {sport} game",
    "Find me a recipe for {item}",
    "Pay the electricity bill", "Transfer {a} dollars to {person}", "Buy {item}", "How old is {person}",
    "Hmm", "Uh, wait", "Okay", "Yes", "No", "Hello", "Hi there", "Are you listening",
    "How much is a flight to {city}", "Book a hotel in {city}", "How much does {item} cost",
    "Is it going to rain {when}", "What's the weather in {city}", "Track my package", "Check my bank balance",
    "How many calories are in {item}", "Find a dentist near me", "Who is the president of France",
    "Convert {a} dollars to euros", "When is the next train to {city}", "Get me a taxi to {city}",
]
TASKS = ["call mom", "buy milk", "take my medicine", "pick up the kids", "pay rent", "water the plants",
         "send the invoice", "go to the gym", "check the oven", "book flights"]
WHENS = ["tomorrow", "at 6", "in an hour", "tonight", "on Monday", "next week", "after lunch", "later"]
ITEMS = ["milk", "eggs", "bread", "coffee", "bananas", "dog food", "paper towels", "chicken curry", "pancakes"]
ARTISTS = ["Taylor Swift", "the Beatles", "Miles Davis", "Radiohead", "Daft Punk", "Adele", "Bach"]
CITIES = ["Kyoto", "Osaka", "Paris", "Boston", "the beach", "Berlin"]
WORDS = ["necessary", "thank you", "rhythm", "good morning", "library", "Wednesday"]


def g_none(rng):
    t = rng.choice(NONE_TEMPLATES).format(
        task=rng.choice(TASKS), when=rng.choice(WHENS), time=rng.choice(TIMES), person=rng.choice(PEOPLE),
        a=rng.randint(2, 99), b=rng.randint(2, 20), word=rng.choice(WORDS), n=rng.randint(2, 8),
        item=rng.choice(ITEMS), artist=rng.choice(ARTISTS), city=rng.choice(CITIES),
        sport=pick(rng, "football", "baseball", "basketball"))
    raw = t.startswith(("What", "How", "Who", "Do you", "Are you", "I ", "Hmm", "Uh", "Okay", "Yes", "No",
                        "Hello", "Hi", "Thank", "Good", "Never", "Forget"))
    t = finish(rng, t, raw=raw)
    if raw and t[-1] == "." and t.split()[0] in {"What", "What's", "How", "How's", "Who", "Do", "Are"}:
        t = t[:-1] + "?"
    return t, None


GENERATORS = {
    "introduce_yourself": g_intro,
    "minimize_window": simple("Minimize this window", "Minimize the window", "Minimize", "Hide this window",
                              "Shrink this window down", "Send this window to the taskbar", "Minimize it",
                              "Minimize the current window", "Get this window out of the way"),
    "maximize_window": simple("Maximize this window", "Maximize the window", "Maximize", "Make this window bigger",
                              "Make the window fill the screen", "Maximize it", "Blow this window up",
                              "Maximize the current window", "Make this window as big as possible"),
    "split_screen": g_split,
    "switch_virtual_desktop": g_desktop,
    "toggle_vpn": g_vpn,
    "switch_audio_output": g_audio_out,
    "switch_audio_input": g_audio_in,
    "print_document": g_print,
    "check_system_updates": simple("Check for updates", "Are there any system updates", "Check for system updates",
                                   "Any updates available", "See if there are updates",
                                   "Look for software updates", "Is my system up to date",
                                   "Check if there are new updates", "Is there a newer version of the system",
                                   "Is there a new version available", "Is my computer out of date",
                                   "Do I need to update anything", "Has a new version come out",
                                   "Is there an update for the operating system"),
    "install_system_updates": simple("Install the updates", "Install system updates", "Update the system",
                                     "Install all available updates", "Run the updates", "Update my computer",
                                     "Go ahead and install the updates", "Apply the system updates"),
    "take_screenshot_region": g_screenshot,
    "kill_process": g_kill,
    "check_system_resources": simple("Check system resources", "How much memory am I using",
                                     "What's the CPU usage", "Show me CPU and memory usage", "How's my disk space",
                                     "Check the CPU, memory and disk", "Why is my computer so slow",
                                     "How much RAM is free", "Show system usage", "How busy is the CPU"),
    "open_app": g_open_app,
    "switch_to_app": g_switch_app,
    "close_window": simple("Close this window", "Close the window", "Close it", "Close the current window",
                           "Get rid of this window", "Shut this window", "Close that"),
    "restore_window": simple("Restore the window", "Restore this window", "Put the window back to normal size",
                             "Unmaximize this window", "Restore the window size", "Make the window normal again",
                             "Restore it to its previous size", "Return the window to its old size"),
    "toggle_fullscreen": simple("Toggle full screen", "Go full screen", "Make this full screen", "Exit full screen",
                                "Leave full screen", "Full screen", "Make it full screen",
                                "Get out of full screen mode", "Enter full screen mode"),
    "show_desktop": simple("Show the desktop", "Show desktop", "Hide all windows", "Take me to the desktop",
                           "Minimize everything", "Clear the screen so I can see the desktop",
                           "Go to the desktop", "Hide everything and show the desktop"),
    "show_all_windows": simple("Show all windows", "Show me all my open windows", "Window overview",
                               "Show the overview", "Show every open window", "What windows do I have open",
                               "Open the window overview", "Show everything that's open"),
    "move_window_to_monitor": g_monitor,
    "move_window_to_desktop": lambda rng: g_desktop(rng, move=True),
    "set_volume": g_set_volume,
    "adjust_volume": g_adjust_volume,
    "mute_audio": g_mute,
    "mute_microphone": lambda rng: g_mute(rng, mic=True),
    "media_control": g_media,
    "set_brightness": g_brightness,
    "toggle_dark_mode": g_toggle(["dark mode", "the dark theme", "night mode for the theme"],
                                 ["Switch to dark mode", "Use the dark theme", "Make everything dark"],
                                 ["Switch to light mode", "Use the light theme", "Go back to light mode"]),
    "toggle_night_light": g_toggle(["night light", "the night light", "the blue light filter", "night shift"],
                                   ["Filter out the blue light", "Make the screen warmer"],
                                   ["Stop filtering blue light"]),
    "start_screen_recording": simple("Start recording the screen", "Start a screen recording", "Record my screen",
                                     "Begin screen recording", "Start capturing the screen", "Record the screen",
                                     "Start screen capture"),
    "stop_screen_recording": simple("Stop recording", "Stop the screen recording", "End the screen recording",
                                    "Stop recording and save it", "Finish the recording", "Stop capturing the screen",
                                    "That's enough, stop recording"),
    "lock_screen": simple("Lock the screen", "Lock my computer", "Lock it", "Lock the computer", "Lock up",
                          "Lock my screen", "Lock the PC"),
    "log_out": simple("Log out", "Log me out", "Sign out", "Log out of my account", "Sign me out",
                      "End my session", "Log off"),
    "sleep_computer": simple("Put the computer to sleep", "Sleep", "Go to sleep", "Put it to sleep",
                             "Suspend the computer", "Sleep mode", "Make the computer sleep"),
    "restart_computer": simple("Restart the computer", "Restart", "Reboot", "Reboot the computer",
                               "Restart my PC", "Restart the machine", "Do a reboot"),
    "shut_down_computer": g_shutdown,
    "set_power_mode": g_power_mode,
    "check_battery": simple("Check the battery", "How much battery do I have", "What's my battery level",
                            "Is it charging", "Battery status", "How's the battery", "Is the laptop plugged in",
                            "How much charge is left"),
    "toggle_do_not_disturb": g_toggle(["do not disturb", "Do Not Disturb", "focus mode"],
                                      ["Don't let anything disturb me", "Silence notifications for a while"],
                                      ["Let notifications through again"]),
    "clear_notifications": simple("Clear my notifications", "Clear all notifications", "Dismiss all notifications",
                                  "Get rid of the notifications", "Clear notifications", "Dismiss everything",
                                  "Clear the notification center"),
    "open_settings": g_settings,
    "toggle_wifi": g_toggle(["Wi-Fi", "the Wi-Fi", "wireless", "the wireless"],
                            ["Connect to Wi-Fi"], ["Disconnect from Wi-Fi"]),
    "toggle_bluetooth": g_toggle(["Bluetooth", "the Bluetooth"]),
    "connect_bluetooth_device": g_bt_device,
    "open_folder": g_folder,
    "open_file": g_file,
    "search_files": g_search_files,
    "empty_trash": simple("Empty the trash", "Empty the recycle bin", "Clear the trash", "Delete everything in the trash",
                          "Empty trash", "Take out the trash", "Clean out the trash"),
    "open_website": g_website,
    "web_search": g_web_search,
    "type_text": g_type,
    "press_keys": g_keys,
    "copy_to_clipboard": g_clipboard,
}


def joined(rng, a, b):
    b = b[:1].lower() + b[1:] if not b.split()[0] in {"I", "VPN", "Wi-Fi", "HDMI"} else b
    a = a.rstrip(".?!")
    return pick(rng, f"{a} and {b}", f"{a}, then {b}", f"{a}, and also {b}", f"{a} and then {b}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tools", default="examples/tools/desktop.json")
    p.add_argument("--out", default="data/voice/desktop_commands.jsonl")
    p.add_argument("--per-tool", type=int, default=150)
    p.add_argument("--multi", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", default=None, help="Comma-separated tools to write commands for (default: all)")
    p.add_argument("--wide", action="store_true",
                   help="Draw most values from long lists or make them up, so Dana learns to copy values")
    p.add_argument("--none", type=int, default=0,
                   help="Also this many requests no tool can carry out, whose answer is no call")
    args = p.parse_args()
    if args.wide:
        global WIDE
        WIDE = True
        _wide_setup()

    names = [t["name"] for t in json.load(open(args.tools))]
    if args.only is not None:
        wanted = [n for n in args.only.split(",") if n]
        unknown = set(wanted) - set(names)
        if unknown:
            raise SystemExit(f"--only: not in {args.tools}: {', '.join(sorted(unknown))}")
        names = wanted
    missing = set(names) - set(GENERATORS)
    if missing:
        raise SystemExit(f"no templates for: {', '.join(sorted(missing))}")
    rng = random.Random(args.seed)
    rows, seen, singles = [], set(), {}

    def key(q):
        return " ".join(re.sub(r"[^\w' ]", " ", q.lower()).split())

    for name in names:
        got, tries = [], 0
        while len(got) < args.per_tool and tries < args.per_tool * 40:
            tries += 1
            query, arguments = GENERATORS[name](rng)
            if key(query) in seen:
                continue
            seen.add(key(query))
            got.append({"query": query, "answers": [call(name, **arguments)]})
        singles[name] = got
        rows += got
    got, tries = 0, 0
    while got < args.none and tries < args.none * 60:
        tries += 1
        query, _ = g_none(rng)
        if key(query) not in seen:
            seen.add(key(query))
            rows.append({"query": query, "answers": []})
            got += 1
    if got < args.none:
        print(f"only {got} distinct no-tool requests (asked for {args.none})", file=sys.stderr)
    for _ in range(round(len(rows) * args.multi) if len(names) > 1 else 0):
        a, b = rng.sample(names, 2)
        ra, rb = rng.choice(singles[a]), rng.choice(singles[b])
        query = joined(rng, ra["query"], rb["query"])
        query = query[:1].upper() + query[1:]
        if query[-1] not in ".?!":
            query += "."
        if key(query) in seen:
            continue
        seen.add(key(query))
        rows.append({"query": query, "answers": ra["answers"] + rb["answers"]})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    thin = {n: len(v) for n, v in singles.items() if len(v) < args.per_tool}
    print(f"{len(rows)} commands -> {args.out}", file=sys.stderr)
    if thin:
        print("fewer than --per-tool for: " + ", ".join(f"{n} ({c})" for n, c in thin.items()), file=sys.stderr)


if __name__ == "__main__":
    main()
