# config.py
# ----------------------------------------------------------------
# configuration file and constant settings
# ----------------------------------------------------------------
# adriana r.f. (@adrmisty)
# jan-2026


# VERBS: ACTION ROOTS
V_ROOTS = [
    "click", "post", "chat", "link", "tag", "tweet", 
    "scan", "format", "hack", "ban", "log", "reset", 
    "download", "stream", "like", "scroll", "update", 
    "forward", "spam", "check", "spoiler", "troll"
]

# NOUNS: ENTITIES (no light verbs)
N_ROOTS = [
    "bug", "server", "cloud", "software", "hardware", 
    "online", "interface", "user", "bot", "app", "pixel"
]

ALL_ROOTS = V_ROOTS + N_ROOTS

# GREEK TRANSLITERATIONS
GREEK_TRANSLITERATION = {
    "click": "κλικ", 
    "post": "ποστ", 
    "chat": "τσατ", 
    "link": "λινκ",
    "tag": "ταγκ", 
    "tweet": "τουίτ", 
    "scan": "σκαν", 
    "format": "φορμάτ",
    "hack": "χακ", 
    "ban": "μπαν", 
    "log": "λογκ", 
    "reset": "ρισέτ",
    "download": "νταουνλόουντ", 
    "stream": "στριμ", 
    "like": "λάικ",
    "scroll": "σκρολ", 
    "update": "απντέιτ", 
    "forward": "φόργουορντ",
    "spam": "σπαμ", 
    "check": "τσεκ", 
    "spoiler": "σπόιλερ",
    "pixel": "πίξελ", 
    "bug": "μπαγκ", 
    "server": "σέρβερ",
    "cloud": "κλάουντ", 
    "software": "σόφτγουερ", 
    "hardware": "χάρντγουερ",
    "online": "ονλάιν", 
    "interface": "ιντερφέις", 
    "user": "γιούζερ",
    "bot": "μποτ", 
    "app": "απ", #problematic!
    "troll": "τρολ"
}

# TO AVOID MINING ENGLISH TEXTS
ENGLISH_STOPWORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there",
    "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time", "no",
    "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could", "them", "see", "other", "than", "then",
    "now", "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", "how", "our", "work", "first", "well",
    "way", "even", "new", "want", "because", "any", "these", "give", "day", "most", "us", "is", "was", "are", "were", "has", "had", "been",
    "references", "external", "links", "bibliography", "source", "title", "date", "author", "publisher", "retrieved",
    "archived", "original", "page", "pages", "volume", "issue", "doi", "isbn", "issn", "pmid", "journal", "university", "press",
    "abstract", "introduction", "conclusion", "chapter"
}