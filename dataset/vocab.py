characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

char2idx = {
    char : idx
    for idx, char in enumerate(characters)
}

idx2char = {
    idx : char
    for char, idx in char2idx.items()
}

BLANK_IDX = len(characters)