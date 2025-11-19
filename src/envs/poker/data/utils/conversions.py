
# Convert card rank to a representative number.
rank_to_number = {
    "2": 1, 
    "3": 2, 
    "4": 3, 
    "5": 4, 
    "6": 5, 
    "7": 6, 
    "8": 7, 
    "9": 8, 
    "T": 9, 
    "J": 10, 
    "Q": 11, 
    "K": 12, 
    "A": 13
}

# Convert the card suit to a number (index).
suit_to_index = {
    "S":1,
    "H":2,
    "D":3,
    "C":4
}

# Convert an action character to a vector.
action_to_vector = {
    "c": [0],  # call
    "r": [1],  # raise
    "b": [1],  # bet (= raise)
    "f": [2],  # fold
    "k": [3]   # check
}