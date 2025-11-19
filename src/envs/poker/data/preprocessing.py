from envs.poker.data.utils.conversions import rank_to_number, suit_to_index, action_to_vector

def get_data(num_transitions: int = 100_000,
             verbose: bool = True) -> tuple[list, list]:
    """ 
    Gathers and returns the poker data in vector form.

    Arguments
    ---------
    num_transitions :  The number of transitions (i.e., data samples) to gather.
    verbose         :  The verbosity.

    Returns
    -------
    vector_data :  The poker data in vector form.

    """
    if verbose:
        print('Gathering data ... ')
    from envs.poker.data.dataset import readable_poker_data

    # Create the state-action vectors.
    progress = 0
    state_vectors = []
    action_vectors = []

    done = False
    for game_id, game in readable_poker_data.items():
        if done:
            break

        # Process the game states.
        states = game[0]
        actions = game[1]

        for i in range(len(states)):
            if progress >= num_transitions:
                done = True
                break
            
            # Get the game state information.
            state = states[i]
            private_cards = state[0]
            public_cards = state[1]
            raises = state[2]

            # Get the game action information.
            action = actions[i]

            # Create the vectors.
            state_vector = get_state_vector(private_cards, public_cards, raises)
            action_vector = action_to_vector[action]

            # Store the vectors.
            state_vectors.append(state_vector)
            action_vectors.append(action_vector)
            progress += 1
    
    return state_vectors, action_vectors

def get_state_vector(private_cards: list, 
                     public_cards: list, 
                     raises: list,
                     card_format: str = 'rs') -> list:
    """
    Given the private cards and public cards of a game state in readable form, 
    this function returns these cards in a (binary) vector form. The vector 
    form requires 49 indices to represent the cards, and 12 indices to represent 
    the raises, resulting in a state vector of size 61 in total.

    Arguments
    ---------
    private_cards :  The private cards of the game state in readable form.
    public_cards  :  The public cards of the game state in readable form.
    card_format   :  Specifies the format of the card. The format is either 
                     rank-suit (rs) or suit-rank (sr).
    
    Returns
    -------
    cards_vector :  The cards of the game state, in vector form.
    
    """
    cards_vector = get_cards_vector(private_cards, public_cards, card_format)
    raises_vector = get_raises_vector(raises)
    return cards_vector + raises_vector

def get_cards_vector(private_cards: list, 
                     public_cards: list,  
                     card_format: str = 'rs') -> list:
    """
    Given the private cards and public cards of a game state in readable form, 
    this function returns these cards in a (binary) vector form. The vector 
    form requires 49 indices to represent the cards.

    Arguments
    ---------
    private_cards :  The private cards of the game state in readable form.
    public_cards  :  The public cards of the game state in readable form.
    card_format   :  Specifies the format of the card. The format is either 
                     rank-suit (rs) or suit-rank (sr).
    
    Returns
    -------
    cards_vector :  The cards of the game state, in vector form.
    
    """
    # Create a vector representing the given cards.
    cards_vector = []
    all_cards = private_cards + public_cards
    for card in all_cards:
        if card_format == 'rs':  # Rank then suit, e.g., '3C'.
            rank_binary = number_to_binarylist(rank_to_number[card[0].upper()], 13)
            suit_binary = number_to_binarylist(suit_to_index[card[1].upper()], 4)
        else:                    # Suit then rank, e.g., 'C3'.
            suit_binary = number_to_binarylist(suit_to_index[card[0].upper()], 4)
            rank_binary = number_to_binarylist(rank_to_number[card[1].upper()], 13)
        cards_vector += rank_binary + suit_binary
    cards_vector += [0] * (49 - len(cards_vector))

    return cards_vector

def get_raises_vector(raises: list) -> list:
    """
    Given the raises of a game in readable form, 
    this function returns them in vector form. 

    Arguments
    ---------
    raises :  The raises of the game in readable form.

    Returns
    -------
    raises_vector :  The raises of the game in vector form.
    
    """
    raises_vector = []
    for r in raises:
        binary = number_to_binarylist(r, 4)
        raises_vector += binary
    return raises_vector

def number_to_binarylist(number: int, max_number: int) -> list:
    """
    Converts a number into a binary number in list form.

    Arguments
    ---------
    number     :  The number we want to convert.
    max_number :  The largest number allowed. 
                  This determines the length of the list.
    
    Returns
    -------
    binary :  The binary number in list form.
    
    """
    binary_size = len(bin(max_number)[2:])
    binary = bin(number)[2:]
    binary = '0' * (binary_size - len(binary)) + binary
    binary = [int(digit) for digit in binary]
    return binary
