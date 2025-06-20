"""Games or Adversarial Search (Chapter 5)"""

import copy
import random
from collections import namedtuple
import numpy as np
import time

# namedtuple used to generate game state:
GameState = namedtuple('GameState', 'to_move, move, utility, board, moves')

def gen_state(move = '(1, 1)', to_move='X', x_positions=[], o_positions=[], h=3, v=3):
    """
        move = the move that has lead to this state,
        to_move=Whose turn is to move
        x_position=positions on board occupied by X player,
        o_position=positions on board occupied by O player,
        (optionally) number of rows, columns and how many consecutive X's or O's required to win,
    """
    moves = set([(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]) - set(x_positions) - set(o_positions)
    moves = list(moves)
    board = {}
    for pos in x_positions:
        board[pos] = 'X'
    for pos in o_positions:
        board[pos] = 'O'
    return GameState(to_move=to_move, move=move, utility=0, board=board, moves=moves)



# ______________________________________________________________________________
# MinMax Search
def minmax(game, state):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. [Figure 5.3]"""

    player = game.to_move(state)  # Determine which player is making the move

    def max_value(state):
        """Computes the maximum utility value for the maximizing player."""
        if game.terminal_test(state):  # If game over, return the utility value
            return game.utility(state, player)
        v = -np.inf  # Initialize to negative infinity for maximization
        for a in game.actions(state):  # Iterate over all possible actions
            v = max(v, min_value(game.result(state, a)))  # Choose the best value
        return v

    def min_value(state):
        """Computes the minimum utility value for the minimizing player."""
        if game.terminal_test(state):  # If game over, return the utility value
            return game.utility(state, player)
        v = np.inf  # Initialize to positive infinity for minimization
        for a in game.actions(state):  # Iterate over all possible actions
            v = min(v, max_value(game.result(state, a)))  # Choose the best value
        return v

    # Body of minmax:
    # to be implemented by students
    #print("your code goes here 5pt")
    
    # Select the move that maximizes the minimum value the opponent can force
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a)), default=None)


def minmax_cutoff(game, state):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the cutoff depth. At that level use evaluation func."""
    #print("your code goes here 10pt")
    
    player = game.to_move(state)  # Determine which player is making the move

    def max_value(state, d):
        """Compute the maximum utility value for the maximizing player up to depth d."""
        if game.terminal_test(state) or d == 0:  # Stop if it's a terminal state or depth limit is reached
            return game.utility(state, player) if game.terminal_test(state) else game.eval1(state)

        v = -np.inf  # Initialize to negative infinity for maximization

        for a in game.actions(state):  # Iterate over all legal actions
            v = max(v, min_value(game.result(state, a), d - 1))  # Get the best minimum response
        return v

    def min_value(state, d):
        """Compute the minimum utility value for the minimizing player up to depth d."""
        if game.terminal_test(state) or d == 0:  # Stop if it's a terminal state or depth limit is reached
            return game.utility(state, player) if game.terminal_test(state) else game.eval1(state)

        v = np.inf  # Initialize to positive infinity for minimization

        for a in game.actions(state):  # Iterate over all legal actions
            v = min(v, max_value(game.result(state, a), d - 1))  # Get the best maximum response
        return v

    # Choose the move that maximizes the minimum value the opponent can force,
    # considering the cutoff depth game.d.
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a), game.d), default=None)

# ______________________________________________________________________________
def alpha_beta(game, state):
    """Search game to determine best action; use alpha-beta pruning.
     This version searches all the way to the leaves."""
    
    player = game.to_move(state)  # Identify the player whose turn it is

    def max_value(state, alpha, beta):
        """Computes the maximum value for the maximizing player using alpha-beta pruning."""
        if game.terminal_test(state):  # If it's a terminal state, return its utility value
            return game.utility(state, player)

        v = float('-inf')  # Initialize to negative infinity

        for a in game.actions(state):  # Iterate over all legal actions
            v = max(v, min_value(game.result(state, a), alpha, beta))  # Compute max value
            if v >= beta:  # Beta cutoff: stop searching further
                return v
            alpha = max(alpha, v)  # Update alpha
        return v

    def min_value(state, alpha, beta):
        """Computes the minimum value for the minimizing player using alpha-beta pruning."""
        if game.terminal_test(state):  # If it's a terminal state, return its utility value
            return game.utility(state, player)

        v = float('inf')  # Initialize to positive infinity

        for a in game.actions(state):  # Iterate over all legal actions
            v = min(v, max_value(game.result(state, a), alpha, beta))  # Compute min value
            if v <= alpha:  # Alpha cutoff: stop searching further
                return v
            beta = min(beta, v)  # Update beta
        return v

    best_score = float('-inf')  # Track the best score found
    beta = float('inf')  # Initialize beta to positive infinity
    best_action = None  # Track the best move

    for a in game.actions(state):  # Iterate over all legal actions
        v = min_value(game.result(state, a), best_score, beta)  # Compute value of move
        if v > best_score:  # Update best move if the score is better
            best_score = v
            best_action = a

    return best_action  # Return the best move found

def alpha_beta_cutoff(game, state):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""
    
    player = game.to_move(state)  # Identify the player whose turn it is

    def max_value(state, alpha, beta, depth):
        """Computes the maximum value for the maximizing player up to a cutoff depth."""
        if game.terminal_test(state) or depth == 0:  # Stop if terminal state or cutoff depth reached
            return game.utility(state, player) if game.terminal_test(state) else game.eval1(state)

        v = float('-inf')  # Initialize to negative infinity

        for a in game.actions(state):  # Iterate over all legal actions
            v = max(v, min_value(game.result(state, a), alpha, beta, depth - 1))  # Compute max value
            if v >= beta:  # Beta cutoff: stop searching further
                return v
            alpha = max(alpha, v)  # Update alpha
        return v

    def min_value(state, alpha, beta, depth):
        """Computes the minimum value for the minimizing player up to a cutoff depth."""
        if game.terminal_test(state) or depth == 0:  # Stop if terminal state or cutoff depth reached
            return game.utility(state, player) if game.terminal_test(state) else game.eval1(state)

        v = float('inf')  # Initialize to positive infinity

        for a in game.actions(state):  # Iterate over all legal actions
            v = min(v, max_value(game.result(state, a), alpha, beta, depth - 1))  # Compute min value
            if v <= alpha:  # Alpha cutoff: stop searching further
                return v
            beta = min(beta, v)  # Update beta
        return v

    best_score = float('-inf')  # Track the best score found
    beta = float('inf')  # Initialize beta to positive infinity
    best_action = None  # Track the best move

    for a in game.actions(state):  # Iterate over all legal actions
        v = min_value(game.result(state, a), best_score, beta, game.d)  # Compute value of move
        if v > best_score:  # Update best move if the score is better
            best_score = v
            best_action = a

    return best_action  # Return the best move found




def random_player(game, state):
    """A random player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None


def alpha_beta_player(game, state):
    """uses alphaBeta pruning with minmax, or with cutoff version, for AI player"""
    #print("Your code goes here 5pt.")
    """Use a method to speed up at the start to avoid searching down a long tree with not much outcome.
    Hint: for speedup use random_player for start of the game when you see search time is too long"""

    def iterative_deepening(game, state, end_time):
        """
        Perform iterative deepening depth-limited search with alpha-beta pruning.
        
        Parameters:
        - game: The game instance.
        - state: The current state of the game.
        - end_time: The time limit for making a move.

        Returns:
        - The best move found within the given time limit.
        """
        depth = 1  # Start with a depth of 1
        game.d = depth  # Set the game's depth limit
        best_move = None  # Initialize the best move as None

        while time.perf_counter() < end_time and game.d < game.maxDepth:
            # Run alpha-beta cutoff search at increasing depths
            best_move = alpha_beta_cutoff(game, state)
            depth += 1  # Increase the depth limit for the next iteration
            game.d = depth  # Update the game's depth limit

        return best_move  # Return the best move found

    # If the game timer is negative, use standard alpha-beta pruning without cutoff
    if game.timer < 0:
        game.d = -1  # Indicate that no depth limit is used
        return alpha_beta(game, state)  # Perform standard alpha-beta search

    # Start the timer
    start = time.perf_counter()
    end_time = start + game.timer  # Calculate when the time limit ends

    return iterative_deepening(game, state, end_time)  # Execute iterative deepening search


def minmax_player(game, state):
    """uses minmax or minmax with cutoff depth, for AI player"""
    #print("Your code goes here 5pt.")

    def iterative_deepening(game, state, end_time):
        """
        Perform iterative deepening depth-limited search with MinMax.

        Parameters:
        - game: The game instance.
        - state: The current state of the game.
        - end_time: The time limit for making a move.

        Returns:
        - The best move found within the given time limit.
        """
        depth = 1  # Start with a depth of 1
        game.d = depth  # Set the game's depth limit
        best_move = random_player(game, state)  # Start with a random move

        while time.perf_counter() < end_time and game.d < game.maxDepth:
            # Run MinMax cutoff search at increasing depths
            best_move = minmax_cutoff(game, state)
            depth += 1  # Increase the depth limit for the next iteration
            game.d = depth  # Update the game's depth limit

        return best_move  # Return the best move found

    # If the game timer is negative, use standard MinMax without cutoff
    if game.timer < 0:
        game.d = -1  # Indicate that no depth limit is used
        return minmax(game, state)  # Perform standard MinMax search

    # Start the timer
    start = time.perf_counter()
    end_time = start + game.timer  # Calculate when the time limit ends

    return iterative_deepening(game, state, end_time)  # Execute iterative deepening search



# ______________________________________________________________________________
# base class for Games

class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))

class TicTacToe(Game):
    """Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to_move, a cached utility, a list of moves in
    the form of a list of (x, y) positions, and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'.
    depth = -1 means max search tree depth to be used."""

    def __init__(self, size=3, k=3, t=-1):
        self.size = size
        if k <= 0:
            self.k = size
        else:
            self.k = k
        self.d = -1 # d is cutoff depth. Default is -1 meaning no depth limit. It is controlled usually by timer
        self.maxDepth = size * size # max depth possible is width X height of the board
        self.timer = t #timer  in seconds for opponent's search time limit. -1 means unlimited
        moves = [(x, y) for x in range(1, size + 1)
                 for y in range(1, size + 1)]
        self.initial = GameState(to_move='X', move=None, utility=0, board={}, moves=moves)

    def reset(self):
        moves = [(x, y) for x in range(1, self.size + 1)
                 for y in range(1, self.size + 1)]
        self.initial = GameState(to_move='X', move=None, utility=0, board={}, moves=moves)

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    @staticmethod
    def switchPlayer(player):
        assert(player == 'X' or player == 'O')
        return 'O' if player == 'X' else 'X'

    def result(self, state, move):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        try:
            moves = list(state.moves)
            moves.remove(move)
        except (ValueError, IndexError, TypeError) as e:
            print("exception: ", e)

        return GameState(to_move=self.switchPlayer(state.to_move), move=move,
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or lost or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for x in range(0, self.size):
            for y in range(1, self.size + 1):
                print(board.get((self.size - x, y), '.'), end=' ')
            print()

    def compute_utility(self, board, move, player):
        """If player wins with this move, return k if player is 'X' and -k if 'O' else return 0."""
        if (self.k_in_row(board, move, player, (0, 1), self.k) or
                self.k_in_row(board, move, player, (1, 0), self.k) or
                self.k_in_row(board, move, player, (1, -1), self.k) or
                self.k_in_row(board, move, player, (1, 1), self.k)):
            return self.k if player == 'X' else -self.k
        else:
            return 0
        
    # evaluation function, version 1
    def eval1(self, state):
        """design and implement evaluation function for state.
        Some ideas: 
        	: 1-use the number of k-1 matches for X and O For this you can use function possibleKComplete().
            : 2- expand it for all k matches
            : 3- include double matches where one move can generate 2 matches.
            """
        
        """ computes number of (k-1) completed matches. This means number of row or columns or diagonals 
        which include player position and in which k-1 spots are occuppied by player.
        """

        def possiblekComplete(move, board, player, k):
            """
            Count the number of k-length sequences in which the given move participates.
            
            Parameters:
            - move: The position being checked.
            - board: The current board state.
            - player: The player ('X' or 'O').
            - k: The required sequence length.

            Returns:
            - The total number of k-length sequences involving the given move.
            """
            match = self.k_in_row(board, move, player, (0, 1), k)  # Check horizontal sequences
            match += self.k_in_row(board, move, player, (1, 0), k)  # Check vertical sequences
            match += self.k_in_row(board, move, player, (1, -1), k) # Check diagonal (\) sequences
            match += self.k_in_row(board, move, player, (1, 1), k)  # Check diagonal (/) sequences
            return match  # Return the total number of sequences found

        # Optimization tip:
        # If the number of remaining moves is very small, it's likely unnecessary to evaluate.
        # Uncomment the following lines to skip evaluation in early game stages.
        # if len(state.moves) <= self.k / 2:
        #     return 0

        #print("Your code goes here 15pt.")

        def possiblekComplete(board, player, k):
            """
            Count the number of (k-1) sequences that exist for a player on the board.
            
            Parameters:
            - board: The current board state.
            - player: The player ('X' or 'O') whose potential sequences are counted.
            - k: The sequence length being checked.

            Returns:
            - The count of (k-1) sequences that could potentially be completed by the player.
            """
            count = 0  # Initialize count of possible sequences
            
            # Iterate through all occupied positions on the board
            for move in board:
                if board[move] == player:  # Check only positions occupied by the given player
                    # If the move is part of a nearly completed (k-1) sequence, increment the count
                    if (self.k_in_row(board, move, player, (0, 1), k - 1) or  # Horizontal check
                        self.k_in_row(board, move, player, (1, 0), k - 1) or  # Vertical check
                        self.k_in_row(board, move, player, (1, -1), k - 1) or # Diagonal (\) check
                        self.k_in_row(board, move, player, (1, 1), k - 1)):   # Diagonal (/) check
                        count += 1  # Increment count for every nearly completed sequence found
            return count  # Return the final count of (k-1) sequences

        # Compute the score for each player based on potential winning sequences
        x_score = possiblekComplete(state.board, 'X', self.k)  # Count potential wins for 'X'
        o_score = possiblekComplete(state.board, 'O', self.k)  # Count potential wins for 'O'

        return x_score - o_score  # Return the difference (positive favors 'X', negative favors 'O')



    #@staticmethod
    def k_in_row(self, board, pos, player, dir, k):
        """helpe function: Return true if there is a line of k cells in direction dir including position pos on board for player."""
        (delta_x, delta_y) = dir
        x, y = pos
        n = 0  # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = pos
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1  # Because we counted move itself twice
        return n >= k


