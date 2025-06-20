import copy
import random
import time
import sys
import math
from collections import namedtuple

# Define the GameState structure using namedtuple
GameState = namedtuple('GameState', 'to_move, move, utility, board, moves')

def random_player(game, state):
    """A random player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None

# MonteCarlo Tree Search support

class MCTS:  # Monte Carlo Tree Search implementation
    class Node:
        def __init__(self, state, par=None):
            """Initialize a node in the MCTS tree.

            Parameters:
            - state: The game state associated with this node.
            - par: Parent node (None for root).
            """
            self.state = copy.deepcopy(state)  # Store a deep copy of the game state
            self.parent = par  # Parent node
            self.children = []  # List of child nodes
            self.visitCount = 0  # Number of times this node has been visited
            self.winScore = 0  # Accumulated win score from simulations

        def getChildWithMaxScore(self):
            """Returns the child node with the highest visit count."""
            maxScoreChild = max(self.children, key=lambda x: x.visitCount)
            return maxScoreChild

    def __init__(self, game, state):
        """Initialize the Monte Carlo Tree Search.

        Parameters:
        - game: The game instance.
        - state: The initial state of the game.
        """
        self.root = self.Node(state)  # Create the root node
        self.state = state  # Store the initial game state
        self.game = game  # Store the game instance
        self.exploreFactor = math.sqrt(2)  # Exploration factor for UCT calculation

    def isTerminalState(self, utility, moves):
        """Check if the game state is terminal (win, loss, or draw)."""
        return utility != 0 or len(moves) == 0

    def monteCarloPlayer(self, timelimit=4):
        """Entry point for Monte Carlo tree search.

        Uses four stages:
        1. SELECT: Traverse the tree using the best UCT value.
        2. EXPAND: Expand the node by generating all possible child states.
        3. SIMULATE: Play a random game from the expanded node to a terminal state.
        4. BACKUP: Propagate results back up the tree.

        Parameters:
        - timelimit: Maximum search time in seconds.

        Returns:
        - The best move determined by the search.
        """
        start = time.perf_counter()
        end = start + timelimit

        #print("MCTS: your code goes here. 10pt.")
        #___________________#
        while time.perf_counter() < end:
            # Select the best node to explore
            selectedNode = self.selectNode(self.root)
            if not self.isTerminalState(selectedNode.state.utility, selectedNode.state.moves):
                # Expand the selected node if it is not a terminal state
                self.expandNode(selectedNode)
            # Simulate a random play to determine the winner
            winner = self.simulateRandomPlay(selectedNode)
            # Propagate the result back up the tree
            self.backPropagation(selectedNode, winner)
        # Return the move with the highest visit count
        winnerNode = self.root.getChildWithMaxScore()
        assert (winnerNode is not None)
        return winnerNode.state.move

    """SELECT stage function: Walks down the tree using UCT values."""
    def selectNode(self, nd):
        """Selects the best node using UCT value.

        Parameters:
        - nd: The current node to start selection from.

        Returns:
        - The most promising child node for expansion.
        """
        node = nd
        #print("Your code goes here 5pt.")
        #___________________(done)#
        while len(node.children) != 0:
            # Selecting the best child
            node = self.findBestNodeWithUCT(node)
        return node

    def findBestNodeWithUCT(self, nd):
        """Finds the child node with the highest UCT value.

        Parameters:
        - nd: The parent node whose children are evaluated.

        Returns:
        - The child node with the highest UCT value.
        """
        #print("Your code goes here 2pt.")
        #___________________(done)#
        bestUCT = -math.inf  # Initialize best UCT value
        bestNode = None  # Track the best node

        for child in nd.children:
            # Calculate UCT for each child node
            uct = self.uctValue(nd.visitCount, child.winScore, child.visitCount)
            if uct > bestUCT:
                bestUCT = uct  # Update best UCT value
                bestNode = child  # Update best node
        return bestNode  # Return the best child node

    def uctValue(self, parentVisit, nodeScore, nodeVisit):
        """Compute the Upper Confidence Bound for Trees (UCT) value for a node.

        Parameters:
        - parentVisit: Number of times the parent node was visited.
        - nodeScore: The total accumulated score of the node.
        - nodeVisit: The number of times this node has been visited.

        Returns:
        - The computed UCT value.
        """
        #print("Your code goes here 3pt.")
        #___________________(done)#
        if nodeVisit == 0:
            return math.inf  # Encourage exploration of unvisited nodes
        exploitation = nodeScore / nodeVisit  # Average score of the node
        exploration = self.exploreFactor * math.sqrt(math.log(parentVisit) / nodeVisit)  # Exploration bonus
        return exploitation + exploration  # Return the final UCT value

    """EXPAND stage function."""
    def expandNode(self, nd):
        """Expands the given node by generating all possible child nodes.

        Parameters:
        - nd: The node to be expanded.
        """
        stat = nd.state
        tempState = GameState(to_move=stat.to_move, move=stat.move, utility=stat.utility, board=stat.board, moves=stat.moves)
        for a in self.game.actions(tempState):
            childNode = self.Node(self.game.result(tempState, a), nd)
            nd.children.append(childNode)  # Add new child node to the parent

    """SIMULATE stage function"""
    def simulateRandomPlay(self, nd):
        """Simulates a random playthrough from the given node until a terminal state is reached.

        Parameters:
        - nd: The node from which to start the simulation.

        Returns:
        - 'X' if X wins, 'O' if O wins, or 'N' for a tie.
        """
        # First, check if the node is already a winning position
        winStatus = self.game.compute_utility(nd.state.board, nd.state.move, nd.state.board[nd.state.move])
        #print("your code goes here 5pt.")
        #___________________#
        if winStatus != 0:
            # Return the winning player
            return 'X' if winStatus > 0 else 'O'

        """Now roll out a random play down to a terminal state."""
        tempState = copy.deepcopy(nd.state)  # Create a deep copy of the game state

        while len(tempState.moves) > 0:
            # Randomly select a move
            action = random.choice(tempState.moves)
            # Update the state
            tempState = self.game.result(tempState, action)
            if self.game.terminal_test(tempState):
                # Check if the game has ended
                break
        # Return the winning player or 'N' for a tie
        return 'X' if tempState.utility > 0 else 'O' if tempState.utility < 0 else 'N'

    def backPropagation(self, nd, winningPlayer):
        """Propagates simulation results back up the tree to update visit counts and scores.

        Parameters:
        - nd: The leaf node from which backpropagation starts.
        - winningPlayer: The player who won the simulation ('X', 'O', or 'N' for a tie).
        """
        #print("Your code goes here 5pt.")
        #___________________(seems done)#
        tempNode = nd  # Start from the current node

        while tempNode is not None:
            tempNode.visitCount += 1  # Increment visit count
            if tempNode.state.to_move != winningPlayer:
                tempNode.winScore += 1  # Reward winning paths
            tempNode = tempNode.parent  # Move up to the parent node
