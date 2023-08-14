import pygame
import numpy as np
import pickle
import os
import time
import imageio

class TicTacToe:
    """
    This class implements the TicTacToe game as a Markov Decision Process, with the opponent embedded into the environment.

    Each state represents a board configuration in which the next turn is from the player (agent). For each move from the player the environment
    will return the resulting board with the player's move and the opponent's reaction (if the game is not over).

    We have a winner when one of the 2 players forms 3 consecutive marks in the board in the horizontal, vertical or diagonal direction. We have
    a draw when all the slots from the board are filled and none of the players were able to get 3 consecutive marks.
    """
    REWARD_WIN = 1
    REWARD_LOSS = -1
    REWARD_DRAW = 0
    REWARD_INVALID = -2
    REWARD_REGULAR = 0

    def __init__(self, adversary_policy='random', render_lag=None, exploring_starts=False):

        self.intermediate_board = None
        self.board = None
        self.render_lag = render_lag
        self.exploring_starts = exploring_starts
        self.obs_to_state_map, self.obs_to_after_state_map = self.load_obs_to_state_mapping() # mapping from encoded board to a state number
        self.boards = []

        if adversary_policy == 'random':
            self.adversary_policy = self.random_policy
        elif adversary_policy == 'optimal':
            self.adversary_policy = self.optimal_policy
        else:
            raise "adversary_policy can only be 'random' or 'optimal'."

    def reset(self, first_turn=0):
        """Resets the intermediate board to an empty board. Resets the board to an empty board if the agent is the first player, or to a board with
        a random move from the opponent if the opponent is the first player.

        Args:
            int 'first_turn': -1 if first turn is from player, 1 if first turn is from opponent and 0 if first turn is randomly assigned

        Returns:
            int state: initial board represented as a state number
            dict info: dictionary with additional relevant information
        """

        assert first_turn in (-1,0,1), "'first_turn' must be -1, 0 or 1."

        if self.render_lag is not None:
            pygame.init()
            self.ttt = pygame.display.set_mode((245,245))
            pygame.display.set_caption('Tic-Tac-Toe')
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            self._draw_empty_board()
            self._show_board()

        if first_turn == 0:
            first_turn = (np.random.random() >= 0.5)*2 -1

        self.intermediate_board = np.zeros((3,3))
        self.board = np.zeros((3,3))

        if first_turn == -1:
            self._adversary_move()

        if self.exploring_starts:
            encoded_board = np.random.choice(list(self.obs_to_state_map.keys()))
            board = self._decode_board(encoded_board)
            self.intermediate_board = board
            self.board = board

        return self.obs_to_state(self.board), {}

    def step(self, action):
        """
        Plays move from player. If game is not over, plays move from opponent according to the adversary policy and verifies if game is over.

        If player chooses a position in the board already filled, the game is truncated and a -2 reward is returned.
        Args:
            int 'action': encoded action from 0 to 8, representing the 9 positions in the board to play

        Returns:
            int state: board represented as a state number, after the end of the current round
            float reward: reward provided by the environment (-1 if opponents wins, 1 if player wins, 0 if draw and -2 if invalid move)
            bool done: True if the game is over (win, loss or draw) and False otherwise
            bool truncated: True if the player plays an invalid move in a position already filled with a mark and False otherwise
            dict info: dictionary with additional relevant information
        """

        self.intermediate_board[self.board != self.intermediate_board] = self.board[self.board != self.intermediate_board]

        done, result = self.is_game_over(self.board)

        if done:
            return self.obs_to_state(self.board), 0, done, False, {}

        if not self._validate_move(action):
            return self.obs_to_state(self.board), self.REWARD_INVALID, False, True, {}

        self._play_move(action)
        done, result = self.is_game_over(self.board)

        if done:
            return self.obs_to_state(self.board), result, True, False, {}

        self._adversary_move()
        done, result = self.is_game_over(self.board)

        return self.obs_to_state(self.board), result if done else self.REWARD_REGULAR, done, False, {}

    def is_game_over(self, board):
        """
        Checks if the game is over due to a win, loss or draw.

        Args:
            np.array (3,3) 'board': board represented as an array of integers (-1, 0 and 1)

        Returns:
            bool done: True if the game is over (win, loss or draw) and False otherwise
            int result: 1 if player wins, -1 if opponent wins and 0 if draw
        """

        if (np.abs(board.sum(axis=0)) == 3).any():
            if (board.sum(axis=0) == 3).any():
                return True, 1
            return True, -1

        if (np.abs(board.sum(axis=1)) == 3).any():
            if (board.sum(axis=1) == 3).any():
                return True, 1
            return True, -1

        if board[[0,1,2],[0,1,2]].sum() == 3:
            return True, 1

        if board[[0,1,2],[0,1,2]].sum() == -3:
            return True, -1

        if board[[0,1,2],[2,1,0]].sum() == 3:
            return True, 1

        if board[[0,1,2],[2,1,0]].sum() == -3:
            return True, -1

        if (board != 0).all():
            return True, 0

        return False, 0

    def action_to_pos(self, action):
        """
        Converts the action number into the respective position in the board.

        Args:
            int 'action': encoded action from 0 to 8, representing the 9 positions in the board to play

        Returns:
            int x: x-coordinate of the board position
            int y: y-coordiante of the board position
        """

        x = action // 3
        y = action % 3
        return x,y

    def pos_to_action(self, x, y):
        """
        Converts the position in the board into the respective action number.

        Args:
            int x: x-coordinate of the board position
            int y: y-coordiante of the board position

        Returns:
            int 'action': encoded action from 0 to 8, representing the 9 positions in the board to play
        """

        return 3*x+y

    def obs_to_state(self, board):
        """
        Encodes a board represented as a 3x3 array into an integer and maps it into a state number.

        Args:
            np.array (3,3) 'board': board represented as an array of integers (-1, 0 and 1)

        Returns:
            int state: board represented as a state number, after the end of the current round
        """

        return self.obs_to_state_map[self._encode_board(board)]

    def obs_to_after_state(self, board):
        """
        Encodes a board represented as a 3x3 array into an integer and maps it into an after state number.

        Args:
            np.array (3,3) 'board': board represented as an array of integers (-1, 0 and 1)

        Returns:
            int after_state: board represented as an after state number, after the player's move
        """

        return self.obs_to_after_state_map[self._encode_board(board)]

    def save_obs_to_state_mapping(self):
        """
        Computes the mapping between the encoded board and state/after-state number, for each possible board configuration. Saves the mappings into
        .pkl files.

        A state is related to a board configuration where the next turn is from the player. An after state is a board configuration where
        the next move is from the opponent.

        Returns:
            dict obs_to_state_map: mapping between the encoded board and state number
            dict obs_to_after_state_map: mapping between the encoded board and after state number
        """

        obs_to_state_map, obs_to_after_state_map = dict(), dict()
        i,j = self._recursive_obs_to_state_mapping(np.zeros((3,3)), obs_to_state_map, obs_to_after_state_map, 1, 2, 1)
        self._recursive_obs_to_state_mapping(np.zeros((3,3)), obs_to_state_map, obs_to_after_state_map, i, j, -1)
        curr_path = os.path.dirname(__file__)
        with open(os.path.join(curr_path, 'obs_to_state_map.pkl'), 'wb') as f:
            pickle.dump(obs_to_state_map, f)
        with open(os.path.join(curr_path, 'obs_to_after_state_map.pkl'), 'wb') as f:
            pickle.dump(obs_to_after_state_map, f)
        return obs_to_state_map, obs_to_after_state_map

    def load_obs_to_state_mapping(self):
        """
        Tries to find the .pkl files with the mappings from encoded boards into state/after state. If files are not found, calls
        'save_obs_to_state_mapping' method.

        Args:
            bool after_state: True if the desired mapping output is the after state, False if it is the state.

        Returns:
            dict obs_to_state_map: mapping between encoded board and state or after state.
        """

        try:
            curr_path = os.path.dirname(__file__)
            with open(os.path.join(curr_path, 'obs_to_state_map.pkl'), 'rb') as f:
                obs_to_state_map = pickle.load(f)
            with open(os.path.join(curr_path, 'obs_to_after_state_map.pkl'), 'rb') as f:
                obs_to_after_state_map = pickle.load(f)
            return obs_to_state_map, obs_to_after_state_map
        except Exception as e:
            print(e)
            return self.save_obs_to_state_mapping()

    def get_allowed_actions_mask(self):
        """Computes a boolean matrix of allowed actions for each state, based on the empty positions in the board.

        Returns:
            bool np.array (n_s, n_a) allowed_actions_mask: boolean mask that is True for the allowed actions and False otherwise.
        """

        num_states = max(self.obs_to_state_map.values())+1
        num_actions = 9
        allowed_actions_mask = np.ones((num_states, num_actions), dtype=bool)

        for encoded_obs, s in self.obs_to_state_map.items():
            if s != 0:
                board = self._decode_board(encoded_obs)
                mask = (board == 0).flatten()
                allowed_actions_mask[s] = mask

        return allowed_actions_mask

    def get_state_action_to_after_state_map(self, rotation_invariant=False):
        """
        For each state-action pair, assign an after-state integer based on the configuration of the board after the action is performed.

        Args:
            bool rotation_invariant: True if rotated and mirrored boards have same after-state and False otherwise

        Returns:
            int np.array (n_s, n_a) saas_map: mapping between state-action pair and an integer represeting the after-state.
        """

        num_states = max(self.obs_to_state_map.values())+1
        num_actions = 9
        saas_map = np.zeros((num_states, num_actions), dtype=int)

        saas_map[0,:] = 0
        for encoded_obs, s in self.obs_to_state_map.items():
            if s != 0:
                board = self._decode_board(encoded_obs)
                for a in range(num_actions):
                    i,j = self.action_to_pos(a)
                    if board[i,j] == 0:
                        next_board = board.copy()
                        next_board[i,j] = 1
                        saas_map[s,a] = self.obs_to_after_state(next_board)
                    else:
                        saas_map[s,a] = 1

        if rotation_invariant:
            ri_map = np.zeros(max(self.obs_to_after_state_map.values())+1, dtype=int)
            ri_map[0] = 0
            ri_map[1] = 1
            for encoded_obs, s in self.obs_to_state_map.items():
                if s != 0:
                    board = self._decode_board(encoded_obs)
                    for a in range(num_actions):
                        i,j = self.action_to_pos(a)
                        if board[i,j] == 0:
                            next_board = board.copy()
                            next_board[i,j] = 1
                            after_state = self.obs_to_after_state(next_board)
                            if ri_map[after_state] == 0:
                                for _ in range(4):
                                    next_board = np.rot90(next_board)
                                    ri_map[self.obs_to_after_state(next_board)] = after_state
                                next_board = np.flip(next_board, 1)
                                for _ in range(4):
                                    next_board = np.rot90(next_board)
                                    ri_map[self.obs_to_after_state(next_board)] = after_state
            saas_map = ri_map[saas_map]

        return saas_map

    def random_policy(self, board):
        """
        Given a board configuration, this method randomly chooses an empty position from the board.

        Args:
            int np.array (3,3) board: board configuration (1 if agent, -1 if opponent and 0 if empty).

        Returns:
            int (x,y): coordinates of chosen position.
        """

        available_x, available_y = np.nonzero(board == 0)
        i = np.random.randint(low=0, high=len(available_x))
        return available_x[i], available_y[i]

    def optimal_policy(self, board, player=-1):
        """
        Given a board configuration, this method will choose an action according to Newell and Simon's 1972 strategy.

        Args:
            int np.array (3,3) board: board configuration (1 if agent, -1 if opponent and 0 if empty).
            int player: 1 if agent's turn and -1 if opponent's turn.

        Returns:
            int (x,y): coordinates of chosen position.
        """

        # 1: check if it can win
        pos = self._win_move(board, player)
        if pos is not None:
            return pos

        # 2: block opponent's win
        pos = self._win_move(board, -player)
        if pos is not None:
            return pos

        # 3: cause a fork
        positions = self._fork_moves(board, player)
        if len(positions) > 0:
            return positions[0]

        # 4: block opponent's fork
        pos = self._block_fork(board, player)
        if pos is not None:
            return pos

        # 5: play center
        if board[1][1] == 0:
            return (1,1)

        # 6: play opposite corner
        pos = self._opposite_corner(board, player)
        if pos is not None:
            return pos

        # 7: play empty corner
        pos = self._empty_corner(board)
        if pos is not None:
            return pos

        # 8: play empty side
        return self._empty_side(board)

    def render(self, save=False):
        """
        Renders the intermediate board and current board with a delay between them.

        Args:
            bool save: True if board image arrays are stored and False otherwise
        """

        self._draw_board(self.intermediate_board.copy())
        self._show_board()
        if save: self.boards.append(np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface()))))
        time.sleep(self.render_lag)
        self._draw_board(self.board.copy())
        self._show_board()
        if save: self.boards.append(np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface()))))
        time.sleep(self.render_lag)

    def close(self):
        """
        Closes pygame window.
        """

        pygame.quit()

    def save_gif(self, path, fps=1):
        """
        Saves board image arrays into a GIF file.

        Args:
            str path: path to save GIF file.
            int fps: frames er second desired for the GIF file.
        """

        imageio.mimsave(path, self.boards, fps=fps)

    def clean_boards(self):
        """
        Empties list of board image arrays.
        """

        self.boards = []

    def _encode_board(self, board):
        return int((3**np.arange(9)) @ (board.flatten() + 1))

    def _decode_board(self, encoded_board):
        x = encoded_board
        digits = []
        for _ in range(9):
            digits.append(x % 3)
            x = x // 3
        return np.array(digits).reshape((3,3)) - 1

    def _recursive_obs_to_state_mapping(self, board, obs_to_state_map, obs_to_after_state_map, i, j, turn):

        encoded_board = self._encode_board(board)

        if self.is_game_over(board)[0]:
            if encoded_board not in obs_to_state_map:
                obs_to_state_map[encoded_board] = 0
            if (turn == -1) and (encoded_board not in obs_to_after_state_map):
                obs_to_after_state_map[encoded_board] = 0
            return i,j

        if (turn == 1) and (encoded_board not in obs_to_state_map):
            obs_to_state_map[encoded_board] = i
            i += 1

        if (turn == -1) and (encoded_board not in obs_to_after_state_map):
            obs_to_after_state_map[encoded_board] = j
            j += 1

        available_moves_x, available_moves_y = np.nonzero(board == 0)
        for x, y in zip(available_moves_x, available_moves_y):
            next_board = board.copy()
            next_board[x,y] = turn
            i, j = self._recursive_obs_to_state_mapping(next_board, obs_to_state_map, obs_to_after_state_map, i, j, -turn)

        return i,j

    def _validate_move(self, action):
        x,y = self.action_to_pos(action)
        return self.board[x][y] == 0

    def _play_move(self, action):
        x,y = self.action_to_pos(action)
        self.intermediate_board[x][y] = 1
        self.board[x][y] = 1

    def _adversary_move(self):
        x,y = self.adversary_policy(self.intermediate_board)
        self.board[x][y] = -1

    def _win_move(self, board, player):

        win_rows = np.nonzero(board.sum(axis=1) == 2*player)[0]
        if len(win_rows) > 0:
            i = win_rows[0]
            j = np.nonzero(board[i] == 0)[0][0]
            return (i,j)

        win_cols = np.nonzero(board.sum(axis=0) == 2*player)[0]
        if len(win_cols) > 0:
            j = win_cols[0]
            i = np.nonzero(board[:,j] == 0)[0][0]
            return (i,j)

        diag = np.array([board[0][0], board[1][1], board[2][2]])
        if diag.sum() == 2*player:
            i = np.nonzero(diag == 0)[0][0]
            return (i,i)

        diag = np.array([board[0][2], board[1][1], board[2][0]])
        if diag.sum() == 2*player:
            i = np.nonzero(diag == 0)[0][0]
            return (i,2-i)

        return None

    def _fork_moves(self, board, player):

        moves = []
        empty_x, empty_y = np.nonzero(board == 0)
        if len(empty_x) > 0:
            for i,j in zip(empty_x, empty_y):
                if (board[i].sum() == player) or (board[:,j].sum() == player):
                    if board[i].sum() == board[:,j].sum():
                        moves.append((i,j))
                    elif (((i+j) % 2) == 0) and ((i,j) != (1,1)) and (board[1][1] + board[2-i][2-j] == player):
                        moves.append((i,j))
                    elif (i == j == 1) and ((board[0][0] + board[2][2] == player) or (board[0][2] + board[2][0] == player)):
                        moves.append((i,j))
                elif (i == j == 1) and (board[0][0] + board[2][2] == player) and (board[0][2] + board[2][0] == player):
                    moves.append((i,j))
        return moves

    def _block_fork(self, board, player):

        opponent = -player
        opponent_fork_pos = self._fork_moves(board, opponent)

        if len(opponent_fork_pos) == 0:
            return None

        if len(opponent_fork_pos) == 1:
            return opponent_fork_pos[0]

        positions = self._get_two_sum_positions(board, player)
        if len(positions) > 0:
            candidate = None
            for two_sum_pos, two_sum_last_pos in positions:
                if two_sum_last_pos not in opponent_fork_pos:
                    if two_sum_pos in opponent_fork_pos:
                        return two_sum_last_pos
                    candidate = two_sum_pos
            if candidate is not None:
                return candidate

        return None

    def _get_two_sum_positions(self, board, player):

        two_sum_positions = []

        rows = np.nonzero(board.sum(axis=1) == player)[0]
        for i in rows:
            cols = np.nonzero(board[i] == 0)[0]
            two_sum_positions.append([(i, j) for j in cols])

        cols = np.nonzero(board.sum(axis=0) == player)[0]

        for j in cols:
            rows = np.nonzero(board[:, j] == 0)[0]
            two_sum_positions.append([(i, j) for i in rows])

        diag = np.array([board[0][0], board[1][1], board[2][2]])

        if diag.sum() == player:
            rows = np.nonzero(diag == 0)[0]
            two_sum_positions.append([(i, i) for i in rows])
            two_sum_positions.append([(i, i) for i in rows[-1::-1]])

        diag = np.array([board[0][2], board[1][1], board[2][0]])

        if diag.sum() == player:
            rows = np.nonzero(diag == 0)[0]
            two_sum_positions.append([(i, 2-i) for i in rows])
            two_sum_positions.append([(i, 2-i) for i in rows[-1::-1]])

        return [pos for pos in two_sum_positions if len(pos) > 0]

    def _opposite_corner(self, board, player):
        opponent = -player
        for i,j in [(0,0), (0,2), (2,0), (2,2)]:
            if board[i][j] == 0 and board[2-i][2-j] == opponent:
                return (i,j)
        return None

    def _empty_corner(self, board):
        for i,j in [(0,0), (0,2), (2,0), (2,2)]:
            if board[i][j] == 0:
                return (i,j)
        return None

    def _empty_side(self, board):
        for i in range(2):
            if board[i][i+1] == 0:
                return (i, i+1)
            if board[i+1][i] == 0:
                return (i+1, i)
        return None

    def _draw_empty_board(self):
        self.surface = pygame.Surface(self.ttt.get_size())
        self.surface = self.surface.convert()
        self.surface.fill((250, 250, 250))
        #horizontal line
        pygame.draw.line(self.surface, (0, 0, 0), (85, 10), (85, 235), 2)
        pygame.draw.line(self.surface, (0, 0, 0), (160, 10), (160, 235), 2)
        # veritical line
        pygame.draw.line(self.surface, (0, 0, 0), (10,85), (235, 85), 2)
        pygame.draw.line(self.surface, (0, 0, 0), (10,160), (235, 160), 2)

    def _draw_board(self, board):
        self._draw_empty_board()
        font = pygame.font.Font(None, 24)
        filled_x, filled_y = np.nonzero(board)

        for row, col in zip(filled_x, filled_y):
            centerX = ((col) * 75) + 42
            centerY = ((row) * 75) + 42
            text = font.render('X' if board[row][col] == 1 else 'O', 1, (10, 10, 10) if board[row][col] == -1 else (255,0,0))
            self.surface.fill((250, 250, 250), (0, 300, 300, 25))
            self.surface.blit(text, (centerX, centerY))

    def _show_board(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        self.ttt.blit(self.surface, (0, 0))
        pygame.display.flip()
