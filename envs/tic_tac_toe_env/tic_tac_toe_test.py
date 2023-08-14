import numpy as np
from .tic_tac_toe import TicTacToe

def test_encoding():

    ttt = TicTacToe()
    for _ in range(100):
        board = np.random.randint(-1,2, (3,3))
        assert (ttt._decode_board(ttt._encode_board(board)) == board).all()

def test_is_game_over():

    ttt = TicTacToe()

    # unfinished boards

    board = np.zeros((3,3))
    done, result = ttt.is_game_over(board)
    assert not done
    assert result == 0

    board = np.array([[1,0,1],[1,-1,0],[0,0,-1]])
    done, result = ttt.is_game_over(board)
    assert not done
    assert result == 0

    board = np.array([[1,1,0],[-1,-1,1],[-1,0,0]])
    done, result = ttt.is_game_over(board)
    assert not done
    assert result == 0

    # draw

    board = np.array([[1,-1,1],[1,-1,-1],[-1,1,1]])
    done, result = ttt.is_game_over(board)
    assert done
    assert result == 0

    # full board win

    board = np.array([[1,1,1],[-1,-1,1],[-1,-1,1]])
    done, result = ttt.is_game_over(board)
    assert done
    assert result == 1

    # partial board win

    board = np.array([[-1,0,0],[1,-1,0],[0,1,-1]])
    done, result = ttt.is_game_over(board)
    assert done
    assert result == -1

def test_validate_move():

    ttt = TicTacToe()

    ttt.board = np.array([[1,0,1],[1,-1,0],[0,0,-1]])

    assert not ttt._validate_move(0)
    assert not ttt._validate_move(8)
    assert ttt._validate_move(1)
    assert ttt._validate_move(5)

def test_random_policy():

    ttt = TicTacToe()

    cnt_pos = {i:0 for i in range(9)}

    for _ in range(100):
        ttt.intermediate_board = np.array([[1,0,0],[-1,1,0],[0,-1,0]])
        ttt.board = np.array([[1,0,0],[-1,1,0],[0,-1,0]])
        ttt._adversary_move()
        x,y = np.nonzero(np.array([[1,0,0],[-1,1,0],[0,-1,0]]) != ttt.board)
        cnt_pos[3*x[0] + y[0]] += 1

    assert cnt_pos[0] == cnt_pos[3] == cnt_pos[4] == cnt_pos[7] == 0
    assert cnt_pos[1] > 0
    assert cnt_pos[2] > 0
    assert cnt_pos[5] > 0
    assert cnt_pos[6] > 0
    assert cnt_pos[8] > 0

def test_optimal_policy():

    ttt = TicTacToe(adversary_policy='optimal')

    # 1: check if it can win
    ttt.intermediate_board = np.array([[-1,0,0],[1,-1,0],[0,1,0]])
    ttt.board = np.array([[-1,0,0],[1,-1,0],[0,1,0]])
    ttt._adversary_move()
    x,y = np.nonzero(np.array([[-1,0,0],[1,-1,0],[0,1,0]]) != ttt.board)
    assert 3*x[0] + y[0] == 8

    # 2: block opponent's win
    ttt.intermediate_board = np.array([[1,0,0],[-1,1,0],[0,-1,0]])
    ttt.board = np.array([[1,0,0],[-1,1,0],[0,-1,0]])
    ttt._adversary_move()
    x,y = np.nonzero(np.array([[1,0,0],[-1,1,0],[0,-1,0]]) != ttt.board)
    assert 3*x[0] + y[0] == 8

    # 3: cause a fork
    ttt.intermediate_board = np.array([[-1,0,0],[1,1,-1],[0,0,0]])
    ttt.board = np.array([[-1,0,0],[1,1,-1],[0,0,0]])
    ttt._adversary_move()
    x,y = np.nonzero(np.array([[-1,0,0],[1,1,-1],[0,0,0]]) != ttt.board)
    assert 3*x[0] + y[0] == 2

    # 4: block opponent's fork
    ttt.intermediate_board = np.array([[1,0,0],[-1,-1,1],[0,0,0]])
    ttt.board = np.array([[1,0,0],[-1,-1,1],[0,0,0]])
    ttt._adversary_move()
    x,y = np.nonzero(np.array([[1,0,0],[-1,-1,1],[0,0,0]]) != ttt.board)
    assert 3*x[0] + y[0] == 2

    # 5: play center
    ttt.intermediate_board = np.array([[-1,0,0],[0,0,0],[0,0,1]])
    ttt.board = np.array([[-1,0,0],[0,0,0],[0,0,1]])
    ttt._adversary_move()
    x,y = np.nonzero(np.array([[-1,0,0],[0,0,0],[0,0,1]]) != ttt.board)
    assert 3*x[0] + y[0] == 4

    # 6: play opposite corner
    ttt.intermediate_board = np.array([[1,0,0],[0,-1,0],[0,0,0]])
    ttt.board = np.array([[1,0,0],[0,-1,0],[0,0,0]])
    ttt._adversary_move()
    x,y = np.nonzero(np.array([[1,0,0],[0,-1,0],[0,0,0]]) != ttt.board)
    assert 3*x[0] + y[0] == 8

    # 7: play empty corner
    ttt.intermediate_board = np.array([[-1,1,-1],[0,1,0],[0,-1,1]])
    ttt.board = np.array([[-1,1,-1],[0,1,0],[0,-1,1]])
    ttt._adversary_move()
    x,y = np.nonzero(np.array([[-1,1,-1],[0,1,0],[0,-1,1]]) != ttt.board)
    assert 3*x[0] + y[0] == 6

def test_reset():

    ttt = TicTacToe()

    # X first
    s, _ = ttt.reset(first_turn=1)
    assert (ttt.intermediate_board == ttt.board).all()
    assert s == 1

    # O first
    ttt.reset(first_turn=-1)
    assert (ttt.intermediate_board == 0).all()
    assert ttt.board.sum() == -1
    assert (ttt.board == 0).sum() == 8

    # random first
    cnt_x = 0
    cnt_o = 0
    for _ in range(50):
        ttt.reset()
        if ttt.board.sum() == -1:
            cnt_o += 1
        elif (ttt.board == 0).all():
            cnt_x += 1

    assert cnt_x + cnt_o == 50
    assert cnt_x > 0
    assert cnt_o > 0

    # exploring starts
    ttt = TicTacToe(exploring_starts=True)

    num_states = max(ttt.obs_to_state_map.values()) + 1
    num_obs = len(ttt.obs_to_state_map.values())

    state_cnt = np.zeros(num_states)
    for _ in range(10*num_obs):
        s, _ = ttt.reset()
        state_cnt[s] += 1

    print(np.nonzero(state_cnt == 0))

    assert state_cnt.mean() == 10*num_obs/num_states
    assert (state_cnt > 0).all()

def test_step():

    ttt_random = TicTacToe()
    ttt_optimal = TicTacToe(adversary_policy='optimal')

    # X wins
    ttt_optimal.intermediate_board = np.array([[1,-1,1], [-1,0,0], [1,0,0]])
    ttt_optimal.board = np.array([[1,-1,1], [-1,0,0], [1,0,-1]])
    _, reward, done, truncated, _ = ttt_optimal.step(4)
    assert reward == TicTacToe.REWARD_WIN
    assert done
    assert not truncated
    assert (ttt_optimal.intermediate_board == ttt_optimal.board).all()
    assert (ttt_optimal.board == np.array([[1,-1,1], [-1,1,0], [1,0,-1]])).all()
    terminal_board = ttt_optimal.board.copy()
    _, reward, done, truncated, _ = ttt_optimal.step(np.random.randint(0,9))
    assert reward == 0
    assert done
    assert not truncated
    assert (ttt_optimal.intermediate_board == terminal_board).all()
    assert (ttt_optimal.board == terminal_board).all()

    # O wins
    ttt_optimal.intermediate_board = np.array([[1,0,-1], [-1,0,0], [1,0,0]])
    ttt_optimal.board = np.array([[1,0,-1], [-1,-1,0], [1,0,0]])
    _, reward, done, truncated, _ = ttt_optimal.step(7)
    assert reward == TicTacToe.REWARD_LOSS
    assert done
    assert not truncated
    assert (ttt_optimal.intermediate_board == np.array([[1,0,-1], [-1,-1,0], [1,1,0]])).all()
    assert (ttt_optimal.board == np.array([[1,0,-1], [-1,-1,-1], [1,1,0]])).all()
    terminal_board = ttt_optimal.board.copy()
    _, reward, done, truncated, _ = ttt_optimal.step(np.random.randint(0,9))
    assert reward == 0
    assert done
    assert not truncated
    assert (ttt_optimal.intermediate_board == terminal_board).all()
    assert (ttt_optimal.board == terminal_board).all()

    # Draw
    ttt_optimal.intermediate_board = np.array([[1,0,-1], [-1,1,1], [0,1,-1]])
    ttt_optimal.board = np.array([[1,-1,-1], [-1,1,1], [0,1,-1]])
    _, reward, done, truncated, _ = ttt_optimal.step(6)
    assert reward == TicTacToe.REWARD_DRAW
    assert done
    assert not truncated
    assert (ttt_optimal.intermediate_board == ttt_optimal.board).all()
    assert (ttt_optimal.board == np.array([[1,-1,-1], [-1,1,1], [1,1,-1]])).all()

    ttt_optimal.intermediate_board = np.array([[1,-1,1], [0,-1,0], [0,1,-1]])
    ttt_optimal.board = np.array([[1,-1,1], [0,-1,-1], [0,1,-1]])
    _, reward, done, truncated, _ = ttt_optimal.step(3)
    assert reward == TicTacToe.REWARD_DRAW
    assert done
    assert not truncated
    assert (ttt_optimal.intermediate_board == np.array([[1,-1,1], [1,-1,-1], [0,1,-1]])).all()
    assert (ttt_optimal.board == np.array([[1,-1,1], [1,-1,-1], [-1, 1,-1]])).all()
    terminal_board = ttt_optimal.board.copy()
    _, reward, done, truncated, _ = ttt_optimal.step(np.random.randint(0,9))
    assert reward == 0
    assert done
    assert not truncated
    assert (ttt_optimal.intermediate_board == terminal_board).all()
    assert (ttt_optimal.board == terminal_board).all()

    # Invalid move
    ttt_optimal.intermediate_board = np.array([[1,-1,1], [-1,0,0], [1,0,0]])
    ttt_optimal.board = np.array([[1,-1,1], [-1,0,0], [1,0,-1]])
    _, reward, done, truncated, _ = ttt_optimal.step(6)
    assert reward == TicTacToe.REWARD_INVALID
    assert not done
    assert truncated
    assert (ttt_optimal.intermediate_board == ttt_optimal.board).all()
    assert (ttt_optimal.board == np.array([[1,-1,1], [-1,0,0], [1,0,-1]])).all()

    # Regular move
    ttt_optimal.intermediate_board = np.array([[1,0,0],[0,0,0],[0,0,0]])
    ttt_optimal.board = np.array([[1,0,-1],[0,0,0],[0,0,0]])
    _, reward, done, truncated, _ = ttt_optimal.step(4)
    assert reward == TicTacToe.REWARD_REGULAR
    assert not done
    assert not truncated
    assert (ttt_optimal.intermediate_board == np.array([[1,0,-1],[0,1,0],[0,0,0]])).all()
    assert (ttt_optimal.board == np.array([[1,0,-1],[0,1,0],[0,0,-1]])).all()

    # Random policy
    results = {ttt_random.REWARD_LOSS: 0, ttt_random.REWARD_DRAW: 0, ttt_random.REWARD_WIN: 0, ttt_random.REWARD_INVALID: 0}
    for _ in range(500):
        ttt_random.reset()
        done = False
        while not done and not truncated:
            available_x, available_y = np.nonzero(ttt_random.board == 0)
            available_moves = 3*available_x + available_y
            _, reward, done, truncated, _ = ttt_random.step(np.random.choice(available_moves))

        results[reward] += 1

    assert results[ttt_random.REWARD_INVALID] == 0
    assert results[ttt_random.REWARD_LOSS] + results[ttt_random.REWARD_DRAW] + results[ttt_random.REWARD_WIN] == 500
    assert results[ttt_random.REWARD_LOSS] > 0
    assert results[ttt_random.REWARD_DRAW] > 0
    assert results[ttt_random.REWARD_WIN] > 0

    # Optimal policy
    results = {-1: 0, 0: 0, 1: 0, 2: 0}
    for _ in range(200):
        ttt_optimal.reset()
        done = False
        while not done:
            available_x, available_y = np.nonzero(ttt_optimal.board == 0)
            available_moves = 3*available_x + available_y
            _, reward, done, _, _ = ttt_optimal.step(np.random.choice(available_moves))
        results[reward] += 1

    assert results[1] == 0
    assert results[2] == 0
    assert results[-1] + results[0] == 200

def test_after_state_map():

    ttt = TicTacToe()
    saas_map = ttt.get_state_action_to_after_state_map()

    board = np.array([[0,0,0],[-1,-1,0],[0,1,0]])
    assert ttt._encode_board(board) not in ttt.obs_to_after_state_map

    board = np.array([[1,0,0],[-1,1,0],[0,-1,0]])
    s = ttt.obs_to_state(board)
    assert saas_map[s,8] == 0
    assert saas_map[s,0] == 1
    assert saas_map[s,1] == ttt.obs_to_after_state(np.array([[1,1,0],[-1,1,0],[0,-1,0]]))

    board = np.array([[-1,0,0],[1,1,-1],[0,0,0]])
    s = ttt.obs_to_state(board)
    assert saas_map[s,0] == 1
    assert saas_map[s,1] == ttt.obs_to_after_state(np.array([[-1,1,0],[1,1,-1],[0,0,0]]))

    board = np.array([[1,0,0],[-1,-1,1],[0,0,0]])
    s = ttt.obs_to_state(board)
    assert saas_map[s,4] == 1
    assert saas_map[s,6] == ttt.obs_to_after_state(np.array([[1,0,0],[-1,-1,1],[1,0,0]]))

    board = np.array([[-1,0,0],[0,0,0],[0,0,1]])
    s = ttt.obs_to_state(board)
    assert saas_map[s,0] == 1
    assert saas_map[s,1] == ttt.obs_to_after_state(np.array([[-1,1,0],[0,0,0],[0,0,1]]))

    board = np.array([[1,-1,0],[1,-1,0],[0,0,0]])
    s = ttt.obs_to_state(board)
    assert saas_map[s,6] == 0
    assert saas_map[s,3] == 1
    assert saas_map[s,2] == ttt.obs_to_after_state(np.array([[1,-1,1],[1,-1,0],[0,0,0]]))

    board = np.zeros((3,3))
    s = ttt.obs_to_state(board)
    for a in range(9):
        next_board = board.copy()
        next_board[ttt.action_to_pos(a)] = 1
        assert saas_map[s,a] == ttt.obs_to_after_state(next_board)

    saas_ri_map = ttt.get_state_action_to_after_state_map(rotation_invariant=True)

    board_1 = np.array([[1,0,0],[-1,1,0],[0,-1,0]])
    board_2 = np.array([[0,-1,1],[-1,1,0],[0,0,0]])
    board_3 = np.array([[0,0,1],[0,1,-1],[0,-1,0]])
    s1 = ttt.obs_to_state(board_1)
    s2 = ttt.obs_to_state(board_2)
    s3 = ttt.obs_to_state(board_3)
    assert saas_ri_map[s1,8] == 0
    assert saas_ri_map[s1,0] == 1
    assert saas_ri_map[s1,1] == saas_ri_map[s2,5] == saas_ri_map[s3,1]
 
    board_1 = np.array([[-1,0,0],[1,1,-1],[0,0,0]])
    board_2 = np.array([[0,0,0],[-1,1,1],[0,0,-1]])
    board_3 = np.array([[0,-1,0],[0,1,0],[0,1,-1]])
    s1 = ttt.obs_to_state(board_1)
    s2 = ttt.obs_to_state(board_2)
    s3 = ttt.obs_to_state(board_3)
    assert saas_ri_map[s1,0] == 1
    assert saas_ri_map[s1,1] == saas_ri_map[s2,7] == saas_ri_map[s3,5]

    board_1 =np.array([[1,0,0],[-1,-1,1],[0,0,0]])
    board_2 = np.array([[0,1,0],[0,-1,0],[1,-1,0]])
    board_3 = np.array([[0,0,0],[-1,-1,1],[1,0,0]])
    s1 = ttt.obs_to_state(board_1)
    s2 = ttt.obs_to_state(board_2)
    s3 = ttt.obs_to_state(board_3)
    assert saas_ri_map[s1,4] == 1
    assert saas_ri_map[s1,6] == saas_ri_map[s2,8] == saas_ri_map[s3,0]

