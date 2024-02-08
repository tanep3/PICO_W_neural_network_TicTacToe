import my_numpy as np
import random
import machine
import _thread
import gc
gc.enable()

LED = machine.Pin("LED", machine.Pin.OUT)
PLAYER_O = 1
PLAYER_X = -1
EMPTY = 0
GAME_DRAW = 0
GAME_CONTINUE = 2
learning_rate = 0.1
game_is_start = False
your_piece = PLAYER_O
ai_piece = PLAYER_X
#panel index
PANEL_INDEX = (
    "00","01","02","10","11","12","20","21","22",
)
OPTION_SELECTED = PLAYER_O

# マルバツゲームの状態を表すクラス
class TicTacToe:
    def __init__(self):
        self.__board = np.zeros((3, 3)) # 3x3の盤面

    def reset(self):
        self.__board = np.zeros((3, 3)) # 3x3の盤面

    @property
    def board(self):
        return np.copy(self.__board)
    
    def put_piece(self, row, col, player):
        self.__board[row][col] = player
    
    def print_board(self):
        for n in self.__board:
            print(n)

# board操作用の補助関数
# マス目上に駒が２つ並んでいるかどうかのチェック
def is_two_piece(board):
    global PLAYER_O, PLAYER_X, EMPTY
    count_player1 = 0
    count_player2 = 0
    for i in range(3):
        #横
        count_player1 += 1 if np.count_nonzero(board[i]) and np.count_num(board[i], PLAYER_O) == 2 else 0
        count_player2 += 1 if np.count_nonzero(board[i]) and np.count_num(board[i], PLAYER_X) == 2 else 0
        #縦
        count_player1 += 1 if np.count_nonzero(np.T(board)[i]) and np.count_num(np.T(board)[i], PLAYER_O) == 2 else 0
        count_player2 += 1 if np.count_nonzero(np.T(board)[i]) and np.count_num(np.T(board)[i], PLAYER_X) == 2 else 0
        #斜め
        count_player1 += 1 if np.count_nonzero(np.diag(board)) and np.count_num(np.diag(board), PLAYER_O) == 2 else 0
        count_player2 += 1 if np.count_nonzero(np.diag(board)) and np.count_num(np.diag(board), PLAYER_X) == 2 else 0
        count_player1 += 1 if np.count_nonzero(np.diag_fliplr(board)) and np.count_num(np.diag_fliplr(board), PLAYER_O) == 2 else 0
        count_player2 += 1 if np.count_nonzero(np.diag_fliplr(board)) and np.count_num(np.diag_fliplr(board), PLAYER_X) == 2 else 0
    return count_player1 - count_player2

#中央に自分の駒があるかどうかのチェック
def is_center(board, player):
    return board[1][1] == player

# 勝利条件のチェック（横、縦、斜め）
def is_winner(board, player):
    res1 = False
    res2 = False
    res3 = False
    res4 = False
    for i in range(3):
        if np.count_num(board[i], player) == 3:
            res1 = True
            break
        if np.count_num(np.T(board)[i], player) == 3:
            res2 = True
            break
    if np.count_num(np.diag(board), player) == 3:
        res3 = True
    if np.count_num(np.diag_fliplr(board), player) == 3:
        res4 = True
    return res1 or res2 or res3 or res4

# 盤面がすべて埋まっているかどうかのチェック
def is_board_full(board):
    global EMPTY
    return np.count_num(board, EMPTY) == 0

# シンプルなニューラルネットワークのクラス
class NeuralNetwork:
    def __init__(self):
        np.random_seed()
        self.w1 = np.random_list(9, 18)
        self.b1 = np.zeros(18) 
        self.w2 = np.random_list(18, 18) 
        self.b2 = np.zeros(18)
        self.w3 = np.random_list(18, 9)
        self.b3 = np.zeros(9)
        # 保存された学習データを読み込む
        self.data_read()
        self.lock = _thread.allocate_lock()
        self.saving_count = 0

        # ゲームの進行状況を記録する配列
        self.stock_of_procedures = []
    
    def data_write(self):
        # 10回に1回バックアップ
        if self.saving_count < 10:
            self.saving_count += 1
            return
        self.saving_count = 0
        print("今書いてます")
        datas = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        try:
            with self.lock:
                fw = open("ai.bin", "wb")
                fw.write(repr(datas))
                fw.close()
        except OSError:
            print("Error: Can't write file")
        print("書き終わり")
            
    def data_read(self):
        try:
            with open("ai.bin", "rb") as fr:
                datas = eval(fr.read())
        except OSError:
            print("Error: Can't read file")
            return
        self.w1 = datas[0]
        self.b1 = datas[1]
        self.w2 = datas[2]
        self.b2 = datas[3]
        self.w3 = datas[4]
        self.b3 = datas[5]

# 活性化関数(sigmoid関数) 
def sigmoid(x):
    # オーバーフローを避けるために、入力を制限
    # x = np.clip(x, -1.0, 1.0)
    return np.reversal_of_numerator(np.add(np.exp(np.multiple(x,-1.0)), 1.0))
# シグモイド関数の導関数
def sigmoid_derivative(x):
    return np.multiple(sigmoid(x), np.sub(1, sigmoid(x)))
    
# 正規化
def normalize(x):
    gc.collect()
    x_min = np.my_min(x)
    x_max = np.my_max(x)
    if x_min == 0 and x_max == 0:
        return x
    if x_min == x_max:
        return np.divide(x, x_max)
    normalized_x = np.sub(np.divide(x,(x_max - x_min)), x_min/(x_max - x_min))
    return normalized_x

# モデル定義
def model(neural_network, board):
    z1 = normalize(np.add(np.dot(np.flatten(board), neural_network.w1),neural_network.b1))
    a1 = sigmoid(z1)
    z2 = normalize(np.add(np.dot(a1, neural_network.w2), neural_network.b2))  
    a2 = sigmoid(z2)
    z3 = normalize(np.add(np.dot(a2, neural_network.w3), neural_network.b3))
    y = sigmoid(z3)
    gc.collect()
    return y

# 損失関数
def loss_fn(y_true, y_pred):
    loss = 0
    return loss

# 誤差逆伝播
def backprop(neural_network, board, y_true, learning_rate=0.01):
    # 順伝播の計算
    z1 = np.add(np.dot(np.flatten(board), neural_network.w1), neural_network.b1)  
    a1 = sigmoid(z1)  
    z2 = np.add(np.dot(a1, neural_network.w2), neural_network.b2)   
    a2 = sigmoid(z2)
    z3 = np.add(np.dot(a2, neural_network.w3), neural_network.b3)
    y_pred = sigmoid(z3) 
    # 損失の計算
    loss = loss_fn(y_true, y_pred)
    # 出力層のデルタの計算
    delta3 = np.sub(y_pred, y_true)
    # 第2隠れ層のデルタ計算 
    delta2 = np.multiple(np.dot(delta3, np.T(neural_network.w3)), sigmoid_derivative(a2))
    # 第1隠れ層のデルタ計算
    delta1 = np.multiple(np.dot(delta2, np.T(neural_network.w2)), sigmoid_derivative(a1))
    # 勾配計算
    grad_w3 = normalize(np.outer(a2, delta3)) 
    grad_b3 = np.mean(delta3, axis=None)
    gw2 = np.outer(a1, delta2)
    grad_w2 = normalize(gw2) 
    grad_b2 = np.mean(delta2, axis=None)
    grad_w1 = normalize(np.outer(np.flatten(board), delta1)) 
    grad_b1 = np.mean(delta1, axis=None)
    gc.collect()
    # 重み更新
    neural_network.w1 = np.sub(neural_network.w1, np.multiple(learning_rate, grad_w1))
    neural_network.b1 = np.sub(neural_network.b1, np.multiple(learning_rate, grad_b1))   
    neural_network.w2 = np.sub(neural_network.w2, np.multiple(learning_rate, grad_w2))
    gc.collect()
    neural_network.b2 = np.sub(neural_network.b2, np.multiple(learning_rate, grad_b2))
    neural_network.w3 = np.sub(neural_network.w3, np.multiple(learning_rate, grad_w3))
    neural_network.b3 = np.sub(neural_network.b3, np.multiple(learning_rate, grad_b3))
    neural_network.w1 = normalize(neural_network.w1)
    neural_network.w2 = normalize(neural_network.w2)
    neural_network.w3 = normalize(neural_network.w3)
    return loss

# 学習結果を元に、最善の一手を返す
def best_action(neural_network, state, player):
    # 簡単な予測ロジック（仮）
    state_copy = np.copy(state)
    if player == PLAYER_X:
        player = PLAYER_O
        state_copy = np.multiple(state_copy, -1)
    putable_matrix = [idx for idx, value in enumerate(np.flatten(state)) if value == 0]
    if len(putable_matrix) > 0:
        predict_matrix = []
        get_model = model(neural_network, state_copy)
        for idx in putable_matrix:
            row, col = divmod(idx, 3)
            predict_matrix.append(get_model[idx])
        best_index = np.argmax(predict_matrix)
        row, col = divmod(putable_matrix[best_index], 3) 
        return (row, col)
    else:
        # すべてのセルが埋まっている場合の処理
        return 0, 0

# 機械学習を実行
def train(neural_network, player, row, col, tic_tac_toe, learning_rate=0.01):
    board_state = tic_tac_toe.board
    target = np.zeros(9)
    neural_network.stock_of_procedures.append(((row, col), player))
    reward = calculate_reward(player, tic_tac_toe)
    if reward is None:
        # ゲーム進行中はrewordはNone
        return 0
    for i, place in enumerate(neural_network.stock_of_procedures):
        if is_winner(board_state, PLAYER_O):
            if place[1] == PLAYER_O:
                target[place[0][0] * 3 + place[0][1]] = reward
            elif place[1] == PLAYER_X:
                target[place[0][0] * 3 + place[0][1]] = -reward
        elif is_winner(board_state, PLAYER_X):
            if place[1] == PLAYER_X:
                target[place[0][0] * 3 + place[0][1]] = reward
            elif place[1] == PLAYER_O:
                target[place[0][0] * 3 + place[0][1]] = -reward
        elif is_board_full(board_state):
            if place[1] == PLAYER_O:
                target[place[0][0] * 3 + place[0][1]] = reward
            elif place[1] == PLAYER_X:
                target[place[0][0] * 3 + place[0][1]] = -reward

    # 損失計算と学習
    loss = backprop(neural_network, board_state, target, learning_rate)
    # print(loss)
    neural_network.stock_of_procedures = []
    gc.collect()
    return loss

# 機械学習のため、成功報酬を算出する。
def calculate_reward(player, tic_tac_toe):
    global PLAYER_O, PLAYER_X
    board = tic_tac_toe.board
    if is_winner(board, PLAYER_O):
        reward = 1
        reward += is_two_piece(board)/5.0
        reward += 0.3 if is_center(board, PLAYER_O) else 0
        reward -= 0.1 if is_center(board, PLAYER_X) else 0
    elif is_winner(board,PLAYER_X):
        reward = 0.5
        reward += is_two_piece(board)/5.0
        reward += 0.3 if is_center(board, PLAYER_X) else 0
        reward -= 0.1 if is_center(board, PLAYER_O) else 0
    elif is_board_full(board):
        reward = -0.5
    else:
        return None
    return reward

def play_Human(player, tic_tac_toe, _):
    # プレイヤーの手を入力
    state = tic_tac_toe.board
    while True:
        try:
            str_row = int(input(f"縦：{player}の手を入力して下さい: "))
            if 0 <= str_row and str_row <= 2:
                break
        except ValueError:
            pass
    while True:
        try:
            str_col = int(input(f"横：{player}の手を入力して下さい: "))
            if 0 <= str_col and str_col <= 2:
                break
        except ValueError:
            pass
    return int(str_row), int(str_col)

def play_Random(player, tic_tac_toe, _):
    # プレイヤーの手をランダムに選択
    state = tic_tac_toe.board
    putable_indices = [idx for idx, value in enumerate(np.flatten(state)) if value == 0]
    action = random.choice(putable_indices)
    row, col = divmod(action, 3)
    return row, col

def play_AI(player, tic_tac_toe, neural_network):
    # プレイヤーの手をAIに選択
    state = tic_tac_toe.board
    row, col = best_action(neural_network, state, player)
    gc.collect()
    print("free memory:", gc.mem_free())
    return row, col

# ランダムとAIの対戦
def play_game(tic_tac_toe, func_player1, func_player2, neural_network, print_state=False, learning_rate=0.01):
    global PLAYER_O, PLAYER_X
    tic_tac_toe.reset()
    board = tic_tac_toe.board
    while not is_board_full(board):
        if print_state:
            tic_tac_toe.print_board()
        row, col = func_player1(PLAYER_O, tic_tac_toe, neural_network)
        # プレイヤーの手を盤面に反映
        tic_tac_toe.put_piece(row, col, PLAYER_O)
        loss = train(neural_network, PLAYER_O, row, col, tic_tac_toe, learning_rate)
        board = tic_tac_toe.board
        # if print_state:
        #     print(loss)
        if is_winner(board, PLAYER_O):
            if print_state:
                tic_tac_toe.print_board()
                print("Player 1 won!")
            return
        if is_board_full(board):
            if print_state:
                tic_tac_toe.print_board()
                print("It's a draw!")
            return
        # プレイヤー2の手を予測
        if print_state:
            tic_tac_toe.print_board()
        row, col = func_player2(PLAYER_X, tic_tac_toe, neural_network)
        # プレイヤーの手を盤面に反映
        tic_tac_toe.put_piece(row, col, PLAYER_X)
        loss = train(neural_network,PLAYER_X, row, col, tic_tac_toe, learning_rate)
        board = tic_tac_toe.board
        # if print_state:
        #     print(loss)
        if is_winner(board, PLAYER_X):
            if print_state:
                tic_tac_toe.print_board()
                print("Player 2 won!")
            return
        if is_board_full(board):
            if print_state:
                tic_tac_toe.print_board()
                print("It's a draw!")
            return

# オブジェクトのインスタンス化
BOARD = TicTacToe()
BRAIN = NeuralNetwork()

# 事前学習
def study():
    # 学習の実行
    study_count = 10
    learning_rate = 0.1
    for i in range(study_count):
        LED.off()
        play_game(BOARD, play_Random, play_Random, BRAIN, False, learning_rate)
        LED.on()
        play_game(BOARD, play_Random, play_AI, BRAIN, False, learning_rate)
        LED.off()
        play_game(BOARD, play_AI, play_Random, BRAIN, False, learning_rate)
        LED.on()
        play_game(BOARD, play_AI, play_AI, BRAIN, False, learning_rate)
    LED.off()

# ゲームを１ターン進める関数です。
def game_one_turn(player, row, col):
    global learning_rate, BOARD, BRAIN, game_is_start, GAME_DRAW, GAME_CONTINUE
    if is_board_full(BOARD.board):
        game_is_start = False
        return GAME_DRAW
    # プレイヤーの手を盤面に反映
    BOARD.put_piece(row, col, player)
    loss = train(BRAIN, player, row, col, BOARD, learning_rate)
    board = BOARD.board
    if is_winner(board, player):
        return player
    if is_board_full(board):
        return  GAME_DRAW
    return GAME_CONTINUE

# 画面のスタートボタンが押されたらここが実行されます。
def game_start(page_data, posted_data):
    global BOARD, BRAIN, player1, player2, game_is_start, OPTION_SELECTED, your_piece, ai_piece
    if game_is_start:
        return make_your_turn_page(page_data)
    BOARD.reset()
    game_is_start = True
    if posted_data['turn'] == 'O':
        print("You are O")
        your_piece = PLAYER_O
        ai_piece = PLAYER_X
        OPTION_SELECTED = PLAYER_O
    else:
        print("You are X")
        your_piece = PLAYER_X
        ai_piece = PLAYER_O
        row, col = play_AI(ai_piece, BOARD, BRAIN)
        result = game_one_turn(ai_piece, row, col)
        OPTION_SELECTED = PLAYER_X
    return make_your_turn_page(page_data)

# 画面上のパネルをクリックしたらここが実行されます。
def put_piece(page_data, posted_data):
    global game_is_start, your_piece, ai_piece, GAME_CONTINUE, BOARD, BRAIN
    if not game_is_start:
        return make_your_turn_page(page_data)
    # あなたの手を盤面に反映
    panel = list(posted_data.keys())[0]
    if posted_data[panel] != "":
        return make_your_turn_page(page_data)
    penel = panel
    if posted_data[panel] != '':
        return make_your_turn_page(page_data)
    row = int(penel[0])
    col = int(penel[1])
    result = game_one_turn(your_piece, row, col)
    # ゲーム・オーバーの処理
    if result != GAME_CONTINUE:
        game_is_start = False
        # AIの学習データをファイルに保存
        _thread.start_new_thread(BRAIN.data_write, ())
        return make_end_page(page_data, result)
    # AIの手を盤面に反映
    LED.on()
    row, col = play_AI(ai_piece, BOARD, BRAIN)
    result = game_one_turn(ai_piece, row, col)
    LED.off()
    # ゲーム・オーバーの処理
    if result != GAME_CONTINUE:
        game_is_start = False
        # AIの学習データをファイルに保存
        _thread.start_new_thread(BRAIN.data_write, ())
        return make_end_page(page_data, result)
    return make_your_turn_page(page_data)

# 画面をGETメソッドにより開いたときに、画面を初期化する関数です。
def init_index_page(page_data):
    p = page_data
    p = p.replace('{selected_O}', 'selected')
    p = p.replace('{selected_X}', '')
    p = p.replace('{disabled}', '')
    p = p.replace('{message}', 'STARTを押して下さい。')
    for n in PANEL_INDEX:
        p = p.replace(f'{{{n}}}', '""')
    return p

# 進行中のゲーム画面を作成する関数です。
def make_your_turn_page(page_data):
    global BOARD, PANEL_INDEX, PLAYER_O, PLAYER_X, game_is_start, OPTION_SELECTED
    board = BOARD.board
    p = page_data
    if OPTION_SELECTED == PLAYER_O:
        p = p.replace('{selected_O}', 'selected')
        p = p.replace('{selected_X}', '')
    else:
        p = p.replace('{selected_O}', '')
        p = p.replace('{selected_X}', 'selected')
    if not game_is_start:
        # STARTボタンを活性化する
        p = p.replace('{disabled}', '')
        p = p.replace('{message}', 'STARTを押して下さい。')
    # STARTボタンを非活性にする
    p = p.replace('{disabled}', 'disabled')
    p = p.replace('{message}', 'あなたの番です。<br>パネルをクリックして下さい。')
    for n in PANEL_INDEX:
        row = int(n[0])
        col = int(n[1])
        if board[row][col] == PLAYER_O:
            p = p.replace(f'{{{n}}}', 'O')
        elif board[row][col] == PLAYER_X:
            p = p.replace(f'{{{n}}}', 'X')
        else:
            p = p.replace(f'{{{n}}}', '')
    return p

# 終了画面を作成する関数です。
def make_end_page(page_data, result):
    global BOARD, PANEL_INDEX, PLAYER_O, PLAYER_X, GAME_DRAW, OPTION_SELECTED, BRAIN
    board = BOARD.board
    p = page_data
    if OPTION_SELECTED == PLAYER_O:
        p = p.replace('{selected_O}', 'selected')
        p = p.replace('{selected_X}', '')
    else:
        p = p.replace('{selected_O}', '')
        p = p.replace('{selected_X}', 'selected')
    # STARTボタンを活性化する
    p = p.replace('{disabled}', '')
    # 結果の編集
    if BRAIN.saving_count == 0:
        save_message = '<br>学習データを保存しました。'
    else:
        save_message = ''
    if result == PLAYER_O:
        p = p.replace('{message}', 'Oの勝ちです！' + save_message)
    elif result == PLAYER_X:
        p = p.replace('{message}', 'Xの勝ちです！' + save_message)
    else:
        p = p.replace('{message}', '引き分けです！' + save_message)
    # 盤面の編集
    for n in PANEL_INDEX:
        row = int(n[0])
        col = int(n[1])
        if board[row][col] == PLAYER_O:
            p = p.replace(f'{{{n}}}', 'O')
        elif board[row][col] == PLAYER_X:
            p = p.replace(f'{{{n}}}', 'X')
        else:
            p = p.replace(f'{{{n}}}', '')
    return p
