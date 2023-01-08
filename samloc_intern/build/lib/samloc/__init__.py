import numpy as np
import random as rd
from numba import njit
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaExperimentalFeatureWarning, NumbaWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

NUMBER_CARDS = 52
NUMBER_PLAYERS = 4
NUMBER_ACTIONS = 122
NUMBER_ACTIONS_WITH_CARDS = 120
CARD_PER_PLAYER = 10
ENV_SIZE = 57
STATE_SIZE = 25

@njit
def initEnv():    
    env_state = np.full(ENV_SIZE, 0)
    
    # Chia bài cho 4 người chơi    
    cards = np.full(NUMBER_CARDS, 0)
    
    for idx in range(NUMBER_CARDS):
        cards[idx] = idx + 1
    np.random.shuffle(cards)
    
    env_state[0:40] = cards[0:40] % 13 + 1
    
    return env_state

@njit
def getStateSize():
    return STATE_SIZE

@njit
def getAgentState(env_state):
    p_state = np.full(STATE_SIZE, 0)
    p_id = env_state[54] % 4
    
    # Lấy 10 thẻ đang cầm trên tay
    p_state[0:10] = env_state[10*p_id:10*(p_id+1)]
    
    # Lấy bài đã đánh trên bàn
    p_state[10:20] = env_state[40:50]
    
    # Lấy trạng thái người chơi (chơi tiếp hay bỏ lượt)
    p_state[20:24] = env_state[50:54]
    
    # Lấy thông tin lượt chơi
    p_state[24] = env_state[54]
    
    return p_state

@njit
def getActionSize():
    return ACTION_SIZE

@njit
def getValidActions(player_state):
    list_action = np.full(NUMBER_ACTIONS, 0)
    list_action_with_cards = np.full(NUMBER_ACTIONS_WITH_CARDS, 0)
    player_cards = player_state[np.where(player_state[0:10] > 0)[0]]
    
    p_id = player_state[24] % 4
    board_cards = player_state[10:20].copy()
    board_cards = board_cards[np.where(board_cards > 0)[0]]
    board_cards = np.sort(board_cards)
    
    num_cards_on_board = len(board_cards)
    if 1 in board_cards:
        for card in range(num_cards_on_board):
            if card == num_cards_on_board - 1: board_cards[card]
            else: board_cards[card] = board_cards[card+1]
    
    # Người chơi luôn có thể bỏ lượt
    list_action[-1] = 1

    # Kiểm tra người chơi có bỏ lượt hay không
    if player_state[20+p_id] < 0: return list_action
    else: 
        # Kiểm tra xem có báo sâm được hay không
        # if player_state[24] == 0: list_action[0] = 1

        # Lấy danh sách các cách đánh bài có thể
        # Input: player_cards, output: list_action_with_cards
        if len(player_cards) == 1: list_action_with_cards[player_cards[0]-1] = 1
        else:
            list_action_with_cards[player_cards-1] = 1

            # Dùng cho kiểm tra dây có độ dài >=3
            action_1 = np.where(list_action_with_cards > 0)[0] + 1
            num_check_cards_1 = len(action_1) - 1
            check_cards_1 = np.full(num_check_cards_1, 0)
            for idx in range(num_check_cards_1):
                check_cards_1[idx] = action_1[idx+1] - action_1[idx]

            for idx in range(num_check_cards_1):
                # Nếu có xuất hiện dây
                if check_cards_1[idx] == 1:
                    idx_end_1 = idx
                    while check_cards_1[idx_end_1] == 1 and idx_end_1 <= num_check_cards_1:
                        idx_end_1 += 1
                        # Lấy độ dài dây
                        dist_1 = idx_end_1 - idx + 1
                        start_ = action_1[idx]

                        # Nếu độ dài dây >= 3 thì thêm action vào list_action_with_cards
                        if dist_1 >= 3:
                            val_tmp = int((dist_1-3)*(dist_1-2)/2)
                            idx_tmp = start_ + 52 + 13 * (dist_1 - 3) - val_tmp
                            list_action_with_cards[idx_tmp-1] = 1
                            
                        # Trường hợp dây chứa lá cuối là số 1
                        check_exist_card1 = False
                        if action_1[0] == 1: check_exist_card1 = True
                        if dist_1 >= 2 and check_exist_card1 and start_ + dist_1 - 1 == 13:
                            dist_1 += 1
                            val_tmp = int((dist_1-3)*(dist_1-2)/2)
                            idx_tmp = start_ + 52 + 13 * (dist_1 - 3) - val_tmp
                            list_action_with_cards[idx_tmp-1] = 1

            # Dùng cho kiểm tra bộ 2, 3, 4
            num_check_cards_0 = len(player_cards) - 1
            check_cards_0 = np.full(num_check_cards_0, 0)
            for idx in range(num_check_cards_0):
                check_cards_0[idx] = player_cards[idx+1] - player_cards[idx]

            for idx in range(13):
                check_card_idx = np.where(player_cards == idx + 1)[0]
                num_card_idx = len(check_card_idx)
                if num_card_idx >= 2:
                    while num_card_idx >= 2:
                        list_action_with_cards[13*(num_card_idx-1)+idx] = 1  
                        num_card_idx -= 1

        # So sánh với các lá đã đánh trên bàn để lấy action hợp lệ
        # Input: list_action_with_cards, output: list_action
        
        # Trên bàn có 0 lá    
        if num_cards_on_board == 0:
            for idx in range(NUMBER_ACTIONS_WITH_CARDS):
                list_action[idx+1] = list_action_with_cards[idx]
        
        # Trên bàn có 1 lá
        elif num_cards_on_board == 1:
            if board_cards[0] == 2: do_nothing = True
            elif board_cards[0] == 1: list_action[2] = list_action_with_cards[1]
            else: 
                for idx in range(13):
                    if board_cards[0] < idx + 1 or idx == 1 or idx == 0:
                        list_action[idx+1] = list_action_with_cards[idx]
        
        # Trên bàn có >= 2 lá
        elif num_cards_on_board >= 2:
            if board_cards[0] == board_cards[1]:
                if board_cards[0] == 2: do_nothing = True
                elif board_cards[0] == 1: 
                    idx_tmp = 2 + 13 * (num_cards_on_board - 1)
                    list_action[idx_tmp] = list_action_with_cards[idx_tmp-1]
                else: 
                    for idx in range(13):
                        if board_cards[0] < idx + 1 or idx == 1 or idx == 0:
                            idx_tmp = idx + 13 * (num_cards_on_board - 1)
                            list_action[idx_tmp+1] = list_action_with_cards[idx_tmp]

            elif board_cards[0] < board_cards[1]:
                for idx in range(15-num_cards_on_board):
                    if board_cards[0] <= idx:
                        val_tmp = int((num_cards_on_board-3)*(num_cards_on_board-2)/2)
                        idx_tmp = idx + 52 + 13 * (num_cards_on_board - 3) - val_tmp
                        list_action[idx_tmp+1] = list_action_with_cards[idx_tmp]
                        
        # Kiểm tra nếu len(board_cards) = 1 và board_cards[0] = 2 thì có thể đánh tứ quý
        if len(board_cards) == 1 and board_cards[0] == 2:
            for idx in range(40, 53):
                list_action[idx] = list_action_with_cards[idx-1]

    return list_action

@njit
def stepEnv(action, env):    
    p_id = env[54] % 4

    # Kiểm tra action có hợp lệ hay không
    actions = getValidActions(getAgentState(env))
    if actions[action] == 0:
        raise Exception('Action error!')
    else: env[54] += 1

    # Báo sâm
    if action == 0:
        do_nothing = True

    # Bỏ lượt
    if action == 121:
        # Kiểm tra nếu trước đó có 2 người bỏ lượt nữa thì cập nhật lại env
        num_players_aval = 0
        for player_ in range(4):
            if env[50+player_] == -1 and player_ != p_id: 
                num_players_aval += 1
                
        if num_players_aval == 2 and env[50+((p_id+1)%4)] != -1:
            # Cập nhật lại trạng thái người chơi
            env[50:54] = 0
            for player_ in range(4):
                if np.sum(env[10*player_:10*player_+10]) == 0:
                    env[50+player_] = -1
            # Cập nhật trạng thái các lá trên bàn chơi = rỗng
            env[40:50] = 0  
        else: env[50+p_id] = -1

    # # Báo sâm
    # elif action == 0:
    #     env[55] = p_id
    #     env[50:54] = -1
    #     env[50+p_id] = 0

    # Đánh 1 lá
    elif action >= 1 and action <=13:
        env[40:50] = 0
        card_remove = action
        for idx in range(10):
            if env[10*p_id+idx] == card_remove:
                env[10*p_id+idx] = 0
                break
        env[40] = action

    # Đánh 2, 3, 4 lá cùng số
    elif action >= 14 and action <= 52:
        # Cập nhật các lá vừa đánh trên bàn, xoá lá bài đã đánh trên tay người chơi
        env[40:50] = 0
        card_remove = action % 13
        if card_remove == 0: card_remove = 13
        for idx in range(1+int((action-1)/13)):
            for card_ in range(10):
                if env[10*p_id+card_] == card_remove:
                    env[10*p_id+card_] = 0
                    break
            env[40+idx] = card_remove

    # Đánh dây 3, 4, 5, 6, 7, 8, 9, 10 lá
    elif action >= 53 and action <= 120:
        # Xác định số lá sẽ đánh
        num_cards_remove = 0
        if action >= 53 and action <= 64:
            num_cards_remove = 3
        elif action >= 65 and action <= 75:
            num_cards_remove = 4
        elif action >= 76 and action <= 85:
            num_cards_remove = 5
        elif action >= 86 and action <= 94:
            num_cards_remove = 6
        elif action >= 95 and action <= 102:
            num_cards_remove = 7
        elif action >= 103 and action <= 109:
            num_cards_remove = 8
        elif action >= 110 and action <= 115:
            num_cards_remove = 9
        elif action >= 116 and action <= 120:
            num_cards_remove = 10
            
        # Xác định lá đầu tiên trong dây
        card_remove_start = action - 130 + int((num_cards_remove - 15) * (num_cards_remove - 16) / 2)       

        # Cập nhật các lá vừa đánh trên bàn, xoá lá bài đã đánh trên tay người chơi
        env[40:50] = 0
        for idx in range(num_cards_remove):
            for card_ in range(10):
                if card_remove_start + idx == 14:
                    env[10*p_id+1] = 0
                    break
                elif env[10*p_id+card_] == card_remove_start + idx:
                    env[10*p_id+card_] = 0
                    break
            if card_remove_start + idx == 14:
                env[40+idx] = 1
            else:
                env[40+idx] = card_remove_start + idx 

@njit
def getAgentSize():
    return NUMBER_PLAYERS

@njit
def checkEnded(env):
    if env[54] == 0: p_id = 0
    else: p_id = (env[54] - 1) % 4
    
    board_cards = env[40:50]
    board_cards = board_cards[np.where(board_cards > 0)[0]]
    board_cards = np.sort(board_cards)
    
    # Kiểm tra xem còn ai đang đánh
    num_players_aval = 0
    for player_ in range(4):
        if env[50+player_] == -1 and player_ != p_id: 
            num_players_aval += 1
            
    # Trường hợp người chơi đã đánh hết bài
    if np.sum(env[10*p_id:10*p_id+10]) == 0 and len(board_cards) > 0:        
        # Nếu vừa đánh hết bài xong
        if env[50+p_id] == 0:
            # Nếu thối 2
            if board_cards[0] == 2 and len(board_cards) == 1:
                if num_players_aval == 3:
                    return 5
                else:
                    env[40:50] = 0
                    env[50+p_id] = -1
                    return -1

            # Nếu thối tứ quý
            elif len(board_cards) == 4 and board_cards[0] == board_cards[1] and board_cards[0] != 0:
                if num_players_aval == 3:
                    return 5
                else:
                    env[40:50] = 0
                    env[50+p_id] = -1
                    return -1

        # Nếu đã thối và ván đấu đang tiếp tục
        elif env[50+p_id] == -1:
            return -1
        
        env[56] = 1
        return p_id + 1
            
        
    # Nếu người chơi chưa đánh hết bài
    else:
        return -1

def bot_random(p_state):
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx]

@njit
def numba_bot_random(p_state):
    arr_action = getValidActions(p_state)
    arr_action = np.where(arr_action == 1)[0]
    act_idx = np.random.randint(0, len(arr_action))
    return arr_action[act_idx]

def run_one_game(list_Agent):
    if len(list_Agent) != 4:
        raise Exception('Agent error')
    
    env = np.full(57, 0)
    env = initEnv()

    _cc = 0
    while _cc <= 10000:
        p_idx = env[54] % 4
        p_state = getAgentState(env)

        action = list_Agent[p_idx](p_state)
        list_action = getValidActions(p_state)
        if list_action[action] != 1:
            raise Exception('Action error')

        stepEnv(action, env)

        if checkEnded(env) != -1:
            break

        _cc += 1

    return checkEnded(env)

@njit
def numba_run_one_game(p0, list_other, print_mode = False):
    env = np.full(57, 0)
    env = initEnv()
    
    def _print_():
        print('----------------------------------------------------------------------------------')
        print('Lượt của người chơi:', env[54] % 4 + 1)
        print('Các cách đánh bài:', np.where(getValidActions(getAgentState(env))>0)[0])
        
        board_cards = env[40:50]
        board_cards = board_cards[np.where(board_cards > 0)[0]]
        board_cards = np.sort(board_cards)
        print('Các lá đã đánh trên bàn:', board_cards, 'Turn:', env[54])

        print('P1:', np.sort(env[0:10]))
        print('P2:', np.sort(env[10:20]))
        print('P3:', np.sort(env[20:30]))
        print('P4:', np.sort(env[30:40]))
        print('Người chơi bỏ lượt:', np.sort(env[50:54]))
        print('-------')

    _cc = 0

    while _cc <= 10000:
        p_id = env[54] % 4
        player_state = getAgentState(env)

        if list_other[p_id] == -1:
            action = p0(player_state)
        else:
            action = numba_bot_random(player_state)

        list_action = getValidActions(player_state)
        if list_action[action] != 1:
            raise Exception('Action error!')

        if print_mode: _print_()

        stepEnv(action, env)

        if checkEnded(env) != -1:
            break

        _cc += 1 

    for idx in range(4):
        if list_other[idx] == -1 and print_mode:
            print('________Đã kết thúc game__________')
            _print_()
            if checkEnded(env) == 5:
                print('Ván chơi hoà!')
            elif np.where(list_other == -1)[0] ==  (checkEnded(env) - 1):
                print('Xin chúc mừng bạn là người chiến thắng')
            else:
                print('Người chơi đã thua, chúc bạn may mắn lần sau')

    winner = 0
    if checkEnded(env) == 5: winner = -1
    elif np.where(list_other == -1)[0] == (checkEnded(env) - 1): winner = 1
    else: winner = 0
    
    return winner

def run_game(list_Agent, num_game):
    if len(list_Agent) != 4:
        raise Exception('Agent error')

    win = np.full(4, 0)
    list_idx = np.array([0, 1, 2, 3])

    for _n in range(num_game):
        # if print_mode == False and num_game > 1:
        #     progress_bar(_n, num_game)
        np.random.shuffle(list_idx)
        winner  = run_one_game([list_Agent[list_idx[0]], list_Agent[list_idx[1]], list_Agent[list_idx[2]], list_Agent[list_idx[3]]])
        if winner != 5: win[list_idx[winner-1]] += 1
    return win

@njit
def numba_run_n_game(p0, num_game, print_mode = False):
    win = 0
    for _n in range(num_game):
        # if print_mode == False and num_game > 1:
        #     progress_bar(_n, num_game)
        list_other = np.append(np.random.choice(np.arange(3), 3, replace = False), -1)
        np.random.shuffle(list_other)
        winner  = numba_run_one_game(p0, list_other, print_mode)
        if winner != -1: win += winner
    print()
    return win

@njit
def numba_run_main(p0, num_game, print_mode = False):
    return numba_run_n_game(p0, num_game, print_mode)

@njit
def visualize_env(env):
    dict_env = {}
    dict_env['Lượt đánh của người chơi'] = np.asarray([env[54] % 4 + 1])
    dict_env['Các cách đánh bài trên tay'] = np.asarray(np.where(getValidActions(getAgentState(env))>0)[0])
    dict_env['Người chơi 1'] = np.sort(env[0:10])
    dict_env['Người chơi 2'] = np.sort(env[10:20])
    dict_env['Người chơi 3'] = np.sort(env[20:30])
    dict_env['Người chơi 4'] = np.sort(env[30:40])
    
    board_cards = env[40:50]
    board_cards = board_cards[np.where(board_cards > 0)[0]]
    board_cards = np.sort(board_cards)
    dict_env['Các lá đã đánh'] = board_cards
    
    dict_env['Những người bỏ lượt'] = env[50:54]
    dict_env['Ai báo sâm?'] = np.asarray([env[55]])
    dict_env['Tiếp tục chơi?'] = np.asarray([env[56]])
    
    for key_ in dict_env.keys():
        print(key_, ':', dict_env[key_])