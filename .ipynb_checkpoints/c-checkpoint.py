from sub import checkBuyCard, openCard

import numpy as np
from numba import njit
from numba.typed import List

__NORMAL_CARD__ = np.load("SysData/normalCard.npy", allow_pickle=True)
__NOBLE_CARD__ = np.load("SysData/nobleCard.npy", allow_pickle=True)

__ENV_SIZE__ = 94
__STATE_SIZE__ = 275
__ACTION_SIZE__ = 41
__AGENT_SIZE__ = 4


@njit
def initEnv():
    lv1 = np.arange(41)
    lv2 = np.arange(40, 71)
    lv3 = np.arange(70, 91)

    np.random.shuffle(lv1[:-1])
    np.random.shuffle(lv2[:-1])
    np.random.shuffle(lv3[:-1])

    lv1[-1] = 4
    lv2[-1] = 4
    lv3[-1] = 4

    env = np.full(__ENV_SIZE__, 0)

    env[0:6] = np.array([7,7,7,7,7,5])

    noble_ = np.arange(10)
    np.random.shuffle(noble_)
    env[6:11] = noble_[:5]

    env[11:15] = lv1[:4]
    env[15:19] = lv2[:4]
    env[19:23] = lv3[:4]

    # 23:38:53:68:83
    for pIdx in range(4):
        temp_ = 15*pIdx
        # env[23+temp_:35+temp_] = 0
        env[35+temp_:38+temp_] = -1

    # env[83] == 0 # Turn
    # env[84:89] = 0 # Dùng khi lấy nguyên liệu
    # env[89:93] = 0 # Số thẻ đã mua
    # env[93] = 0 # 1 khi game kết thúc

    return env, lv1, lv2, lv3


@njit
def getStateSize():
    return __STATE_SIZE__


@njit
def getAgentState(env, lv1, lv2, lv3):
    state = np.zeros(__STATE_SIZE__)

    state[0:6] = env[0:6]

    # 6:12:18:24:30:36 # Thẻ Noble
    for i in range(5):
        nobleId = env[6+i]
        if nobleId != -1:
            temp_ = 6*i
            state[6+temp_:12+temp_] = __NOBLE_CARD__[nobleId]

    # 36:47:58:69:80:91:102:113:124:135:146:157:168 # Thẻ normal
    for i in range(12):
        cardId = env[11+i]
        if cardId != -1:
            cardIn4 = __NORMAL_CARD__[cardId]
            temp_ = 11*i
            state[36+temp_] = cardIn4[0] # Điểm
            state[37+temp_+cardIn4[1]] = 1 # Bonus gem
            state[42+temp_:47+temp_] = cardIn4[2:7]

    pIdx = env[83] % 4
    for i in range(4):
        pEnvIdx = (pIdx + i) % 4
        temp1 = 12*i
        temp2 = 15*pEnvIdx

        # 201:213:225:237:249 # Player infor
        state[201+temp1:213+temp1] = env[23+temp2:35+temp2]

        if i == 0:
            # 168:179:190:201 # Thẻ úp
            for j in range(3):
                cardId = env[35+temp2+j]
                if cardId != -1:
                    cardIn4 = __NORMAL_CARD__[cardId]
                    temp_ = 11*j
                    state[168+temp_] = cardIn4[0]
                    state[169+temp_+cardIn4[1]] = 1
                    state[174+temp_:179+temp_] = cardIn4[2:7]

        else:
            # 249:252:255:258 # Đếm cấp thẻ úp
            temp_ = 3*(i-1)
            for j in range(3):
                cardId = env[35+temp2+j]
                if cardId != -1:
                    if cardId < 40:
                        state[249+temp_] += 1
                    elif cardId < 70:
                        state[250+temp_] += 1
                    else:
                        state[251+temp_] += 1

        # [258:262] # Số thẻ đã mua
        state[258+i] = env[89+pEnvIdx]

    # [262:266] # Vị trí của người chơi
    state[262+pIdx] = 1

    # [266:271] # Nguyên liệu đã lấy
    state[266:271] = env[84:89]

    # [271]
    state[271] = env[93]

    if lv1[-1] < 40: # Còn thẻ ẩn cấp 1
        state[272] = 1
    if lv2[-1] < 30: # Còn thẻ ẩn cấp 2
        state[273] = 1
    if lv3[-1] < 20: # Còn thẻ ẩn cấp 3
        state[274] = 1

    return state


@njit
def getActionSize():
    return __ACTION_SIZE__


@njit
def getValidActions(state):
    '''
    [0:5] Lấy nguyên liệu
    [5:20] Các action mua thẻ ([5:17] thẻ trên bàn, [17:20] thẻ úp)
    [20:35] Các action úp thẻ
    [35:40] Các action trả nguyên liệu
    [40]: Bỏ qua lượt (trường hợp hiếm rất đặc biệt)
    '''

    validActions = np.full(__ACTION_SIZE__, 0)
    boardStocks = state[0:6]

    takenStocks = state[266:271]
    if (takenStocks > 0).any(): # Đang lấy nguyên liệu
        temp_ = np.where(boardStocks[0:5]>0)[0]
        validActions[temp_] = 1

        s_ = np.sum(takenStocks)
        if s_ == 1:
            t_ = np.where(takenStocks==1)[0][0]
            if boardStocks[t_] < 3:
                validActions[t_] = 0
        else:
            t_ = np.where(takenStocks==1)[0]
            validActions[t_] = 0

        return validActions

    if np.sum(state[201:207]) > 10: # Thừa nguyên liệu, cần trả nguyên liệu
        temp_ = np.where(state[201:206]>0)[0] + 35
        validActions[temp_] = 1
        return validActions

    # Lấy nguyên liệu
    temp_ = np.where(boardStocks[0:5]>0)[0]
    validActions[temp_] = 1

    checkReserveCard = False
    for i in range(3):
        temp_ = 11*i
        if (state[174+temp_:179+temp_]==0).all():
            checkReserveCard = True

    # Các action mua thẻ (và úp thẻ)
    for i in range(15):
        temp_ = 11*i
        cardPrice = state[42+temp_:47+temp_]
        if (cardPrice > 0).any():
            if checkReserveCard and i < 12:
                validActions[20+i] = 1

            if checkBuyCard(state[201:207], state[207:212], cardPrice):
                validActions[5+i] = 1

    # Check úp thẻ ẩn
    for i in range(3):
        if checkReserveCard and state[272+i] == 1:
            validActions[32+i] = 1

    # Check nếu không có action nào có thể thực hiện (bị kẹt) thì cho action bỏ lượt
    if (validActions > 0).any():
        return validActions

    validActions[40] = 1
    return validActions


@njit
def stepEnv(action, env, lv1, lv2, lv3):
    pIdx = env[83] % 4
    temp_ = 15*pIdx
    pStocks = env[23+temp_:29+temp_]
    bStocks = env[0:6]

    # Lấy nguyên liệu
    if action < 5:
        takenStocks = env[84:89]
        takenStocks[action] += 1
        pStocks[action] += 1
        bStocks[action] -= 1

        check_ = False
        s_ = np.sum(takenStocks)
        if s_ == 1:
            if bStocks[action] < 3 and (np.sum(bStocks[0:5]) - bStocks[action]) == 0:
                check_ = True
        elif s_ == 2:
            if np.max(takenStocks) == 2 or (np.sum(bStocks[0:5]) - np.sum(bStocks[np.where(takenStocks==1)[0]])) == 0:
                check_ = True
        else:
            check_ = True

        if check_:
            takenStocks[:] = 0

            # Nếu không thừa nguyên liệu thì next turn
            if np.sum(pStocks) <= 10:
                env[83] += 1

    # Trả nguyên liệu
    elif action >= 35 and action < 40:
        gem = action - 35
        pStocks[gem] -= 1
        bStocks[gem] += 1

        # Nếu không thừa nguyên liệu thì next turn
        if np.sum(pStocks) <= 10:
            env[83] += 1

    # Úp thẻ
    elif action >= 20 and action < 35:
        temp_hideCard = 35 + temp_
        posP = np.where(env[temp_hideCard:temp_hideCard+3]==-1)[0][0] + temp_hideCard

        if bStocks[5] > 0:
            pStocks[5] += 1
            bStocks[5] -= 1

        if action == 32: # Úp thẻ ẩn cấp 1
            env[posP] = lv1[lv1[-1]]
            lv1[-1] += 1
        elif action == 33: # Úp thẻ ẩn cấp 2
            env[posP] = lv2[lv2[-1]]
            lv2[-1] += 1
        elif action == 34: # Úp thẻ ẩn cấp 3
            env[posP] = lv3[lv3[-1]]
            lv3[-1] += 1
        else: # Úp thẻ trên bàn
            posE = action - 9 # [11:23] với [20:32]
            cardId = env[posE]
            env[posP] = cardId
            
            # Mở thẻ từ chồng úp lên trên bàn chơi
            openCard(env, lv1, lv2, lv3, cardId, posE)
        
        # Nếu không thừa nguyên liệu thì next turn
        if np.sum(pStocks) <= 10:
            env[83] += 1
    
    # Mua thẻ
    elif action >= 5 and action < 20:
        pPerStocks = env[29+temp_:34+temp_]

        if action < 17: # Mua thẻ trên bàn chơi
            posE = action + 6 # [11:23] với [5:17]
        else: # Mua thẻ úp
            posE = 18 + temp_ + action
        
        cardId = env[posE]
        cardIn4 = __NORMAL_CARD__[cardId]
        price = cardIn4[2:7]

        nlMat = (price > pPerStocks) * (price - pPerStocks)
        nlBt = np.minimum(nlMat, pStocks[0:5])
        nlG = np.sum(nlMat - nlBt)

        # Trả nguyên liệu
        pStocks[0:5] -= nlBt # Trả nguyên liệu
        pStocks[5] -= nlG
        bStocks[0:5] += nlBt
        bStocks[5] += nlG

        # Nhận các phần thưởng từ thẻ
        env[89+pIdx] += 1 # Tăng số thẻ đã mua
        if action < 17: # Mua thẻ trên bàn chơi
            openCard(env, lv1, lv2, lv3, cardId, posE)
        else: # Mua thẻ úp
            env[posE] = -1
        
        env[34+temp_] += cardIn4[0] # Cộng điểm
        pPerStocks[cardIn4[1]] += 1 # Tăng nguyên liệu vĩnh viễn

        # Check noble
        if pPerStocks[cardIn4[1]] >= 3 and pPerStocks[cardIn4[1]] <= 4:
            for i in range(5):
                nobleId = env[6+i]
                if nobleId != -1:
                    nobleIn4 = __NOBLE_CARD__[nobleId]
                    price = nobleIn4[1:6]
                    if (price <= pPerStocks).all():
                        env[6+i] = -1
                        env[34+temp_] += nobleIn4[0]
        
        # Next turn
        env[83] += 1
    
    # 40: Bỏ qua lượt (trường hợp đặc biệt khi không thể thực hiện action nào)
    else:
        env[83] += 1


@njit
def getAgentSize():
    return __AGENT_SIZE__


@njit
def checkEnded(env):
    scoreArr = env[np.array([34, 49, 64, 79])]
    maxScore = np.max(scoreArr)
    if maxScore >= 15 and env[83] % 4 == 0:
        env[93] = 1
        maxScorePlayers = np.where(scoreArr==maxScore)[0]
        if len(maxScorePlayers) == 1:
            return maxScorePlayers[0]
        else:
            playerBoughtCards = env[maxScorePlayers+89]
            min_ = np.min(playerBoughtCards)
            winnerIdx = np.where(playerBoughtCards==min_)[0][-1]
            return maxScorePlayers[winnerIdx]
    else:
        return -1


@njit
def getReward(state):
    if state[271] == 0:
        return 0
    else:
        scoreArr = state[np.array([212, 224, 236, 248])]
        maxScore = np.max(scoreArr)
        if scoreArr[0] < maxScore: # Điểm của bản thân không cao nhất
            return -1
        else:
            maxScorePlayers = np.where(scoreArr==maxScore)[0]
            if len(maxScorePlayers) == 1: # Bản thân là người duy nhất đạt điểm cao nhất
                return 1
            else:
                playerBoughtCards = state[maxScorePlayers+258]
                min_ = np.min(playerBoughtCards)
                if playerBoughtCards[0] > min_: # Số thẻ của bản thân nhiều hơn
                    return -1
                else:
                    lstChk = maxScorePlayers[np.where(playerBoughtCards==min_)[0]]
                    if len(lstChk) == 1: # Bản thân là người duy nhất có số lượng thẻ ít nhất
                        return 1
                    else:
                        selfId = np.where(state[262:266] == 1)[0][0]
                        if selfId + lstChk[1] >= 4: # Chứng tỏ bản thân đi sau cùng trong lst
                            return 1
                        else: # Chứng tỏ có ít nhất một người trong list đi sau bản thân
                            return -1



def run(listAgent, perData):
    env, lv1, lv2, lv3 = initEnv()
    tempData = []
    for _ in range(__AGENT_SIZE__):
        dataOnePlayer = List()
        dataOnePlayer.append(np.array([[0.]]))
        tempData.append(dataOnePlayer)

    winner = -1
    while env[83] < 400:
        pIdx = env[83] % 4
        action, tempData[pIdx], perData = listAgent[pIdx](getAgentState(env, lv1, lv2, lv3), tempData[pIdx], perData)
        stepEnv(action, env, lv1, lv2, lv3)
        winner = checkEnded(env)
        if winner != -1:
            break
    
    for pIdx in range(4):
        env[83] = pIdx
        action, tempData[pIdx], perData = listAgent[pIdx](getAgentState(env, lv1, lv2, lv3), tempData[pIdx], perData)
    
    return winner, perData


@njit
def numbaRun(p0, p1, p2, p3, perData, pIdOrder):
    env, lv1, lv2, lv3 = initEnv()

    tempData = []
    for _ in range(__AGENT_SIZE__):
        dataOnePlayer = List()
        dataOnePlayer.append(np.array([[0.]]))
        tempData.append(dataOnePlayer)
    
    winner = -1
    while env[83] < 400:
        pIdx = env[83] % 4
        if pIdOrder[pIdx] == 0:
            action, tempData[pIdx], perData = p0(getAgentState(env, lv1, lv2, lv3), tempData[pIdx], perData)
        elif pIdOrder[pIdx] == 1:
            action, tempData[pIdx], perData = p1(getAgentState(env, lv1, lv2, lv3), tempData[pIdx], perData)
        elif pIdOrder[pIdx] == 2:
            action, tempData[pIdx], perData = p2(getAgentState(env, lv1, lv2, lv3), tempData[pIdx], perData)
        elif pIdOrder[pIdx] == 3:
            action, tempData[pIdx], perData = p3(getAgentState(env, lv1, lv2, lv3), tempData[pIdx], perData)
        
        stepEnv(action, env, lv1, lv2, lv3)
        winner = checkEnded(env)
        if winner != -1:
            break
    
    for pIdx in range(4):
        env[83] = pIdx
        if pIdOrder[pIdx] == 0:
            action, tempData[pIdx], perData = p0(getAgentState(env, lv1, lv2, lv3), tempData[pIdx], perData)
        elif pIdOrder[pIdx] == 1:
            action, tempData[pIdx], perData = p1(getAgentState(env, lv1, lv2, lv3), tempData[pIdx], perData)
        elif pIdOrder[pIdx] == 2:
            action, tempData[pIdx], perData = p2(getAgentState(env, lv1, lv2, lv3), tempData[pIdx], perData)
        elif pIdOrder[pIdx] == 3:
            action, tempData[pIdx], perData = p3(getAgentState(env, lv1, lv2, lv3), tempData[pIdx], perData)
    
    return winner, perData



def main(listAgent, times, perData, printMode=False, k=100):
    if len(listAgent) != __AGENT_SIZE__:
        raise Exception('Hệ thống chỉ cho phép có đúng 4 người chơi!!!')
    
    numWin = np.full(5, 0)
    pIdOrder = np.arange(__AGENT_SIZE__)
    for _ in range(times):
        if printMode and _ != 0 and _ % k == 0:
            print(_, numWin)

        np.random.shuffle(pIdOrder)
        shuffledListAgent = [listAgent[i] for i in pIdOrder]
        winner, perData = run(shuffledListAgent, perData)

        if winner == -1:
            numWin[4] += 1
        else:
            numWin[pIdOrder[winner]] += 1
    
    if printMode:
        print(_+1, numWin)

    return numWin, perData


@njit
def numbaMain(p0, p1, p2, p3, times, perData, printMode=False, k=100):
    numWin = np.full(5, 0)
    pIdOrder = np.arange(__AGENT_SIZE__)
    for _ in range(times):
        if printMode and _ != 0 and _ % k == 0:
            print(_, numWin)

        np.random.shuffle(pIdOrder)
        winner, perData = numbaRun(p0, p1, p2, p3, perData, pIdOrder)

        if winner == -1:
            numWin[4] += 1
        else:
            numWin[pIdOrder[winner]] += 1
    
    if printMode:
        print(_+1, numWin)

    return numWin, perData



def randomBot(state, tempData, perData):
    validActions = getValidActions(state)
    validActions = np.where(validActions==1)[0]
    idx = np.random.randint(0, len(validActions))
    return validActions[idx], tempData, perData


@njit
def numbaRandomBot(state, tempData, perData):
    validActions = getValidActions(state)
    validActions = np.where(validActions==1)[0]
    idx = np.random.randint(0, len(validActions))
    return validActions[idx], tempData, perData