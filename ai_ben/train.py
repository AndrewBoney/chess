import os
import time
import math
import json
import torch
import random

import pandas as pd

from copy import deepcopy
from shutil import copyfile
from tqdm import tqdm

import sys
sys.path.insert(0, os.getcwd())

from chess import Chess
from ai_ben.ai import Agent, Plumbing
from ai_ben.model import TransformerModel
from ai_ben.config import Config

"""
Reinforcement training for AI
"""

def get_batch(source, x, y):
    """
    Input: source - pytorch tensor containing data you wish to get batches from
            x - integer representing the index of the data you wish to gather
            y - integer representing the amount of rows you want to grab
    Description: Generate input and target data for training model
    Output: list of pytorch tensors containing input and target data [x,y]
    """
    data = torch.tensor([])
    v_target = torch.tensor([])
    p_target = torch.tensor([])
    for i in range(y):
        #Training data
        if len(source) > 0 and x+i < len(source):
            d_seq = source[x+i][:len(source[x+i])-4099]
            data = torch.cat((data, d_seq))
            #Target data
            v_seq = source[x+i][-3:]
            v_target = torch.cat((v_target, v_seq))
            p_seq = source[x+i][-4099:-3]
            p_target = torch.cat((p_target, p_seq))

    return data.reshape(min(y, len(source[x:])), len(source[0])-4099).to(torch.int64), v_target.reshape(min(y, len(source[x:])), 3).to(torch.float), p_target.reshape(min(y, len(source[x:])), 4096).to(torch.float)

class train:
    """
    Input: game_name - string representing the game board name
           epoch - integer representing the current epoch
           train - boolean representing the training control (Default=False) [OPTIONAL]
           white - string representing the white player type for white (Default='ai') [OPTIONAL]
           black - string representing the black player type for white (Default='ai') [OPTIONAL]
           active_model - string representing the file name for the active model (Default='model-active.pth.tar') [OPTIONAL]
           new_model - string representing the file name for the new model (Default='model-new.pth.tar') [OPTIONAL]
           search_amount - integer representing the amount of searches the ai's should perform (Default=50) [OPTIONAL]
           max_depth - integer representing the max depth each search can go (Default=5) [OPTIONAL]
           best_of - integer representing the amount of games played in a bracket (Default=5) [OPTIONAL]
    Description: Plays game for training
    Output: tuple containing game state, training data and which of the players won
    """
    def play_game(game_name, epoch, train=False, white='ai', black='ai', active_model='model-active.pth.tar', new_model='model-new.pth.tar', search_amount=50, max_depth=5, best_of=5):
        # TODO: deprecate ai vs human
        if str(white).lower() == 'ai' and str(black).lower() == 'ai':
            if (epoch+1) % best_of == 0:
                a_colour = random.choice(['w', 'b'])
            elif (epoch+1) % 2 == 0:
                a_colour = 'b'
            else:
                a_colour = 'w'
        elif str(white).lower() != 'ai' and str(black).lower() == 'ai':
            a_colour = 'b'
        elif str(white).lower() == 'ai' and str(black).lower() != 'ai':
            a_colour = 'w'
        else:
            a_colour = None
        
        if a_colour == 'w' and str(white).lower() == 'ai' and str(black).lower() == 'ai':
            w_bot = deepcopy(Agent(search_amount=search_amount, max_depth=max_depth, train=train, model=active_model))
            b_bot = deepcopy(Agent(search_amount=search_amount, max_depth=max_depth, train=train, model=new_model))
        elif a_colour == 'b' and str(white).lower() == 'ai' and str(black).lower() == 'ai':
            w_bot = deepcopy(Agent(search_amount=search_amount, max_depth=max_depth, train=train, model=new_model))
            b_bot = deepcopy(Agent(search_amount=search_amount, max_depth=max_depth, train=train, model=active_model))
        else:
            w_bot = None
            b_bot = None
        log = []
        plumbing = Plumbing()
        chess_game = deepcopy(Chess()) #'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'

        pbar = tqdm()
        while True:
            if chess_game.p_move == 1:
                cur,next = w_bot.choose_action(chess_game)
                w_log = pd.DataFrame(w_bot.log).drop_duplicates()
                if len(w_log) > 0:
                    if 'imag_log' not in locals():
                        imag_log = pd.DataFrame(columns=list(w_log.columns.values))
                    for k,v in w_log.groupby(['value0','value1','value2']):
                        t_log = pd.DataFrame(log)
                        for i,x in enumerate(k):
                            t_log[f'value{i}'] = [x]*len(t_log)
                        t_log = pd.concat([t_log, v], ignore_index=True)
                        imag_log = pd.concat([imag_log, t_log], ignore_index = True) 
                    imag_log = imag_log.drop_duplicates()
            else:
                cur,next = b_bot.choose_action(chess_game)
                b_log = pd.DataFrame(b_bot.log).drop_duplicates()
                if len(b_log) > 0:
                    if 'imag_log' not in locals():
                        imag_log = pd.DataFrame(columns=list(b_log.columns.values))
                    for k,v in b_log.groupby(['value0','value1','value2']):
                        t_log = pd.DataFrame(log)
                        for i,x in enumerate(k):
                            t_log[f'value{i}'] = [x]*len(t_log)
                        t_log = pd.concat([t_log, v], ignore_index=True)
                        imag_log = pd.concat([imag_log, t_log], ignore_index = True) 
                    imag_log = imag_log.drop_duplicates()
            #print(f'w {cur.lower()}-->{next.lower()} | EPOCH:{epoch} BOARD:{game_name} MOVE:{len(log)} HASH:{chess_game.EPD_hash()}\n') if chess_game.p_move > 0 else print(f'b {cur.lower()}-->{next.lower()} | EPOCH:{epoch} BOARD:{game_name} MOVE:{len(log)} HASH:{chess_game.EPD_hash()}\n')
            enc_game = plumbing.encode_state(chess_game)
            valid = False
            if chess_game.move(cur, next) == False:
                pass 
                #print('Invalid move')
            else:
                valid = True
                cur_pos = chess_game.board_2_array(cur)
                next_pos = chess_game.board_2_array(next)
                log.append({**{f'state{i}':float(s) for i, s in enumerate(enc_game[0])},
                            **{f'action{x}':1 if x == ((cur_pos[0]+(cur_pos[1]*8))*64)+(next_pos[0]+(next_pos[1]*8)) else 0 for x in range(4096)}})
            if (str(white).lower() == 'ai' and chess_game.p_move == 1) or (str(black).lower() == 'ai' and chess_game.p_move == -1):
                state = chess_game.check_state(chess_game.EPD_hash())
                if state == '50M' or state == '3F':
                    state = [0, 1, 0] #Auto tie
                elif state == 'PP':
                    chess_game.pawn_promotion(n_part='Q') #Auto queen
                if state != [0, 1, 0]:
                    state = chess_game.is_end()
            else:
                state = chess_game.is_end()
                if state == [0, 0, 0]:
                    if chess_game.check_state(chess_game.EPD_hash()) == 'PP':
                        chess_game.pawn_promotion()
            if sum(state) > 0:
                #print(f'FINISHED | EPOCH:{epoch} BOARD:{game_name} MOVE:{len(log)} STATE:{state}\n')
                game_train_data = pd.DataFrame(log)
                for i, x in enumerate(state):
                    game_train_data[f'value{i}'] = [x]*len(log)
                if 'imag_log' in locals():
                    game_train_data = pd.concat([game_train_data, imag_log], ignore_index = True) 
                game_train_data = game_train_data.astype(float)
                break
            if valid == True:
                chess_game.p_move = chess_game.p_move * (-1)
            
            pbar.update()
        
        pbar.close()
        return state, game_train_data, a_colour

    """
    Input: boards - dictionary containing different games you want to play
           epoch - integer representing the current epoch
    Description: Play multiple games in parrallel
    Output: data frame containing training data
    """
    def run_multiple_matches(boards, epoch):
        train_data = pd.DataFrame()
        func = [{'name':f'game-{x}', 'func':train.play_game, 'args':(x, epoch, True)} for x in range(boards)]
        games = Plumbing.multi_process(func, workers=BOARDS)
        for g in games:
            state, game_train_data, a_colour = games[g]
            if (state == [0, 0, 1] and a_colour == 'b') or (state == [1, 0, 0] and a_colour == 'w'):
                print('ACTIVE WINS\n')
                game_results['active'] += 1
            elif (state == [0, 0, 1] and a_colour == 'w') or (state == [1, 0, 0] and a_colour == 'b'):
                print('NEW WINS\n')
                game_results['new'] += 1
            else:
                print('TIE GAME\n')
                game_results['tie'] += 1
            train_data = pd.concat([train_data, game_train_data], ignore_index=True)

        return train_data

if __name__ == '__main__':
    GAMES = 10 #Games to play on each board
    BOARDS = 1 #Amount of boards to play on at a time
    BEST_OF = 5 #Amount of games played when evaluating the models

    white = 'ai' #Values ['human', 'ai']
    black = 'ai' #Values ['human', 'ai']

    folder = 'ai_ben/data' #Folder name where data is saved
    parameters = 'model_param.json' #Model parameters filename
    active_weights = 'model-active.pth.tar' #Active model saved weights filename
    new_weights = 'model-new.pth.tar' #Active model saved weights filename

    #Model parameters
    config = Config() # TODO: add args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Set divice training will use

    #Training parameters
    bsz = 50 #Batch size
    lr = 0.0005 #Learning rate
    total_loss = 0.0 #Initalize total loss

    #Begin training games
    train_data = pd.DataFrame()
    game_results = {'active':0, 'new':0, 'tie':0}
    for epoch in range(GAMES):
        print(f'STARTING GAMES\n')

        state, g_results, a_colour = train.play_game(0, epoch, train=True, white=white, black=black, best_of=BEST_OF)
        if ((state == [0, 0, 1] and a_colour == 'b') or (state == [1, 0, 0] and a_colour == 'w')) and str(white).lower() == 'ai' and str(black).lower() == 'ai':
            print('ACTIVE AI WINS\n')
            game_results['active'] += 1
        elif ((state == [0, 0, 1] and a_colour == 'w') or (state == [1, 0, 0] and a_colour == 'b')) and str(white).lower() == 'ai' and str(black).lower() == 'ai':
            print('NEW AI WINS\n')
            game_results['new'] += 1
        elif state == [1, 0, 0] and a_colour == 'b' and str(white).lower() != 'ai' and str(black).lower() == 'ai':
            print('YOU WIN\n')
            game_results['active'] += 1
        elif state == [0, 0, 1] and a_colour == 'w' and str(white).lower() == 'ai' and str(black).lower() != 'ai':
            print('YOU WIN\n')
            game_results['active'] += 1
        elif state == [1, 0, 0] and a_colour == 'w' and str(white).lower() == 'ai' and str(black).lower() != 'ai':
            print('NEW AI WINS\n')
            game_results['new'] += 1
        elif state == [0, 0, 1] and a_colour == 'b' and str(white).lower() != 'ai' and str(black).lower() == 'ai':
            print('NEW AI WINS\n')
            game_results['new'] += 1
        else:
            print('TIE GAME\n')
            game_results['tie'] += 1

        train_data = pd.concat([train_data, g_results], ignore_index=True).drop_duplicates()

        print(epoch,game_results,'\n')
        if sum([v for v in game_results.values()]) >= BEST_OF and game_results['new']/max(sum([game_results['new'],game_results['active']]),1) >= 0.51 and str(white).lower() == 'ai' and str(black).lower() == 'ai':
            print(f"NEW MODEL OUTPERFORMED ACTIVE MODEL ({round(game_results['new']/max(sum([game_results['new'],game_results['active']]),1),3)*100}%)\n")
            copyfile(os.path.join(folder,new_weights),os.path.join(folder,active_weights)) #Overwrite active model with new model
        if sum([v for v in game_results.values()]) >= BEST_OF:
            game_results = {'active':0, 'new':0, 'tie':0}

        #Load current new model
        model = TransformerModel(config).to(device) #Initialize the transformer model
        filepath = os.path.join(folder, new_weights)
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])

        #Initailize training
        criterion = torch.nn.BCELoss() #Binary cross entropy loss
        optimizer = torch.optim.SGD(model.parameters(), lr=lr) #Optimization algorithm using stochastic gradient descent
        model.train() #Turn on the train mode
        start_time = time.time() #Get time of starting process
        train_data = train_data.sample(frac=1).reset_index(drop=True) #Shuffle training data
        train_data = torch.tensor(train_data.values) #Set training data to a tensor

        #Start training model
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bsz)):
            data, v_targets, p_targets = TransformerModel.get_batch(train_data, i, bsz) #Get batch data with the selected targets being masked
            output = model(data) #Make prediction using the model
            v_loss = criterion(output[0], v_targets) #Apply loss function to results
            p_loss = criterion(output[1], p_targets) #Apply loss function to results
            loss = v_loss + p_loss
            loss.backward() #Backpropegate through model
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item() #Increment total loss

        #Updated new model
        filepath = os.path.join(folder, new_weights)
        if not os.path.exists(folder):
            os.mkdir(folder)
        torch.save({
            'state_dict': model.state_dict(),
        }, filepath)
        print(f'{epoch} | {BOARDS} games | {time.time() - start_time} ms | {train_data.size(0)} samples | {total_loss/(GAMES*BOARDS)} loss\n')