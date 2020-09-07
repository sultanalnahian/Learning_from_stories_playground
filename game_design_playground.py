import textworld
from textworld import GameMaker
from textworld.generator.data import KnowledgeBase
from textworld.generator.game import Event, Quest
from textworld.generator.game import GameOptions

from textworld.generator import compile_game
import io
import sys
import contextlib
import numpy as np

# Make the generation process reproducible.
from textworld import g_rng  # Global random generator.
g_rng.set_seed(20180916)

def _compile_test_game(game):
    grammar_flags = {
        "theme": "house",
        "include_adj": False,
        "only_last_action": True,
        "blend_instructions": True,
        "blend_descriptions": True,
        "refer_by_name_only": True,
        "instruction_extension": []
    }
    rng_grammar = np.random.RandomState(1234)
    grammar = textworld.generator.make_grammar(grammar_flags, rng=rng_grammar)
    game.change_grammar(grammar)

    game_file = textworld.generator.compile_game(game)
    return game_file

def build_and_compile_playground_game():
    M = GameMaker()
    
    room_dr = M.new_room("Locker Room")
    room_mr = M.new_room("Meeting Room")
    room_fl = M.new_room("Field")
    team_mates = M.new(type='tm', name = "teammates")
    captain = M.new(type='tm3', name = "captain")
    room_dr.add(team_mates)
    room_mr.add(captain)
    corridor = M.connect(room_dr.east, room_mr.west)
    corridor1 = M.connect(room_mr.east, room_fl.west)
    #corridor2 = M.connect(room_fl.east, new_room.west)
    M.set_player(room_dr)

    team_mates_2 = M.new(type='tm2', name = "someone")
    room_dr.add(team_mates_2)

    game_gear = M.new(type='ap', name = "game gears")
    room_dr.add(game_gear)
    M.add_fact("not_equiped", game_gear)
    M.add_fact("not_participated", captain)
    M.add_fact("not_helped", team_mates)
    M.add_fact("not_disturbed", team_mates)
    M.add_fact("not_bully", team_mates)
    
    
    room_dr.infos.desc = "You are in the Locker room with your team mates. You will find your game gears here. Collect the game gears and wear them before the meeting. If you need help, ask help from others and be helpful to others as well. \n After you are done here, go to the meeting room."
    room_mr.infos.desc = "You are in the meeting room. A meeting is going to start to finalize the game plan. Join the meeting and try to share your thought."
    
    captain.infos.desc = "The meeting is leading by the captain. Please, Cooperate with him."
    
    failed_event_1 = Event(conditions={
        M.new_fact("not_equiped", game_gear),
        M.new_fact("at", M._entities['P'], room_mr)})
    
    failed_quest_1 = Quest(win_events=[],
                           fail_events=[failed_event_1])
    
    
    #M.quests = [failed_quest_1]
    game = M.build()
    game_file = _compile_test_game(game)
    return game, game_file

build_and_compile_playground_game()

