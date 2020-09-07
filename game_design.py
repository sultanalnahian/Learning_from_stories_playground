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
    
    
    room_dr.infos.desc = "You are in the Locker room with your team mates. You will find your game gears here. Collect the game gears and wear them before the meeting. If you need help, ask help from others and be helpful to others as well. \n After your are done here, go to the meeting room."
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

def build_and_compile_super_hero_game():
    M = GameMaker()
    roomA = M.new_room("Room A")
    alley = M.new_room("Alley")
    bank1 = M.new_room("Bank1")
    bank2 = M.new_room("Bank2")
    bank3 = M.new_room("Bank3")
    corridor = M.connect(roomA.east, alley.west)
    corridor1 = M.connect(alley.east, bank1.west)
    corridor1 = M.connect(alley.north, bank2.south)
    corridor1 = M.connect(alley.south, bank3.north)
    M.set_player(roomA)
    
    roomA.infos.desc = "You are in a road. Some mobs are planning to rob a bank. You need to stop them. Go east to the alley. You can find a person in the alley who has information about the roberry. Collect information from him and prevent the roberry."
    alley.infos.desc = "This is an alley. There is a person beside the table. He knows about the bank roberry."
    bank2.infos.desc = "This is the north bank. Some robbers are going to rob the bank. You can call the police and try to capture them or convince them to surrender. Or you can also shoot them to stop the robbery."
    
    money = M.new(type="o", name = 'money') 
    money.infos.desc = "it is money"
    M.inventory.add(money) 
    person = M.new(type="pr", name = "informant")
    person.infos.desc = "This person knows about the bank roberry. Do a favor for him. He will help you."
    M.add_fact("not_asked", person)
    M.add_fact("not_given", person)
    alley.add(person)
    
    robber = M.new(type="rbr", name = "joker")
    bank2.add(robber)
    M.add_fact("not_stopped", robber)
    
    M.add_fact("not_conflict", robber)
    M.add_fact("not_allowed", robber)
    
    police = M.new(type="pl", name = "police")
    bank2.add(police)
    M.add_fact("not_called", police)
    
    
    # asking quest
    qst_event_asking = Event(conditions={M.new_fact("asked", person)})
    quest_asking = Quest(win_events=[qst_event_asking],
                      reward=2)
    # the wining quest
    qst_event_stopped_rob = Event(conditions={M.new_fact("asked", person),
                                             M.new_fact("stopped", robber)})
    win_quest = Quest(win_events=[qst_event_stopped_rob],
                      reward=2)

    # 1st failure condition
    failed_cmds1 = ["go east", "go south"]
    failed_event1 = M.new_event_using_commands(failed_cmds1)
    failed_quest_1 = Quest(win_events=[],
                           fail_events=[failed_event1])
    
    # 2nd failure condition
    failed_cmds2 = ["go east", "go east"]
    failed_event2 = M.new_event_using_commands(failed_cmds2)
    failed_quest_2 = Quest(win_events=[],
                           fail_events=[failed_event2])
    
    # 3rd failure condition
    failed_event3 = Event(conditions={
        M.new_fact("not_asked", person),
        M.new_fact("at", M._entities['P'], bank2)})
    
    failed_quest_3 = Quest(win_events=[],
                           fail_events=[failed_event3])
    
    
    failed_event4 = Event(conditions={
        M.new_fact("allowed", robber)})
    
    failed_quest_4 = Quest(win_events=[],
                   fail_events=[failed_event4])
    
    
    M.quests = [win_quest, quest_asking, failed_quest_1, failed_quest_2, failed_quest_3, failed_quest_4]
    game = M.build()

    game.main_quest = win_quest
    game_file = _compile_test_game(game)
    return game, game_file

#build_and_compile_playground_game()
build_and_compile_super_hero_game()
