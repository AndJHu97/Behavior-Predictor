from enum import Enum
import numpy as np

LOW_DIFF = 0.1
MID_DIFF = 0.5
HIGH_DIFF = 1

STAGE_DIRE = 10
STAGE_BAD = 20
STAGE_OKAY = 50
STAGE_BETTER = 70
#STAGE_PERFECT 

DIRE_MOD = 1.8
BAD_MOD = 1.5
OKAY_MOD = 1.2
BETTER_MOD = 1.1
PERFECT_MOD = 1


class Action(Enum):
    Fight = 0
    Flee = 1
    #Befriend = 2
    #Cry = 3
    #Investigate = 4
    #Ridicule = 5
    #Submit = 6
    #Ignore = 7

class SituationType(Enum):
    Threat = 0
    Ally = 1
    Count = 2

class Situation:
    def __init__(self, sitL, sitB, sitType):
        self.sitL = sitL
        self.sitB = sitB
        self.sitType = sitType
    def check_death(self, character):
        if character.relB < 0 or character.relL < 0:
            return True
        else:
            return False
    def calculate_reward(self, stat):
        if stat < STAGE_DIRE:
            statChange = DIRE_MOD
        elif stat < STAGE_BAD:
            statChange = BAD_MOD
        elif stat < STAGE_OKAY:
            statChange = OKAY_MOD
        elif stat < STAGE_BETTER:
            statChange = BETTER_MOD
        else:
            statChange = PERFECT_MOD

        return statChange
        
class Threat(Situation):
    def __init__(self, sitL, sitB, sitType):
        super().__init__(sitL, sitB, sitType) 
    def process_action(self, character, action):
        death = False
        lChange, bChange = 0, 0
        if action == Action.Fight.value:
            lChange = self.calculateLFight(character.relL, self.sitL)
            bChange = self.calculateBFight(character.relL, self.sitL)
        elif action == Action.Flee.value:
            lChange = self.calculateLFlee(character.relL, self.sitL)
            bChange = self.calculateBFlee(character.relL, self.sitL)
        elif action == Action.Befriend.value:
            lChange = self.calculateLFriend(character.relL, self.sitL)
            bChange = self.calculateBFriend(character.relL, self.sitL)
        
        #get the x value of the curve to change
        character.absL += lChange
        character.absB += bChange
        #use the curve to calculate the y value
        newRelL = character.calculate_relative_stats(character.absL)
        newRelB = character.calculate_relative_stats(character.absB)
        #Get the change of the y curve to use as reward
        relLChange = newRelL - character.relL
        relBChange = newRelB - character.relB
        #update new one
        character.relL = newRelL
        character.relB = newRelB
        #reward is adding together change in b/l 
        lReward = self.calculate_reward(newRelL) + relLChange
        bReward = self.calculate_reward(newRelB) + relBChange
        death = self.check_death(character)
        if death:
            if newRelL < 0:
                lReward -= 20
            else:
                bReward -= 20
            return lReward, bReward, death, character.survival_rounds
        character.survival_rounds += 1
        return lReward + 5, bReward + 5, death, character.survival_rounds
    def calculateLFight(self, lPlayer, lOpp):
        #graph it
        x = lPlayer - lOpp
        y = x - 30
        if y > 0:
            y = 0
        return y
    def calculateBFight(self, lPlayer, lOpp):
        x = lPlayer - lOpp
        if x > 50:
            y = -(x/5 - 10)**2 + 60
            if y < 0:
                y = 0
        elif x <=50 and x >= 0:
            y = -(x/10 - 5)**2 + 60
        elif x < 0 and x >= -50:
            y = (x/10 + 5)**2 - 60
        elif x < - 50:
            y = (x/5 +10)**2 - 60
            if y > 0:
                y = 0
        return y
    def calculateLFlee(self,lPlayer,lOpp):
        x = lPlayer - lOpp
        if x > -15:
            y = 0
        #when running away and you fail. Starts at -40
        else:
            y = x - 15
        return y
    def calculateBFlee(self, lPlayer, lOpp):
        x = lPlayer - lOpp
        #if you're strong, it looks worse when you flee
        if x > -15:
            y = -x - 40
        #when you fail running away, youre going to lose more, the more of a beating you take
        else:
            y = x - 15
        return y
    def calculateLFriend(self, lPlayer, lOpp):
        x = lPlayer - lOpp
        #cannot befriend a human threat
        y = x - 40
        if y > 0:
            y = 0
        return y
    def calculateBFriend(self, lPlayer, lOpp):
        x = lPlayer - lOpp
        #cannot befriend a human threat
        y = x - 40
        if y > 0:
            y = 0
        return y

class Ally(Situation):
    def __init__(self, sitL, sitB, sitType):
        super().__init__(sitL, sitB, sitType), 
    def process_action(self, character, action):
        death = False
        lChange, bChange = 0, 0
        if action == Action.Fight.value:
            if character.relL > self.sitL * MID_DIFF:
                lChange = -10
                bChange = -10
            else:
                lChange = -15
                bChange = -10
        elif action == Action.Flee.value:
            if character.relL > self.sitL * LOW_DIFF:
                lChange = 10
                bChange = 10
            else:
                lChange = 10
                bChange = 10
        elif action == Action.Befriend.value:
            lChange = 10
            bChange = 25
        
        #get the x value of the curve to change
        character.absL += lChange
        character.absB += bChange
        #use the curve to calculate the y value
        newRelL = character.calculate_relative_stats(character.absL)
        newRelB = character.calculate_relative_stats(character.absB)
        #Get the change of the y curve to use as reward
        relLChange = newRelL - character.relL
        relBChange = newRelB - character.relB
        #update new one
        character.relL = newRelL
        character.relB = newRelB
        #reward is adding together change in b/l 
        lReward = self.calculate_reward(newRelL) * relLChange
        bReward = self.calculate_reward(newRelB) * relBChange
        #print("L reward: ", lReward, " change in L: ", relLChange, " new L: ", newRelL)
        #print("B reward: ", bReward, " change in B: ", relBChange, " new B: ", newRelB)
        death = self.check_death(character)
        if death:
            if newRelL < 0:
                lReward -= 20
            else:
                bReward -= 20
            return lReward, bReward, death, character.survival_rounds
        character.survival_rounds += 1
        return lReward , bReward, death, character.survival_rounds
            

class Character:
    def __init__(self, absL, absB):
        self.absL = absL
        self.absB = absB
        self.relL = self.calculate_relative_stats(absL)
        self.relB = self.calculate_relative_stats(absB)
        self.survival_rounds = 0
    
    def set_stats(self, absL, absB):
        self.absL = absL
        self.absB = absB

    #this will mimic real life more. It's harder to lose when are close to dying and harder to gain when almost full health. 
    def calculate_relative_stats(self, stat):
        return modified_exponential_bound(stat)

def modified_exponential_bound(x, k=0.02):
    return 100 * (1 - np.exp(-k * x))

