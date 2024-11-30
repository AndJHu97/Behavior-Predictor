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
    def __init__(self, sitL, sitDB, sitNB, sitType):
        self.sitL = sitL
        self.sitDB = sitDB
        self.sitType = sitType
        self.sitNB = sitNB
    def check_death(self, character):
        #Could make it so they have an identity later on
        if (character.mainRelB() < 0) or character.relL < 0:
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
    def __init__(self, sitL, sitDB, sitNB, sitType):
        super().__init__(sitL, sitDB, sitNB, sitType) 
    def process_action(self, character, action):
        death = False
        lChange, dbChange = 0, 0
        if action == Action.Fight.value:
            lChange = self.calculateLFight(character.relL, self.sitL)
            dbChange = self.calculateDBFight(character.relL, character.relDB, self.sitL, self.sitDB)
        elif action == Action.Flee.value:
            lChange = self.calculateLFlee(character.relL, self.sitL)
            dbChange = self.calculateDBFlee(character.relL, character.relDB, self.sitL, self.sitDB)
        elif action == Action.Befriend.value:
            lChange = self.calculateLFriend(character.relL, self.sitL)
            dbChange = self.calculateBFriend(character.relL, self.sitL)
        
        #get the x value of the curve to change
        character.absL += lChange
        character.absDB += dbChange
        #use the curve to calculate the y value
        newRelL = character.calculate_relative_stats(character.absL)
        newRelDB = character.calculate_relative_stats(character.absDB)
        #Get the change of the y curve to use as reward
        relLChange = newRelL - character.relL
        relDBChange = newRelDB - character.relDB
        #update new one
        character.relL = newRelL
        character.relDB = newRelDB
        #reward is adding together change in b/l 
        lReward = self.calculate_reward(newRelL) + relLChange
        dbReward = self.calculate_reward(newRelDB) + relDBChange
        nbReward = 0
        death = self.check_death(character)
        if death:
            if newRelL < 0:
                lReward -= 20
            else:
                dbReward -= 20
            return lReward, dbReward, nbReward, death, character.survival_rounds
        character.survival_rounds += 1
        return lReward + 5, dbReward + 5, nbReward + 5, death, character.survival_rounds
    def calculateLFight(self, lAgent, lEnv):
        lLoss = min(lAgent - lEnv * 1.1, 0)
        print("L fight: ", lLoss)
        print("lAgent: ", lAgent, " lEnv: ", lEnv)
        return lLoss
    def calculateDBFight(self, lAgent, dbAgent, lEnv, dbEnv):
        dbEnd = (dbAgent + dbEnv)/2 + (lAgent - lEnv) / 2
        print("DB fight: ", (dbEnd - dbAgent))
        print("lAgent: ", lAgent, " dbAgent: ", dbAgent, " lEnv: ", lEnv, " dbEnv: ", dbEnv)
        return dbEnd - dbAgent
    def calculateLFlee(self,lAgent,lEnv):
        if(lAgent < lEnv * 0.65): 
            print("L Flee: ", lAgent * 0.65 - lEnv * 1.5)
            print("lAgent: ", lAgent, " lEnv: ", lEnv)
            return lAgent * 0.65 - lEnv * 1.5
        else:
            return 0

    def calculateDBFlee(self, lAgent, dbAgent, lEnv, dbEnv):
        if(lAgent < lEnv * 0.65):
            dbEnd = (dbAgent + dbEnv)/2 + (lAgent * 0.7 - lEnv) / 2 
            print("DB flee caught: ", (dbEnd - dbAgent))
            print("lAgent: ", lAgent, " dbAgent: ", dbAgent, " lEnv: ", lEnv, " dbEnv: ", dbEnv)
            return min(dbEnd - dbAgent, 0)
        else:
            dbEnd = dbEnv * 0.7
            print("DB flee escaped: ", (dbEnd - dbAgent))
            print("lAgent: ", lAgent, " dbAgent: ", dbAgent, " lEnv: ", lEnv, " dbEnv: ", dbEnv)
            return min(dbEnd - dbAgent,0)

    def calculateLFriend(self, lAgent, lEnv):
        return max(lAgent * 0.65 - lEnv * 1.5, 0)
    def calculateBFriend(self, lPlayer, lOpp):
        x = lPlayer - lOpp
        #cannot befriend a human threat
        y = x - 40
        if y > 0:
            y = 0
        return y

class Ally(Situation):
    def __init__(self, sitL, sitDB, sitNB, sitType):
        super().__init__(sitL, sitDB, sitNB, sitType), 
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
    def __init__(self, absL, absDB, absNB, mainB):
        self.absL = absL
        self.absNB = absNB
        self.absDB = absDB
        self.mainB = mainB
        self.relL = self.calculate_relative_stats(absL)
        self.relNB = self.calculate_relative_stats(absNB)
        self.relDB = self.calculate_relative_stats(absDB)
        self.survival_rounds = 0
    
    def set_stats(self, absL, absDB, absNB):
        self.absL = absL
        self.absNB = absNB
        self.absDB = absDB

    def mainRelB(self):
        if self.mainB == "DB":
            return self.relDB
        elif self.mainB == "NB":
            return self.relNB
        else:
            raise ValueError(f"Invalid value for mainB: {self.mainB}")

    #this will mimic real life more. It's harder to lose when are close to dying and harder to gain when almost full health. 
    def calculate_relative_stats(self, stat):
        return modified_exponential_bound(stat)

def modified_exponential_bound(x, k=0.02):
    return 100 * (1 - np.exp(-k * x))

