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
    Befriend = 2
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
            dbChange = self.calculateBFriend(character.relL, character.relDB, self.sitL, self.sitDB)
        
        '''
        Don't use relative stats anymore. It's kind of useless
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
        '''

        newRelL = character.relL + lChange
        newRelDB = character.relDB + dbChange
        newRelL = min(100, newRelL)
        newRelDB = min(100, newRelDB)

        relLChange = newRelL - character.relL
        relDBChange = newRelDB - character.relDB

        #reward is adding together change in b/l 
        lReward = self.calculate_reward(newRelL) * relLChange
        dbReward = self.calculate_reward(newRelDB) * relDBChange
        nbReward = 0

        character.relL = newRelL
        character.relDB = newRelDB

        #Doesn't pay attention to reward based on the B not related to their mainB
        if character.mainB == "DB":
            nbReward = 0
        elif character.mainB == "NB":
            dbReward = 0

        death = self.check_death(character)
        if death:
            if newRelL < 0:
                lReward -= 20
            else:
                if character.mainB == "DB":
                    dbReward -= 20
                elif character.mainB == "NB":
                    nbReward -= 20
            return lReward, dbReward, nbReward, death, character.survival_rounds
        character.survival_rounds += 1
        return lReward, dbReward, nbReward, death, character.survival_rounds
    def calculateLFight(self, lAgent, lEnv):
        lLoss = min(lAgent - lEnv * 1.1, 0)
        print("Threat L fight: ", lLoss)
        print("lAgent: ", lAgent, " lEnv: ", lEnv)
        return lLoss
    def calculateDBFight(self, lAgent, dbAgent, lEnv, dbEnv):
        dbEnd = (dbAgent + dbEnv)/2 + (lAgent - lEnv) / 2
        print("Threat DB fight: ", (dbEnd - dbAgent))
        print("lAgent: ", lAgent, " dbAgent: ", dbAgent, " lEnv: ", lEnv, " dbEnv: ", dbEnv)
        return dbEnd - dbAgent
    def calculateLFlee(self,lAgent,lEnv):
        if(lAgent < lEnv * 0.55): 
            print("Threat L Flee: ", lAgent * 0.7 - lEnv * 1.1)
            print("lAgent: ", lAgent, " lEnv: ", lEnv)
            return lAgent * 0.9 - lEnv * 1.1
        else:
            print("Threat L Flee escaped: ", 0)
            return 0

    def calculateDBFlee(self, lAgent, dbAgent, lEnv, dbEnv):
        if(lAgent < lEnv * 0.65):
            dbEnd = (dbAgent + dbEnv)/2 + (lAgent * 0.7 - lEnv) / 2 
            print("Threat DB flee caught: ", (dbEnd - dbAgent))
            print("lAgent: ", lAgent, " dbAgent: ", dbAgent, " lEnv: ", lEnv, " dbEnv: ", dbEnv)
            return min(dbEnd - dbAgent, 0)
        else:
            dbEnd = dbEnv * 0.7
            print("Threat DB flee escaped: ", (dbEnd - dbAgent))
            print("lAgent: ", lAgent, " dbAgent: ", dbAgent, " lEnv: ", lEnv, " dbEnv: ", dbEnv)
            return min(dbEnd - dbAgent,0)

    def calculateLFriend(self, lAgent, lEnv):
        print("Threat L Friend: ", lAgent * 0.7 - lEnv * 1.1)
        print("lAgent: ", lAgent, " lEnv: ", lEnv)
        return min(lAgent * 0.7 - lEnv * 1.1, 0)
    def calculateBFriend(self, lAgent, dbAgent, lEnv, dbEnv):
        dbEnd = (dbAgent + dbEnv)/2 + (lAgent - lEnv) / 2
        print("Threat DB friend: ", (dbEnd - dbAgent))
        print("lAgent: ", lAgent, " dbAgent: ", dbAgent, " lEnv: ", lEnv, " dbEnv: ", dbEnv)
        return dbEnd - dbAgent

class Ally(Situation):
    def __init__(self, sitL, sitDB, sitNB, sitType):
        super().__init__(sitL, sitDB, sitNB, sitType), 
    def process_action(self, character, action):
        death = False
        lChange, dbChange, nbChange = 0, 0, 0
        if action == Action.Fight.value:
            lChange = self.calculateLFight(character.relL, self.sitL)
            dbChange = self.calculateDBFight(character.relL, character.relDB, self.sitL, self.sitDB)
            nbChange = self.calculateNBFight(self.sitNB)
        elif action == Action.Flee.value:
            lChange = self.calculateLFlee()
            dbChange = self.calculateDBFlee(character.relDB, self.sitDB)
            nbChange = self.calculateNBFlee()
        elif action == Action.Befriend.value:
            lChange = self.calculateLBefriend(character.relL, self.sitNB, self.sitDB)
            dbChange = self.calculateDBBefriend(character.relDB, self.sitDB)
            nbChange = self.calculateNBBefriend(character.relNB, self.sitNB)
        
        newRelL = character.relL + lChange
        newRelDB = character.relDB + dbChange
        newRelNB = character.relNB + nbChange
        newRelL = min(100, newRelL)
        newRelDB = min(100, newRelDB)
        newRelNB = min(100, newRelNB)
        relLChange = newRelL - character.relL
        relDBChange = newRelDB - character.relDB
        relNBChange = newRelNB - character.relNB
        #reward is adding together change in b/l 
        lReward = self.calculate_reward(newRelL) * relLChange
        dbReward = self.calculate_reward(newRelDB) * relDBChange
        nbReward = self.calculate_reward(newRelNB) * relNBChange

        character.relL = newRelL
        character.relDB = newRelDB
        character.relNB = newRelNB
        #print("L reward: ", lReward, " change in L: ", relLChange, " new L: ", newRelL)
        #print("B reward: ", bReward, " change in B: ", relBChange, " new B: ", newRelB)
        death = self.check_death(character)
        if death:
            if newRelL < 0:
                lReward -= 20
            else:
                if character.mainB == "DB":
                    dbReward -= 20
                elif character.mainB == "NB":
                    nbReward -= 20
            return lReward, dbReward, nbReward, death, character.survival_rounds
        character.survival_rounds += 1
        return lReward, dbReward, nbReward, death, character.survival_rounds

    def calculateLFight(self, lAgent, lEnv):
        print("Ally L fight: ", min(0, lAgent - lEnv * 0.5))
        return min(0, lAgent - lEnv * 0.5)
    def calculateNBFight(self, nbEnv):
        print("Ally NB Fight: ", -nbEnv * 0.2)
        return -nbEnv * 0.2
    def calculateDBFight(self, lAgent, dbAgent, lEnv, dbEnv):
        dbEnd = (dbAgent + dbEnv)/2 + (lAgent - lEnv * 0.5)/2
        print("Ally DB fight: ", dbEnd - dbAgent)
        return dbEnd - dbAgent
    
    def calculateLFlee(self):
        print("Ally L Flee: ", 0)
        return 0
    
    def calculateDBFlee(self, dbAgent, dbEnv):
        print("Ally DB Flee: ", min(dbEnv * 0.7 - dbAgent, 0))
        return min(dbEnv * 0.7 - dbAgent, 0)

    def calculateNBFlee(self):
        print("Ally NB Flee: ", 0)
        return 0
    
    def calculateLBefriend(self, lAgent, nbEnv, dbEnv):
        print("Ally L Befriend: ", nbEnv * 0.15 + dbEnv * .15 - lAgent * 0.05)
        return nbEnv * 0.15 + dbEnv * .15 - lAgent * 0.05
    
    def calculateNBBefriend(self, nbAgent, nbEnv):
        print("Ally NB befriend: ", max((nbEnv - nbAgent)/3, 0))
        return max((nbEnv - nbAgent)/3, 0) 
    
    def calculateDBBefriend(self, dbAgent, dbEnv):
        print("Ally DB Befriend: ",  max((dbEnv - dbAgent)/3, 0))
        return max((dbEnv - dbAgent)/3, 0)

class Character:
    #need to get rid of abs, I stopped using them
    def __init__(self, risk_aversion=1.5, risk_cutoff=5, absL=100, absDB=100, absNB=100, mainB="DB"):
        #multiplier for risky actions (negative rewards)
        self.risk_aversion = risk_aversion
        #cut off of negativity to care about
        self.risk_cutoff = risk_cutoff
        self.absL = absL
        self.absNB = absNB
        self.absDB = absDB
        self.mainB = mainB
        self.relL = self.calculate_relative_stats(absL)
        self.relNB = self.calculate_relative_stats(absNB)
        self.relDB = self.calculate_relative_stats(absDB)
        self.survival_rounds = 0
    
    #This doesn't work anymore because I'm using rel now
    def set_stats(self, relL, relDB, relNB):
        self.relL = relL
        self.relNB = relNB
        self.relDB = relDB

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

