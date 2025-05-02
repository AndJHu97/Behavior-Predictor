from enum import Enum
import numpy as np
import random

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
    Chase = 3
    #Work towards a goal
    Cry = 4
    #Investigate = 4
    #Ridicule = 5
    #Submit = 6
    #Ignore = 7

class SituationType(Enum):
    Threat = 0
    Ally = 1
    Prey = 2
    Count = 3

class Situation:
    def __init__(self, sitL, sitDB, sitNB, sitType, societyL =0, societyNB = 0, societyDB = 0):
        self.sitL = sitL
        self.sitDB = sitDB
        self.sitType = sitType
        self.sitNB = sitNB
        self.societyL = societyL
        self.societyNB = societyNB
        self.societyDB = societyDB
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
    #Add society metrics here
    def __init__(self, sitL, sitDB, sitNB, sitType, societyL, societyNB, societyDB):
        super().__init__(sitL, sitDB, sitNB, sitType, societyL, societyNB, societyDB) 
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
        elif action == Action.Chase.value:
            lChange = self.calculateLChase(character.relL, self.sitL)
            dbChange = self.calculateDBChase(character.relDB, self.sitDB, character.relL, self.sitL)
        elif action == Action.Cry.value:
            lChange = self.calculateLCry(character.relL, character.relNB, character.relDB, self.sitL, self.societyL, self.societyDB)
            dbChange = self.calculateDBCry(character.relL, character.relNB, character.relDB, self.sitL, self.sitDB, self.societyL, self.societyDB)

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
            dbEnd = max((dbAgent + dbEnv)/2 + (lAgent * 0.7 - lEnv * 1.1) / 2, 0) 
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
    def calculateLChase(self, lAgent, lEnv):
        print("Fight chase L change: ", min(lAgent * 0.9 - lEnv * 1.1, 0))
        return min(lAgent * 0.7 - lEnv * 1.1, 0)
    def calculateDBChase(self, dbAgent, dbEnv, lAgent, lEnv):
        dbEnd = (dbAgent + dbEnv)/2 + (lAgent * 0.9 - lEnv * 1.1) / 2
        dbEnd = min(dbEnd, 0)
        print("Fight Chase DB Change: ", dbEnd)
        return dbEnd
    def calculateLCry(self, lAgent, nbAgent, dbAgent, lEnv, lSoc, dbSoc):
        mainBValue = 0
        #Getting the majority b value
        if nbAgent == dbAgent:
            mainBValue = random.choice([nbAgent, dbAgent])
        elif nbAgent < dbAgent:
            mainBValue = dbAgent
        else:
            mainBValue = nbAgent
        print("Agent Cry in Fight - Agent's B Opportunity Cost: ", (lAgent - (lAgent * 0.3 - lEnv * 1.1)/lAgent) * mainBValue)
        print("Agent Cry in Fight - Society's B Opportunity Cost: ", (lSoc - (lSoc * 0.9 - lEnv * 1.1)/lSoc * dbSoc))
        if (lAgent - (lAgent * 0.3 - lEnv * 1.1)/lAgent) * mainBValue > (lSoc - (lSoc * 0.9 - lEnv * 1.1)/lSoc * dbSoc):
            lChange = max(lAgent * 0.3 + lSoc - lEnv * 1.1, 0)
            print("Cry in Fight, L loss (worth intervene): ", lChange)
            return lChange
        else:
            lChange = max(0.3 * lAgent - 1.1 * lEnv, 0)
            print("Cry in fight, L loss (society doesn't intervene): ", lChange)
            return lChange
    def calculateDBCry(self, lAgent, nbAgent, dbAgent, lEnv, dbEnv, lSoc, dbSoc):
        mainBValue = 0
        #Getting the majority b value
        if nbAgent == dbAgent:
            mainBValue = random.choice([nbAgent, dbAgent])
        elif nbAgent < dbAgent:
            mainBValue = dbAgent
        else:
            mainBValue = nbAgent
        print("Agent Cry in Fight - Agent's B Opportunity Cost: ", (lAgent - (lAgent * 0.3 - lEnv * 1.1)/lAgent) * mainBValue)
        print("Agent Cry in Fight - Society's B Opportunity Cost: ", (lSoc - (lSoc * 0.9 - lEnv * 1.1)/lSoc * dbSoc))
        if (lAgent - (lAgent * 0.3 - lEnv * 1.1)/lAgent) * mainBValue > (lSoc - (lSoc * 0.9 - lEnv * 1.1)/lSoc * dbSoc):
            dbChange = min((dbEnv * 0.1 - dbAgent) * 0.3, 0)
            print("Agent Cry in Fight - DB loss (societal intervention): ", dbChange)
            return dbChange
        else:
            dbEnd = (dbAgent + dbEnv)/2 + (lAgent * 0.3 - lEnv * 1.1) / 2
            dbEnd = min(dbEnd, 0)
            dbFightChange = dbEnd - dbAgent
            dbChange = min((dbEnv * 0.1 - dbAgent) * 0.3 + dbFightChange, 0)
            print("Agent Cry in Fight - DB loss (no societal intervention): ", dbChange)
            return dbChange



class Ally(Situation):
    def __init__(self, sitL, sitDB, sitNB, sitType):
        super().__init__(sitL, sitDB, sitNB, sitType)
    def process_action(self, character, action):
        death = False
        lChange, dbChange, nbChange = 0, 0, 0
        if action == Action.Fight.value:
            lChange = self.calculateLFight(character.relL, self.sitL)
            dbChange = self.calculateDBFight(character.relL, character.relDB, self.sitL, self.sitDB)
            nbChange = self.calculateNBFight(character.relNB, self.sitNB)
        elif action == Action.Flee.value:
            lChange = self.calculateLFlee()
            dbChange = self.calculateDBFlee(character.relDB, self.sitDB)
            nbChange = self.calculateNBFlee()
        elif action == Action.Befriend.value:
            lChange = self.calculateLBefriend(character.relL, self.sitNB, self.sitDB)
            dbChange = self.calculateDBBefriend(character.relDB, self.sitDB)
            nbChange = self.calculateNBBefriend(character.relNB, self.sitNB)
        elif action == Action.Chase.value:
            lChange = 0
            dbChange = 0
            nbChange = self.calculateNBChase(character.relNB, self.sitNB)
        elif action == Action.Cry.value:
            lChange = 0
            dbChange = self.calculateDBCry(character.relDB, self.sitDB)
            nbChange = 0

        
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
    def calculateNBFight(self, nbAgent, nbEnv):
        print("Ally NB Fight: ", -nbEnv * 0.2/(nbAgent * 0.5))
        return -nbEnv * 0.2/(nbAgent * 0.5)
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
    
    def calculateDBCry(self, dbAgent, dbEnv):
        dbChange = (dbEnv * 0.1 - dbAgent) * 0.3
        print("Ally DB cry: ", dbChange)
        return dbChange
    
    def calculateNBChase(self, nbAgent, nbEnv):
        print("Ally NB Cry: ", -nbEnv * 0.1/(nbAgent * 0.5))
        return -nbEnv * 0.1/(nbAgent * 0.5)


class Prey(Situation):
    def __init__(self, sitL, sitDB, sitNB, sitType):
        super().__init__(sitL, sitDB, sitNB, sitType)
    def process_action(self, character, action):
        death = False
        lChange, dbChange, nbChange = 0, 0, 0
        if action == Action.Fight.value:
            lChange = self.calculateLFight(character.relL, self.sitL)
            dbChange = self.calculateDBFight(character.relL, self.sitL, character.relDB, self.sitDB)
            nbChange = self.calculateNBFight(character.relL, self.sitNB, character.relNB)
        elif action == Action.Flee.value:
            lChange = 0
            dbChange = self.calculateDBFlee(character.relDB, self.sitDB)
            nbChange = 0
        elif action == Action.Befriend.value:
            lChange = self.calculateLBefriend(character.relL)
            dbChange = 0
            nbChange = 0
        elif action == Action.Chase.value:
            lChange = self.calculateLChase(character.relL, self.sitL)
            dbChange = self.calculateDBChase(character.relL, self.sitL, character.relDB, self.sitDB)
            nbChange = self.calculateNBChase(character.relL, self.sitNB, character.relNB)
        elif action == Action.Cry.value:
            lChange = 0
            dbChange = self.calculateDBCry(character.relDB, self.sitDB)
            nbChange = self.calculateNBCry(character.relL, self.sitL, character.relNB)
            
        
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
        if lEnv < 10 and lAgent > lEnv * 0.5: 
            print("Fight won against prey and gained L: ", 0.7 * lEnv)
            return 0.7 * lEnv
        else:
            print("Fight lost against prey")
            return 0
        
    def calculateDBFight(self, lAgent, lEnv, dbAgent, dbEnv):
        if lEnv < 10 and lAgent > lEnv * 0.5:
            dbEnd = (dbAgent + dbEnv)/2 + (lAgent - lEnv * 0.5)/2
            print("Fight won against prey and gained DB ", dbEnd - dbAgent)
            return dbEnd - dbAgent
        elif lEnv * 0.5 > lAgent and lEnv < 10:
            dbEnd = (dbAgent + dbEnv)/2 + (lAgent - lEnv * 0.5)
            print("Fight lost against prey and lost DB ", dbEnd - dbAgent)
            return dbEnd - dbAgent
        else: 
            print("Prey ran away and gained 0 DB")
            return 0
        
    def calculateNBFight(self, lAgent, nbEnv, nbAgent):
        if nbEnv * 0.5 < lAgent and nbEnv < 10:
            print("Fight with prey won and NB gained: ", nbEnv * 0.5)
            return nbEnv * 0.5
        else:
            nbChange = -0.5 * nbEnv * max( lAgent - nbEnv * 0.8, 0 ) / lAgent
            print("Fight with prey lost or ran away and NB lost: ", nbChange)
            return nbChange
        
    
    def calculateDBFlee(self, dbAgent, dbEnv):
        dbChange = dbEnv * 0.7 - dbAgent
        print("Prey Flee with DB change: ", dbChange)
        return max(dbChange, 0)

    def calculateLBefriend(self, lAgent):
        print("Befriend prey and L lost ", lAgent * 0.1)
        return -lAgent * 0.1

    def calculateLChase(self, lAgent, lEnv):
        if lEnv < lAgent or lEnv < 10:
            print("Successful chase of prey and gain ", lEnv * 0.7)
            return lEnv * 0.7
        else:
            print("Unsuccessful chase of prey and gain 0")
            return 0

    def calculateDBChase(self, lAgent, lEnv, dbAgent, dbEnv):
        if(lEnv < lAgent or lEnv < 10):
            dbEnd = (dbAgent + dbEnv)/2 + (lAgent - lEnv) / 2
            print("Prey DB Chase: ", (dbEnd - dbAgent))
            print("lAgent: ", lAgent, " dbAgent: ", dbAgent, " lEnv: ", lEnv, " dbEnv: ", dbEnv)
            return min(dbEnd - dbAgent, 0)
        else:
            return 0

    def calculateNBChase(self, lAgent, nbEnv, nbAgent):
        if (nbEnv < lAgent or nbEnv < 10):
            print("Prey NB chase with prey caught: ", nbEnv * 0.5)
            return nbEnv * 0.5
        else:
            nbChange = -0.5 * nbEnv * max( lAgent - nbEnv * 0.8, 0 ) / lAgent 
            print("Prey NB chase didn't catch: ", nbChange)
            return nbChange
        
    def calculateDBCry(self, dbAgent, dbEnv):
        dbChange = (dbEnv * 0.1 - dbAgent) * 0.3
        print("Prey DB Cry: ", dbChange)
        return dbChange
    
    def calculateNBCry(self, lAgent, lEnv, nbAgent):
        nbChange = -0.1 * lAgent * max(lAgent - lEnv * 0.8, 0) / lAgent / (nbAgent * 0.5)
        print("Prey NB Cry: ", nbChange)
        return nbChange
        

    
    