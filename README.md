# Behavior Predictor Model
## Description
A personal project to predict behaviors (actions) from an agent trained on various, customizable situations. All rewards and stats simplified to 3 variables: livelihood, nurturing belonging, and defensive belonging. 

-Livelihood: the physical well-being of the agent
-Belonging: the social value of the agent.
  - Defensive: the social value of being able to eliminate objects or being who causes negative to society
  - Nurturing: the social value of being able to provide positive to society

Utilizes simple value networks for each actions. Each situation is self-resolvable from a single decision from agent and will not impact future situations (randomly determined). As such, these networks do not consider future rewards. 
Also, I believe a more accurate behavioral model can be determined if we simplify the complexity of our future predictions to snap decisions.  
