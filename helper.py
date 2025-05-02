import matplotlib.pyplot as plt

def plot_curves(
    relL_values, relDB_values, relNB_values,
    sitL_values, sitDB_values, sitNB_values,
    action_values, survival_rounds_values,
    lLoss_values, dbLoss_values, nbLoss_values,
    sit_type, threat_action_values, ally_action_values, prey_action_values
):
    fig = plt.figure(figsize=(18, 30), constrained_layout=True)

    plt.subplot(6, 3, 1)
    plt.plot(relL_values, color = 'brown')
    plt.title('Character.relL')

    plt.subplot(6, 3, 2)
    plt.plot(relDB_values, color = 'navy')
    plt.title('Character.relDB')

    plt.subplot(6, 3, 3)
    plt.plot(relNB_values, color = 'lightgreen')
    plt.title('Character.relNB')

    plt.subplot(6, 3, 4)
    plt.plot(sitL_values, color = 'brown')
    plt.title('Situation sitL')

    plt.subplot(6, 3, 5)
    plt.plot(sitDB_values, color = "navy")
    plt.title('Situation sitDB')

    plt.subplot(6, 3, 6)
    plt.plot(sitNB_values, color = 'lightgreen')
    plt.title('Situation sitNB')

    plt.subplot(6, 3, 7)
    plt.scatter(range(len(action_values)), action_values)
    plt.title('Action')

    plt.subplot(6, 3, 8)
    plt.plot(survival_rounds_values)
    plt.title('Rounds Survived')

    plt.subplot(6, 3, 9)
    plt.scatter(range(len(lLoss_values)), lLoss_values, color= 'brown')
    plt.title('Livelihood Loss')

    plt.subplot(6, 3, 10)
    plt.scatter(range(len(dbLoss_values)), dbLoss_values, color='navy')
    plt.title('Defensive Belonging Loss')

    plt.subplot(6, 3, 11)
    plt.scatter(range(len(nbLoss_values)), nbLoss_values, color = 'lightgreen')
    plt.title('Nurturing Belonging Loss')

    plt.subplot(6, 3, 12)
    plt.scatter(range(len(sit_type)), sit_type)
    plt.title('Situation Type')
    plt.yticks([0, 1, 2], ['Threat','Ally','Prey'])


    plt.subplot(6, 3, 13)
    plt.scatter(range(len(threat_action_values)), threat_action_values, color='red')
    plt.title('Threat Actions')
    plt.yticks([-2, -1, 0, 1, 2, 3, 4], ['Helpless','Nothing Worth Doing','Fight', 'Flee', 'Befriend', 'Chase', 'Cry'])

    plt.subplot(6, 3, 14)
    plt.scatter(range(len(ally_action_values)), ally_action_values, color='blue')
    plt.title('Ally Actions')
    plt.yticks([-2, -1, 0, 1, 2, 3, 4], ['Helpless','Nothing Worth Doing','Fight', 'Flee', 'Befriend', 'Chase', 'Cry'])

    plt.subplot(6, 3, 15)
    plt.scatter(range(len(prey_action_values)), prey_action_values, color='green')
    plt.title('Prey Actions')
    plt.yticks([-2, -1, 0, 1, 2, 3, 4], ['Helpless','Nothing Worth Doing','Fight', 'Flee', 'Befriend', 'Chase', 'Cry'])



    plt.show()
