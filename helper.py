import matplotlib.pyplot as plt

def plot_curves(relL_values, relB_values, sitL_values, sitB_values, action_values, survival_rounds_values, lLoss_values, bLoss_values, sit_type):
    plt.figure(figsize=(12, 12))

    plt.subplot(5, 2, 1)
    plt.plot(relL_values)
    plt.title('Character.relL')

    plt.subplot(5, 2, 2)
    plt.plot(relB_values)
    plt.title('Character.relB')

    plt.subplot(5, 2, 3)
    plt.plot(sitL_values)
    plt.title('Threat.sitL')

    plt.subplot(5, 2, 4)
    plt.plot(sitB_values)
    plt.title('Threat.sitB')

    plt.subplot(5, 2, 5)
    plt.plot(action_values)
    plt.title('Action')

    plt.subplot(5, 2, 6)
    plt.plot(survival_rounds_values)
    plt.title('Rounds Survived')

    plt.subplot(5, 2, 7)
    plt.scatter(range(len(lLoss_values)), lLoss_values)
    plt.title('Livelihood loss')

    plt.subplot(5, 2, 8)
    plt.scatter(range(len(bLoss_values)), bLoss_values)
    plt.title('Belonging loss')

    plt.subplot(5, 2, 9)
    plt.scatter(range(len(sit_type)), sit_type)
    plt.title('Situation Type')

    plt.tight_layout()
    plt.show()


