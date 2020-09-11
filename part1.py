# Assignment 1 ~ part1
# James Hooper ~ NETID: jah171230
# Hritik Panchasara ~ NETID: hhp160130
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


'''
    Gradient Descent Function
        ~ Inputs:
            ~ x = input attributes
            ~ y = output
            ~ weights = initialized weights
            ~ LR = learning rate
            ~ iterations = number of iterations
        ~ Returns:
            ~ final weights 
            ~ Mean Squared Error array to be graphed
'''
def enhanced_gradient_descent(x, y, weights, LR, iterations):
    # Graph MSE
    MSEgraph = np.zeros((iterations,1))
    for k in range(iterations):
        # Initialize Hypothesis
        H = np.dot(x, weights)
        # Define Error
        # E = H - Y
        E = np.subtract(H, y)
        # Define Mean Squared Error
        MSE = (1 / (2 * (int(len(y))))) * np.dot(np.transpose(E), E)
        # Place MSE value in correct array placement
        MSEgraph[k] = MSE
        # Define Gradient -> MSE derivative to weight
        gradient = (1 / (int(len(y)))) * np.dot(np.transpose(x), E)
        # Revise Weights
        # New Weight = Old Weight - Learning Rate * Gradient
        weights = np.subtract(weights, LR * gradient)
    return weights, MSEgraph

'''
    Pre-Processing Function
        ~ Process Data: 
            ~ Modifies Data
        ~ Graphic Display: 
            ~ Displays graphs about the Data for analysis
'''
def pre_process(data, state, print_data_graphs, drop_cols, split_size):
    '''
        Process Data ~ Drop Duplicates
            ~ Keep first instances of duplicates
    '''
    data.drop_duplicates(keep='first', inplace=True)

    if print_data_graphs == True:
        '''
            Graphic Display ~ Attribute Correlation Heatmap 
                COMMENT: Suction S.D.T. & Angle of Attack have .75 correlation value.
        '''
        # Comptue pairwise correlation of columns
        corr = data.corr()
        # Display Heatmap of Correlations
        axHeat = plt.axes()
        cmap = sns.light_palette("#2a9669", as_cmap=True)
        axi1 = sns.heatmap(corr, ax = axHeat, cmap = cmap, annot=True)
        axHeat.set_title('Heatmap of Attribute Correlation', fontsize=24)
        plt.show()

        '''
            Graphic Display ~ Attribute Plots (inputs & output)
        '''
        i = 1
        for column in data:
            plt.subplot(6, 1, i)
            plt.subplots_adjust(hspace=1.2)
            data[column].plot(color='#c73f24')
            plt.title(column, y=1.00, loc='center', color='#23c48e', fontsize=24, fontweight=24)
            i+=1
        plt.show()

    '''
        Process Data ~ Drop Columns
            ~ We can choose Suction S.D.T or Angle of Attack here.
    '''
    if drop_cols == True:
        data = data.drop(columns=['Suction S.D.T'])

    '''
        Process Data ~ Split & Scale Data
    '''
    data_x = data.drop(data.columns[-1], axis=1)
    data_y = data[['Sound Pressure Level']]

    # Define scaler
    scaler = StandardScaler()
    # Transform Data
    scaled_data_x = scaler.fit_transform(data_x)

    x_train, x_test, y_train, y_test = train_test_split(scaled_data_x, data_y, test_size=split_size, random_state=state)

    return x_train, y_train, x_test, y_test, scaler

'''
    Driver Function
'''
def main(state):
    # Attributes:
    # Frequency, Angle of Attack, Chord Length, Free-Stream Velocity, Suction S.D.T., Sound Pressure Level
    # 5 input variables, 1 output variable
    # Retrieve Data from GitHub Repository
    url = "https://raw.githubusercontent.com/jamesH-48/Enhanced-Gradient-Descent/master/airfoil_self_noise.csv"
    data = pd.read_csv(url, header=0)
    print(data)
    values = data.values

    '''
        Pre-Processing
            ~ Will be using standard scaler.
            ~ Can set if you want to print graphs out or not.
            ~ Can drop columns that are deemed droppable.
            ~ Can set train/test split size
                ~ .9 -> 90% train 10% test
                ~ .8 -> 80% train 20% test
                ~ etc.
            ~ Returns:
                    0         1        2       3       4
                ~ x_train, y_train, x_test, y_test, scaler
                ~ We will retrieve this as a list for ease of use.
    '''
    processed_info = []
    processed_info = pre_process(data, state, print_data_graphs=False, drop_cols=False, split_size=.9)

    '''
        Call the Enhanced Gradient Descent Function
            ~ Intialize weights, learning rate, iterations
            ~ Call Enhanced Gradient Descent Function
    '''
    # Initialize Weights
    Weights = np.array([[0],[0],[0],[0],[0]])
    # Initialize Learning Rate
    LR = .0001
    # Initialize Iterations
    iterations = 70000
    Final_Weights, MSEgraph = enhanced_gradient_descent(processed_info[0], processed_info[1], Weights, LR, iterations)

    '''
    Graphic Display ~ Mean Squared Error
    '''
    figureMSE, axMSE = plt.subplots()
    axMSE.plot(MSEgraph, color='#c73f24')
    axMSE.set_title("Mean Squared Error", color='#23c48e')
    axMSE.set_xlabel("No. of Iterations")

    plt.show()

'''
    Main Function
'''
if __name__ == '__main__':
    print("Part 1 of Enhanced Gradient Descent")
    # State is the seeded order of data that is randomized in train-test-split from sklearn
    state = 0
    main(state)
