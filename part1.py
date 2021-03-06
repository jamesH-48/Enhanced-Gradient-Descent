# Assignment 1 ~ part1
# James Hooper ~ NETID: jah171230
# Hritik Panchasara ~ NETID: hhp160130
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

'''
    Gradient Descent Function
    Implements Adam Optimizer
        ~ Inputs:
            ~ x = input attributes
            ~ y = output
            ~ weights = initialized weights
            ~ iterations = number of iterations
        ~ Returns:
            ~ final weights 
            ~ Mean Squared Error array to be graphed
'''
def enhanced_gradient_descent(x, y, weights, iterations):
    # Adam Optimizer Variables
    # Recommended values: alpha = 0.001, beta1 = 0.9, beta2 = 0.999 and epsilon = 10**−8
    alpha = .001
    beta1 = .9
    beta2 = .999
    epsilon = 10**-8
    m = 0
    v = 0

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

        # Calculate m for gradient component
        m = (beta1 * m) + ((1 - beta1) * gradient)
        # Calculate v for learing rate component
        v = (beta2 * v) + ((1 - beta2) * (gradient**2))

        # Get Adam Equation for weight update
        adam_equation = (((alpha)/(np.sqrt(v) + epsilon)) * m)

        # Revise Weights
        # New Weight = Old Weight - Adam Equation
        weights = np.subtract(weights, adam_equation)

    return weights, MSEgraph

'''
    Pre-Processing Function
        ~ Process Data: 
            ~ Modifies Data
        ~ Graphic Display: 
            ~ Displays graphs about the Data for analysis
'''
def pre_process(data, state, drop_cols, print_data_graphs, split_size):
    '''
        Process Data ~ Drop Duplicates
            ~ Keep first instances of duplicates
    '''
    data.drop_duplicates(keep='first', inplace=True)

    if print_data_graphs == True:
        # null value check
        print("null", data.isnull().sum())

        '''
            Graphic Display ~ Attribute Correlation Heatmap 
                COMMENT: AT & V have .84 correlation
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
            plt.subplot(5, 1, i)
            plt.subplots_adjust(hspace=1.2)
            data[column].plot(color='#c73f24')
            plt.title(column, y=1.00, loc='center', color='#23c48e', fontsize=24, fontweight=24)
            i+=1
        plt.show()

    '''
        Process Data ~ Drop Columns
            ~ We can choose AT or V here.
    '''
    if drop_cols == True:
        data = data.drop(columns=['AT'])

    '''
        Process Data ~ Split & Scale Data
    '''
    data_x = data.drop(data.columns[-1], axis=1)
    data_y = data[['PE']]
    # Convert to numpy array
    X = data_x.to_numpy()
    Y = data_y.to_numpy()

    '''
        Add Bias Term
            ~ Column of 1's
    '''
    bias = np.ones(shape=(X.shape[0],1))
    X = np.append(bias, X, axis=1)

    return train_test_split(X, Y, test_size=split_size, random_state=state)

'''
    Driver Function
'''
def main(state):
    # Attributes:
    # Frequency, Angle of Attack, Chord Length, Free-Stream Velocity, Suction S.D.T., Sound Pressure Level
    # 5 input variables, 1 output variable
    # Retrieve Data from GitHub Repository
    url = "https://raw.githubusercontent.com/jamesH-48/Enhanced-Gradient-Descent/master/Combined%20Cycle%20Power%20Plant%20.csv"
    data = pd.read_csv(url, header=0)

    '''
        Pre-Processing
            ~ Can set if you want to print graphs out or not.
            ~ Can drop columns that are deemed droppable.
            ~ Drop columns deemed necessary
            ~ There was a check for NaN values
            ~ Can set train/test split size
                ~ .1 -> 90% train 10% test
                ~ .2 -> 80% train 20% test
                ~ etc.
            ~ Returns:
                ~ x_train, x_test, y_train, y_test from train-test split of the pre-processed data
    '''
    drop_cols = False
    print_data_graphs = False
    split_size = .2
    X_train, X_test, Y_train, Y_test = pre_process(data, state, drop_cols, print_data_graphs, split_size)

    '''
    print("X_train: ", X_train.shape)
    print("X_test: ", X_test.shape)
    print("Y_train: ", Y_train.shape)
    print("Y_test: ", Y_test.shape)
    '''


    '''
        Call the Enhanced Gradient Descent Function
            ~ Intialize weights, learning rate, iterations
            ~ Call Enhanced Gradient Descent Function
            !!! IMPORTANT !!!
            The Enhanced Gradient Descent implements the Adam optimizer.
            The special optimizer values are defined in the function.
            Special Optimizer values include: alpha, beta1, beta2, and epsilon
    '''
    if drop_cols:
        # Initialize Weights
        Weights = np.array([[0],[0],[0],[0]])
    else:
        Weights = np.array([[0], [0], [0], [0], [0]])
    # Initialize Iterations
    iterations = 10000
    Final_Weights, MSEgraph = enhanced_gradient_descent(X_train, Y_train, Weights, iterations)

    '''
    Graphic Display ~ Mean Squared Error
    '''
    figureMSE, axMSE = plt.subplots()
    axMSE.plot(MSEgraph, color='#c73f24')
    axMSE.set_title("Mean Squared Error", color='#23c48e')
    axMSE.set_xlabel("No. of Iterations")

    plt.show()

    '''
        Final Values Print 
            ~ Mean Squared Error & R^2
            ~ Parameters Used
            ~ Coefficients
            ~ Graphs
    '''
    # Apply Model found Weights to Test Data Set
    # Get Y prediction Values from Test Data x Weights Found
    # Compare Y prediction Values with actual output values from test data set
    Y_pred1 = np.dot(X_train, Final_Weights)
    Y_pred2 = np.dot(X_test, Final_Weights)

    # Parameters Used
    print("Parameters Used:")
    print("State: ", state)
    # Adam Optimizer Variables
    # Recommended values: alpha = 0.001, beta1 = 0.9, beta2 = 0.999 and epsilon = 10**−8
    print("Alpha: .001\t|Beta1: .9\t|Beta2: .999\t|Epsilon: 10^-8\t|m = 0\t|v = 0")
    print("Iterations: ", iterations)
    print("Train Split: ", (1 - split_size)*100, "%\t|Test Split: ", split_size*100, "%")

    # Coefficients
    coef = []  # Initialize
    for i in range(Final_Weights.shape[0]):  # For Print & Bar Graph
        coef.append(Final_Weights[i][0])
    print('Coefficients: \n', coef)
    # Train Accuracy
    print("Train Accuracy:")
    print("Mean Squared Error: ", mean_squared_error(Y_pred1, Y_train))
    print("R^2 Value: ", r2_score(Y_pred1, Y_train))
    # Test Accuracy
    print("Test Accuracy:")
    print("Mean Squared Error: ", mean_squared_error(Y_pred2, Y_test))
    print("R^2 Value: ", r2_score(Y_pred2, Y_test))

    '''
    Graphic Display ~ Train Accuracy & Test Accuracy Plots
    '''
    # Print Plot of Outputs
    figure1, ax = plt.subplots()
    figure2, ax2 = plt.subplots()
    # Can't really gather anything from this graph since it is so dense.
    ax.plot(Y_train, color='#060064', markersize=5, label="Actual")
    ax.plot(Y_pred1, color='#daff4f', markersize=5, label="Prediction")
    ax.set_title('Y Train Dataset')
    ax.set_xlabel('No. of Values')
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
    ax2.plot(Y_test, color='black', markersize=5, label="Actual")
    ax2.plot(Y_pred2, color='#00ffc3', markersize=5, label="Prediction")
    ax2.set_title('Y Test Dataset')
    ax2.set_xlabel('No. of Values')
    ax2.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)

    if not drop_cols:
        '''
         Graphic Display ~ Coefficient Bar Graph
         '''
        # Weights Bar Graph
        labels = ['Temperature', 'Ambient Pressure', 'Relative Humidity', 'Exhaust Vacuum', 'Bias']
        x = np.arange(len(labels))  # Location of Labels
        width = .5  # Width of the bars
        figureW, axW = plt.subplots()
        bars = axW.bar(x, coef, width, color='#ff4f72')  # Coef is from Weight Print
        axW.set_ylabel('Weight')
        axW.set_title('Coefficients')
        axW.set_xticks(x)
        axW.set_xticklabels(labels)

    plt.show()

'''
    Main Function
'''
if __name__ == '__main__':
    print("Part 1 of Enhanced Gradient Descent")
    # State is the seeded order of data that is randomized in train-test-split from sklearn
    state = 5
    main(state)