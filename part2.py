# Assignment 1 ~ part1
# James Hooper ~ NETID: jah171230
# Hritik Panchasara ~ NETID: hhp160130
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor

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
    split_size = .1
    X_train, X_test, Y_train, Y_test = pre_process(data, state, drop_cols, print_data_graphs, split_size=split_size)

    '''
    print("X_train: ", X_train.shape)
    print("X_test: ", X_test.shape)
    print("Y_train: ", Y_train.shape)
    print("Y_test: ", Y_test.shape)
    '''

    '''
        Call Sklearn Stochastic Gradient Descent Regressor
            ~ Since we can't find anything that uses the Adam Optimizer we will use this.
            ~ We will be using the default values unless specified otherwise.
            ~ Default values can be found here: 
            ~ https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor
    '''
    # Initialize Iterations
    iterations = 10000
    # Initialize Learning Rate
    LR = .000001

    # Create Linear Regression Object
    regr = SGDRegressor(loss="squared_loss", penalty=None, max_iter=iterations, eta0=LR)
    # Train the model using the training datasets
    regr.fit(X_train,Y_train.ravel())
    # Make predictions using the testing dataset
    Y_pred1 = regr.predict(X_train)
    Y_pred2 = regr.predict(X_test)

    '''
        Final Values Print 
            ~ Mean Squared Error & R^2
            ~ Parameters Used
            ~ Coefficients
            ~ Graphs
    '''
    # Parameters Used
    print("Parameters Used:")
    print("State: ", state)
    print("Iterations: ", iterations)
    print("Learning Rate: ", LR)

    # Coefficients
    print('Coefficients: \n', regr.coef_)
    # Train Accuracy
    print("Train Accuracy:")
    print("Mean Squared Error: ", mean_squared_error(Y_pred1,Y_train))
    print("R^2 Value: ", r2_score(Y_pred1,Y_train))
    # Test Accuracy
    print("Test Accuracy:")
    print("Mean Squared Error: ", mean_squared_error(Y_pred2,Y_test))
    print("R^2 Test: ", r2_score(Y_pred2,Y_test))

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

    '''
     Graphic Display ~ Coefficient Bar Graph
     '''
    # Weights Bar Graph
    labels = ['Temperature', 'Ambient Pressure', 'Relative Humidity', 'Exhaust Vacuum', 'Bias']
    x = np.arange(len(labels))  # Location of Labels
    width = .5  # Width of the bars
    figureW, axW = plt.subplots()
    bars = axW.bar(x, regr.coef_, width, color='#ff4f72')  # Coef is from Weight Print
    axW.set_ylabel('Weight')
    axW.set_title('Coefficients')
    axW.set_xticks(x)
    axW.set_xticklabels(labels)

    plt.show()

'''
    Main Function
'''
if __name__ == '__main__':
    print("Part 2 of Enhanced Gradient Descent")
    # State is the seeded order of data that is randomized in train-test-split from sklearn
    state = 5
    main(state)
