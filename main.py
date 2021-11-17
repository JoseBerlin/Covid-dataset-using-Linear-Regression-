## importing req packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

## loading the dataset
df = pd.read_csv("Dataset.csv")

## Assigning independent and dependent variables to x & y respectively
x = df[['Total Cases', 'Active Ratio (%)', 'Discharge Ratio (%)']]      # independent variables
y = df[['Death Ratio (%)']]                                             # dependent variable
list_training_error = []
list_testing_error = []

## Splitting the data into Train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

## Creating a linear regression model
mlr = linear_model.LinearRegression()

## Training the model with traing data
mlr.fit(x_train.values, y_train.values)

## Predicting the model with test data
# print(x_test)
pred = mlr.predict(x_test.values)

## Using r2_score to evaluate the performance of a linear regression model
test_r2 = r2_score(y_test, pred)
print("Accuracy : {}".format(test_r2))

## printing Actual values vs predicted values
# print('Actual value:\n', y_test, "\nPredicted value: \n", pred)

### With GUI
root = tk.Tk()
canvas1 = tk.Canvas(root, width=500, height=500)
canvas1.pack()

def accuracy():
    Prediction_result = ('Accuracy : ' + str(test_r2))
    label_Prediction = tk.Label(root, text=Prediction_result, bg='yellow')
    canvas1.create_window(260, 280, window=label_Prediction)

def viz():
    root1 = tk.Tk()
    canvas1 = tk.Canvas(root1, width=0, height=0)
    canvas1.pack()

    # plot 1st scatter
    figure3 = plt.Figure(figsize=(5, 4), dpi=50)
    ax3 = figure3.add_subplot(111)
    ax3.scatter(df['Total Cases'].astype(float), df['Death Ratio (%)'].astype(float), color='r')
    scatter3 = FigureCanvasTkAgg(figure3, root1)
    scatter3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH)
    ax3.legend(['Death Ratio (%)'])
    ax3.set_xlabel('Total Cases')
    ax3.set_ylabel('Death Ratio(%)')
    ax3.set_title('Total Cases Vs. Death Ratio (%)')

    # plot 2nd scatter
    figure4 = plt.Figure(figsize=(5, 4), dpi=50)
    ax4 = figure4.add_subplot(111)
    ax4.scatter(df['Active'].astype(float), df['Death Ratio (%)'].astype(float), color='g')
    scatter4 = FigureCanvasTkAgg(figure4, root1)
    scatter4.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH)
    ax4.legend(['Death Ratio (%)'])
    ax4.set_xlabel('Active')
    ax4.set_ylabel('Death Ratio(%)')
    ax4.set_title('Active Vs. Death Ratio (%)')

    # plot 3rd scatter
    figure5 = plt.Figure(figsize=(5, 4), dpi=50)
    ax5 = figure5.add_subplot(111)
    ax5.scatter(df['Discharged'].astype(float), df['Death Ratio (%)'].astype(float), color='b')
    scatter5 = FigureCanvasTkAgg(figure5, root1)
    scatter5.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH)
    ax5.legend(['Death Ratio (%)'])
    ax5.set_xlabel('Discharged')
    ax5.set_ylabel('Death Ratio(%)')
    ax5.set_title('Discharged Vs. Death Ratio (%)')

def predict():
    if txttotalcases.get() !='' or txtactive.get()!='' or txtdischarged.get()!='':
        global total
        total = float(txttotalcases.get())

        global active
        active = float(txtactive.get()) / total * 100

        global discharged
        discharged = float(txtdischarged.get()) / total * 100

        Prediction_result = "Predicted Death Ratio(%): "+str(float(mlr.predict([[total, active, discharged]])))
        print(Prediction_result)

        label_pred = tk.Label(root, text=str(Prediction_result), bg="yellow")
        canvas1.create_window(160, 370, window=label_pred)
    else:
        label_pred = tk.Label(root, text="Invalid Input. Try a valid values", bg="red")
        canvas1.create_window(160, 370, window=label_pred)

## Labels and text boxes for GUI
lbltotalcases = tk.Label(root, text="Enter Total cases ", anchor="w")
lbltotalcases.place(x=30, y=300, width=100)
txttotalcases = tk.Entry(root)
txttotalcases.place(x=150, y=300, width=100)
lblactive = tk.Label(root, text="Enter Active cases: ", anchor="w")
lblactive.place(x=30, y=315, width=100)
txtactive = tk.Entry(root)
txtactive.place(x=150, y=315, width=100)
lbldischarged = tk.Label(root, text="Discharged cases: ", anchor="w")
lbldischarged.place(x=30, y=330, width=100)
txtdischarged = tk.Entry(root)
txtdischarged.place(x=150, y=330, width=100)

## Button to be viewed on Root window
button1 = tk.Button(root, text='View Accuracy', command=accuracy, bg='green')
canvas1.create_window(200, 250, window=button1)

button2 = tk.Button(root, text='Predict now', command=predict, bg='white')
canvas1.create_window(330, 350 , window=button2)

button4 = tk.Button(root, text='view Graphs', command=viz, bg='white')
canvas1.create_window(300, 250, window=button4)
root.mainloop()

