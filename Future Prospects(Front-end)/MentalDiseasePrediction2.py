#Importing the dataframe
import pandas as pd
df=pd.read_csv('data.csv')

#Converting Gender to Numerical data
df['Gender2']=pd.factorize(df.Gender)[0]       #Replacing them

#Converting Stay Category to Numerical data
df['Stay_Cate2']=pd.factorize(df.Stay_Cate)[0]       #Replacing them

#Converting English Category to Numerical data
df['English_cate2'] = pd.factorize(df.English_cate)[0]       #Replacing them

#Converting Department to Numerical data
df['Dep2'] = pd.factorize(df.Dep)[0]       #Replacing them

#Developing the front end GUI Interface
from tkinter import *

root=Tk()

#Creating the GUI Dimension
root.geometry("600x600")
root.configure(background="light green")

#The Title of the Presentation
Label(root,text="MENTAL HEALTH PREDICTION", font=('Helvetica',15,'bold'), bg="light green", relief="solid").pack()

#End Tag
Label(root,text="Application Version 2.0", relief="solid", bg="light green").pack(side=BOTTOM)

#Implementing the User Credentials

#Enter the age
Label(root,text="Enter the age in years", font=('Helvetica',10,'bold'),bg="light green",relief="solid").place(x=40,y=80)

#Enter the Gender
Label(root,text="Enter 0 for Male, 1 for Female: ", font=('Helvetica',10,'bold'),bg="light green",relief="solid").place(x=40,y=130)

#Enter the fear measurement
Label(root,text="Enter the fear measurement  : ", font=('Helvetica',10,'bold'),bg="light green",relief="solid").place(x=40,y=180)

#Enter the stay category
Label(root,text="For stay Enter 0 for long, 1 for short, 2 for medium: ", font=('Helvetica',10,'bold'),bg="light green",relief="solid").place(x=40,y=230)

#Enter the English Proficiency category
Label(root,text="For Eng Prof Enter 0 for High, 1 for Average, 2 for Low: ", font=('Helvetica',10,'bold'),bg="light green",relief="solid").place(x=40,y=280)

#Predicting Result Set-box
Label(root,text="The Predicted Result: ", font=('Helvetica',10,'bold'),bg="light green",relief="solid").place(x=40,y=430)

#Setting the variables
age=StringVar()
gender=StringVar()
fear=StringVar()
stay=StringVar()
eng=StringVar()

#Implementing set-boxes for the user to input values
Entry(root,text=age,width=25).place(x=400,y=80)
Entry(root,text=gender,width=25).place(x=400,y=130)
Entry(root,text=fear,width=25).place(x=400,y=180)
Entry(root,text=stay,width=25).place(x=400,y=230)
Entry(root,text=eng,width=25).place(x=400,y=280)

def model():
    #Defining the features and labels
    x1 = df.iloc[:, 23:24] #Afear feature
    x2 = df.iloc[:, 50:51] #Gender feature
    x3 = df.iloc[:, 4:5] #Age feature
    x4=df.iloc[:, 51:52] #Stay category feature
    x5=df.iloc[:, 52:53] #English feature
    #Joining the features
    x6=x5.join(x4)
    x7=x6.join(x3)
    x8=x7.join(x2)
    x=x8.join(x1) #The feature column
    x.fillna(0, inplace=True) #Filling the empty spaces with 0
    y = df.iloc[:, 53:54] #Label
    y.fillna(0, inplace=True) #Filling the empty spaces with 0

    #Defining the model
    from sklearn.neighbors import KNeighborsClassifier #Using the KNN Classifier
    model=KNeighborsClassifier(n_neighbors=5,metric='minkowski')
    model.fit(x,y) #Training the model
    x_test=[int(age.get()),int(gender.get()),int(fear.get()),int(stay.get()),int(eng.get())] #Getting the values
    y_pred=model.predict([x_test,]) #Getting the predicted result
    s="No" #1 for No Depression
    if(str(list(y_pred)[0])=='0'): #0 for Depression
        s="Yes"
    Label(root,text=s,font=('Helvetica',10,'bold'),bg="light green",relief="solid").place(x=220,y=430) #Printing the output

#Putting the buttons for various commands
Button(root,text="Depression Prediction",width=18,command=model).place(x=400,y=440)  #Predicting the suicidal outcome
Button(root,text="Termination",width=18,command=root.destroy).place(x=400,y=500) #Closing the application

root.resizable(0,0)

root.mainloop() #Ending the GUI