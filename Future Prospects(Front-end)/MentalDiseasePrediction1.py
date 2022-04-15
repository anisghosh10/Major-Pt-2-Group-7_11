#Importing the dataframe
import pandas as pd
df=pd.read_csv('data.csv')

#Converting Intimate to Numerical data
df['Intimate2']=pd.factorize(df.Intimate)[0]       #Replacing them

#Converting Academic to Numerical data
df['Academic2']=pd.factorize(df.Academic)[0]       #Replacing them

#Converting Suicide to Numerical data
df['Suicide2'] = pd.factorize(df.Suicide)[0]       #Replacing them

#Developing the front end GUI Interface
from tkinter import *

root=Tk()

#Creating the GUI Dimension
root.geometry("600x600")
root.configure(background="light green")

#The Title of the Presentation
Label(root,text="MENTAL HEALTH PREDICTION", font=('Helvetica',15,'bold'), bg="light green", relief="solid").pack()

#End Tag
Label(root,text="Application Version 1.0", relief="solid", bg="light green").pack(side=BOTTOM)

#Implementing the User Credentials

#Enter the age
Label(root,text="Enter the age in years", font=('Helvetica',10,'bold'),bg="light green",relief="solid").place(x=40,y=80)

#Enter the Gender
Label(root,text="Enter the Gender: ", font=('Helvetica',10,'bold'),bg="light green",relief="solid").place(x=40,y=130)

#Enter if the person had any intimacy with anyone. Enter 0 for yes, 1 for no
Label(root,text="Enter 0 for Intimacy or 1 for no  : ", font=('Helvetica',10,'bold'),bg="light green",relief="solid").place(x=40,y=180)

#Enter if the person is graduate or not
Label(root,text="Enter 0 if graduate or 1 if under-graduate: ", font=('Helvetica',10,'bold'),bg="light green",relief="solid").place(x=40,y=230)

#Enter the willingness to take Doctor's help
Label(root,text="Enter the willingness to doctor score: ", font=('Helvetica',10,'bold'),bg="light green",relief="solid").place(x=40,y=280)

#Enter the willingness to take Professional's help
Label(root,text="Enter the willingness to professional score: ", font=('Helvetica',10,'bold'),bg="light green",relief="solid").place(x=40,y=330)

#Enter the Age Category from 1 to 5:
Label(root,text="Enter the Age Category from 1 to 5: ", font=('Helvetica',10,'bold'),bg="light green",relief="solid").place(x=40,y=380)

#Predicting Result Set-box
Label(root,text="The Predicted Result: ", font=('Helvetica',10,'bold'),bg="light green",relief="solid").place(x=40,y=430)

#Setting the variables
age=StringVar()
gender=StringVar()
inti=StringVar()
grad=StringVar()
doc=StringVar()
pro=StringVar()
cat=StringVar()

#Implementing set-boxes for the user to input values
Entry(root,text=age,width=25).place(x=400,y=80)
Entry(root,text=gender,width=25).place(x=400,y=130)
Entry(root,text=inti,width=25).place(x=400,y=180)
Entry(root,text=grad,width=25).place(x=400,y=230)
Entry(root,text=doc,width=25).place(x=400,y=280)
Entry(root,text=pro,width=25).place(x=400,y=330)
Entry(root,text=cat,width=25).place(x=400,y=380)

def model():
    #Defining the features and labels
    x1 = df.iloc[:, 5:6] #Age_cate feature
    x2 = df.iloc[:, 50:51] #Intimate feature
    x3 = df.iloc[:, 51:52] #Academic feature
    x4=df.iloc[:,33:34] #Profess feature
    x5=df.iloc[:,34:35] #Doctor feature
    #Joining the features
    x6=x5.join(x4)
    x7=x6.join(x3)
    x8=x7.join(x2)
    x=x8.join(x1) #The feature column
    x.fillna(0, inplace=True) #Filling the empty spaces with 0
    y = df.iloc[:, 52:53] #Label
    y.fillna(0, inplace=True) #Filling the empty spaces with 0

    #Defining the model
    from sklearn.neighbors import KNeighborsClassifier #Using the KNN Classifier
    model=KNeighborsClassifier(n_neighbors=5,metric='minkowski')
    model.fit(x,y) #Training the model
    x_test=[int(inti.get()),int(grad.get()),int(doc.get()),int(pro.get()),int(cat.get())] #Getting the values
    y_pred=model.predict([x_test,]) #Getting the predicted result
    s="No" #0 for No Suicide
    if(str(list(y_pred)[0])=='1'): #1 for Suicide
        s="Yes"
    Label(root,text=s,font=('Helvetica',10,'bold'),bg="light green",relief="solid").place(x=220,y=430) #Printing the output

#Putting the buttons for various commands
Button(root,text="Suicide Prediction",width=18,command=model).place(x=400,y=440)  #Predicting the suicidal outcome
Button(root,text="Termination",width=18,command=root.destroy).place(x=400,y=500) #Closing the application

root.resizable(0,0)

root.mainloop() #Ending the GUI