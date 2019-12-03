#!/usr/bin/env python3

#import relevant libaries 
from tkinter import *
from tkinter import ttk
from ttkthemes import ThemedTk
import pandas as pd
import joblib
import numpy as np 
from tkinter import messagebox
from PIL import ImageTk, Image
import webbrowser

def callback(url):
    webbrowser.open_new(url)

description = '''The Exeter ML Coronary Heart Disease Risk Calculator
uses machine learning to predict whether a patient is 
percieved to be likely or unlikely to develop CHD 
within 10 years based on current indicators.'''

#categories for data entry 
Sex = 'Sex', 
FIELDS= {'Age':('30-70','years'), 'Height':('','m'), 'Weight':('','kg'), 'Cigarettes per day':('','(average)'), 
                   'Systolic BP':('Norm: 100-120','mm Hg'),'Heart Rate':('','bpm'),
                   'Total Cholestrol':('Norm: 150-200','mg/dL'), 'Glucose':('Norm: 70 to 100','mg/dL')}
check = 'BP Medication?', 'Prevalent Hypertension?',
education = 'Education', 


def fetch(entries):
    "Function that fetches the entered data and prints it out"
    data_input = [1.0]
    data_columns = ['const']
    for entry in entries:
        field = (entry[0])
        text  = (entry[1].get())
        data_input.append(text)
        data_columns.append(field)

    df_inputs = pd.DataFrame([data_input], columns=data_columns)
    
    #dictionary for education conversion
    d_edu = {
        "High school/ GCSE level": 1,
        "Sixth Form/ A level": 2,
        "Post 18 training": 3,
        "College/ University Degree": 4
        }
    
    #Convert education to int 1,2,3,4
    df_inputs['Education'].iloc[0] = d_edu[df_inputs['Education'].iloc[0]]
    
    print(df_inputs.iloc[0])
    
    if check_not_empty_or_invalid(df_inputs) == True: 
        df_inputs['BMI'] = float(df_inputs.Weight[0]) / (float(df_inputs.Height[0])**2)
            
        #convert to list to pass to Logistic Regression to calculate % chance 
        new_order = ['const', 'Age','Systolic BP', 'Sex', 'Cigarettes per day', 'Glucose', 'Total Cholestrol',  
                     'Prevalent Hypertension?','Education','Heart Rate','BMI','BP Medication?' ]
        df_inputs = df_inputs[new_order]
        print('p1')
        
        if check_range(df_inputs) == True:
            print('p2')
            
            df=list(df_inputs.iloc[0]) 

            #Load scaling and fit function
            LR_jl = joblib.load('LR.pkl')
            scalar_jl = joblib.load('scaler.pkl')

            #Predict the probability of getting CHD within 10 years
            y = LR_jl.predict_proba(scalar_jl.transform([df_inputs.iloc[0]]))
            prob_yes = y[0,1] # the probability of prediciting yes
            messagebox.showinfo("Result", 'Probability of getting CHD is {:.2f}%:'.format(prob_yes*100))


def check_not_empty_or_invalid(df_inputs):
    '''Function to check that all user enries are valid
    - Check no missing fields
    - Check input is a float or int (not string etc.)
    '''
    def is_number(s):
        "Check whether a string contains a number"
        try:
            s = float(s)
            return True        
        except ValueError: 
            return False
    
    for col in FIELDS.keys():
        if is_number(df_inputs[col].iloc[0]) == False: # if not integer of decimal
            messagebox.showwarning("Warning","Please input a number for {}".format(col))
            return False # error code 
        else:
            #Convert vals to floats
            df_inputs[col].iloc[0] = float(df_inputs[col].iloc[0])
    return True

def check_range(df_inputs):
        #dictionary for parameter ranges - taken as max and min values from Framingham data - avoid extrapolation
        DATA_RANGES = {'const':(1,1),'Age':(30,70),'Systolic BP':(80,300), 'Sex':(0,1), 'Cigarettes per day':(0,70), 
                'Glucose':(40,400), 'Total Cholestrol':(100,700), 'Prevalent Hypertension?':(0,1),
                'Education':(1,4),'Heart Rate':(40,150),'BMI':(15,60),'BP Medication?':(0,1)}
        
        for col in df_inputs.columns:
            # Is value within allowed range
            if df_inputs[col].iloc[0] < DATA_RANGES[col][0] or df_inputs[col].iloc[0] > DATA_RANGES[col][1]:
                result = messagebox.askokcancel("Warning","{} = {} out of normal range {}. Continue anyways? (may result in inaccurate result)"
                                                .format(col,round(df_inputs[col].iloc[0],1),DATA_RANGES[col]))
                #Allow user to cancel or continue and risk inaccuracy
                return result # error code -1
        return True

    
#create function to enter data 
def makeform(root):
    "Function creating the entry widgets for the data, includes combo box, text entry and checkbuttons"
    entries = []
    
    for field in Sex:
        ttk.Separator(root,orient=HORIZONTAL).pack(side=TOP, fill=X)
        row = Frame(root)
        lab = Label(row, width=20, text=field, anchor='w')
        ent = IntVar()
        ent.set(1)
        rad1 = Radiobutton(row, text="Male", var=ent, value=1) # 1 for male 
        rad2 = Radiobutton(row, text="Female", var=ent, value=0) # 0 for female
        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)
        rad1.pack(side=LEFT, anchor='w')
        rad2.pack(side = LEFT, anchor ='w')
        entries.append((field, ent)) 
        
    for field in FIELDS.keys():
        ttk.Separator(root,orient=HORIZONTAL).pack(side=TOP, fill=X)
        row = Frame(root)
        lab = Label(row, width=20, text=field, anchor='w')
        ent = EntryWithPlaceholder(row,FIELDS[field][0])
        entries.append((field, ent))
        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=LEFT)
        labelText=StringVar()
        labelText.set(FIELDS[field][1])
        labelDir=Label(row, textvariable=labelText)
        labelDir.pack(side=LEFT)
    
    for field in check:
        ttk.Separator(root,orient=HORIZONTAL).pack(side=TOP, fill=X)
        row = Frame(root)
        lab = Label(row, width=20, text=field, anchor='w')
        ent = IntVar() 
        ent.set(0)
        chk = Checkbutton(row,  var=ent)  
        row.pack(side=TOP, fill=X, padx=5, pady=5 )
        lab.pack(side=LEFT)
        chk.pack()
        entries.append((field, ent)) 
    
    for field in education: 
        ttk.Separator(root,orient=HORIZONTAL).pack(side=TOP, fill=X)
        row = Frame(root)
        lab = Label(row, width=20, text = field, anchor ='w')
        # ent = ttk.Combobox(row, values = ['0', '1', '2', '3', '4'])
        # 1 for some high school e.g. GCSE, 2 for a 
        # high school diploma or GED e.g. A levels, 3 
        # for some college or vocational school e.g Other post , and 4 for a college degree.
        OPTIONS = [
        "High school/ GCSE level",
        "Sixth Form/ A level",
        "Post 18 training",
        "College/ University Degree"
        ]

        ent = StringVar()
        ent.set(OPTIONS[0]) # default value
        menu = OptionMenu(row, ent, *OPTIONS)
        menu.config(width=25)
        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)
        menu.pack(side=RIGHT, expand=NO, fill=X)
        entries.append((field, ent))
    return entries

#quitting function to prevent crashing 
#def quit(root):
   # root.destroy() 
   # exit()
def clear(root,entries):
    # Activated on press of clear button. Clears all text boxes
    for entry in entries[1:9]:
        entry[1].clear_text()
        
    
class EntryWithPlaceholder(Entry):
    # Custom class for making special text boxes with temporary text
    def __init__(self, master=None, placeholder="PLACEHOLDER", color='grey'):
        super().__init__(master, width = 20)

        self.placeholder = placeholder
        self.placeholder_color = color
        self.default_fg_color = self['fg']
       
        
        self.bind("<FocusIn>", self.foc_in) # When clicked on
        self.bind("<FocusOut>", self.foc_out) # When clicked off

        self.put_placeholder()
        

    def put_placeholder(self):
        self.insert(0, self.placeholder)
        self['fg'] = self.placeholder_color

    def foc_in(self, *args):
        if self['fg'] == self.placeholder_color:
            self.delete('0', 'end')
            self['fg'] = self.default_fg_color

    def foc_out(self, *args):
        if not self.get():
            self.put_placeholder()       
    
    
def exitApp():
    exit()
    
if __name__ == '__main__':
    "Main function to produce the interface"
    LOGO_SIZE = 230
    
    root = ThemedTk(theme = 'arc')
    root.title('CHD predictor')
    root.resizable(False, False)
    #     root.configure(background="#222")
    
    path = "./Logo.png"
    img = Image.open(path)  # PIL solution
    img = img.resize((LOGO_SIZE, LOGO_SIZE), Image.ANTIALIAS) #The (250, 250) is (height, width)
    img = ImageTk.PhotoImage(img)
    panel = ttk.Label(root, image = img, anchor='center')    
    panel.image = img
    panel.pack(side = TOP, fill = X)
    panel.configure(background=root.cget('bg'))
    
    lab = Label(root, width=20, text=description,font='Helvetica 14',pady=5)
    lab.pack(side=TOP, fill=X)
   
    
    ents = makeform(root)   
    
    ttk.Separator(root,orient=HORIZONTAL).pack(side=TOP, fill=X)
    b1 = Button(root, text='Calculate', bg = "#81b4d4", width = 22, command=lambda e=ents:fetch(e))
    b1.pack(side=RIGHT, padx=5, pady=5)
    btn_clear = Button(root, text="Clear",bg = "#81b4d4", width = 22, command=(lambda e=ents,root=root:clear(root,e)))
    btn_clear.pack(side=LEFT, padx=5, pady=5)
    #b2 = Button(root, text="Quit", command=exitApp).pack()
    #b2.pack(side=RIGHT, padx=5, pady=5)
    
    root.mainloop()
