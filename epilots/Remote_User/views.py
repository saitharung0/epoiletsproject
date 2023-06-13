from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import openpyxl
import numpy as np

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,epilots_landing_type,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):
    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city, address=address, gender=gender)
        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html', {'object': obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Landing_Type(request):
    if request.method == "POST":
        url_text = request.POST.get('keyword')
        if request.method == "POST":

            Activity_Id= request.POST.get('Activity_Id')
            Landing_Airport= request.POST.get('Landing_Airport')
            Airline_Name= request.POST.get('Airline_Name')
            Operating_Airline_IATA_Code= request.POST.get('Operating_Airline_IATA_Code')
            Landing_Date= request.POST.get('Landing_Date')
            Published_Airline= request.POST.get('Published_Airline')
            Published_Airline_IATA_Code= request.POST.get('Published_Airline_IATA_Code')
            GEO_Summary= request.POST.get('GEO_Summary')
            GEO_Region= request.POST.get('GEO_Region')
            Landing_Aircraft_Type= request.POST.get('Landing_Aircraft_Type')
            Aircraft_Body_Type= request.POST.get('Aircraft_Body_Type')
            Aircraft_Manufacturer= request.POST.get('Aircraft_Manufacturer')
            Aircraft_Model= request.POST.get('Aircraft_Model')
            Aircraft_Version= request.POST.get('Aircraft_Version')
            Landing_Count= request.POST.get('Landing_Count')
            Total_Landed_Weight= request.POST.get('Total_Landed_Weight')


        data = pd.read_csv("Air_Landings_Statistics.csv", encoding='latin-1')

        def apply_results(label):
            if (label == 0):
                return 0
            elif (label == 1):
                return 1

        data['Results'] = data['Landing_Status'].apply(apply_results)
        x = data['Activity_Id'].apply(str)
        y = data['Results']

        #cv = CountVectorizer(lowercase=False, strip_accents='unicode', ngram_range=(1, 1))
        #x = cv.fit_transform(data['Activity_Id'].apply(lambda x: np.str_(x)))
        cv = CountVectorizer()
        x = cv.fit_transform(x)


        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB

        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print(naivebayes)
        print(confusion_matrix(y_test, predict_nb))
        print(classification_report(y_test, predict_nb))
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm

        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression

        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))

        print("Decision Tree Classifier")
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        dtcpredict = dtc.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, dtcpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, dtcpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, dtcpredict))
        models.append(('DecisionTreeClassifier', dtc))



        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = lin_clf.predict(X_test)

        Activity_Id1 = [Activity_Id]
        vector1 = cv.transform(Activity_Id1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = str(pred.replace("]", ""))

        prediction = int(pred1)

        #Activity_Id1 = [Activity_Id]
        #vector1 = cv.transform(Activity_Id1).toarray()
        #predict_text = lin_clf.predict(vector1)

        #pred = str(predict_text).replace("[", "")
        #pred1 = pred.replace("]", "")
        #prediction = re.sub("[^a-zA-Z]", " ", str(pred1))

        if prediction == 0:
                val = 'Hard Landing'
        elif prediction == 1:
                val = 'Soft Landing'

        print(prediction)
        print(val)


        epilots_landing_type.objects.create(Activity_Id=Activity_Id,
        Landing_Airport=Landing_Airport,
        Airline_Name=Airline_Name,
        Operating_Airline_IATA_Code=Operating_Airline_IATA_Code,
        Landing_Date=Landing_Date,
        Published_Airline=Published_Airline,
        Published_Airline_IATA_Code=Published_Airline_IATA_Code,
        GEO_Summary=GEO_Summary,
        GEO_Region=GEO_Region,
        Landing_Aircraft_Type=Landing_Aircraft_Type,
        Aircraft_Body_Type=Aircraft_Body_Type,
        Aircraft_Manufacturer=Aircraft_Manufacturer,
        Aircraft_Model=Aircraft_Model,
        Aircraft_Version=Aircraft_Version,
        Landing_Count=Landing_Count,
        Total_Landed_Weight=Total_Landed_Weight,
        Prediction=val)

        return render(request, 'RUser/Predict_Landing_Type.html',{'objs': val})
    return render(request, 'RUser/Predict_Landing_Type.html')



