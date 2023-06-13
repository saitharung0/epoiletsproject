from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    address= models.CharField(max_length=300)
    gender= models.CharField(max_length=30)


class epilots_landing_type(models.Model):

    Activity_Id = models.CharField(max_length=3000)
    Landing_Airport = models.CharField(max_length=3000)
    Airline_Name= models.CharField(max_length=3000)
    Operating_Airline_IATA_Code = models.CharField(max_length=3000)
    Landing_Date = models.CharField(max_length=3000)
    Published_Airline= models.CharField(max_length=3000)
    Published_Airline_IATA_Code = models.CharField(max_length=3000)
    GEO_Summary= models.CharField(max_length=3000)
    GEO_Region= models.CharField(max_length=3000)
    Landing_Aircraft_Type= models.CharField(max_length=3000)
    Aircraft_Body_Type= models.CharField(max_length=3000)
    Aircraft_Manufacturer= models.CharField(max_length=3000)
    Aircraft_Model= models.CharField(max_length=3000)
    Aircraft_Version= models.CharField(max_length=3000)
    Landing_Count= models.CharField(max_length=3000)
    Total_Landed_Weight= models.CharField(max_length=3000)
    Prediction= models.CharField(max_length=3000)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



