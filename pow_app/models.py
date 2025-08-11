from django.db import models

class Users(models.Model):
    
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
        ('O','Others')
    ]
    
    STATE_CHOICES = [
        ('Punjab', 'Punjab'),
        ('Haryana', 'Haryana'),
        ('Rajasthan', 'Rajasthan'),
        ('Delhi', 'Delhi'),
        ('UP', 'Uttar Pradesh'),
        ('Uttarakhand', 'Uttarakhand'),
        ('HP', 'Himachal Pradesh'),
        ('J&K', 'Jammu & Kashmir'),
        ('Chandigarh', 'Chandigarh'),
        ('Chhattisgarh', 'Chhattisgarh'),
        ('Gujarat', 'Gujarat'),
        ('MP', 'Madhya Pradesh'),
        ('Maharashtra', 'Maharashtra'),
        ('Goa', 'Goa'),
        ('DNH', 'Dadra and Nagar Haveli'),
        ('Andhra Pradesh', 'Andhra Pradesh'),
        ('Telangana', 'Telangana'),
        ('Karnataka', 'Karnataka'),
        ('Kerala', 'Kerala'),
        ('Tamil Nadu', 'Tamil Nadu'),
        ('Pondy', 'Puducherry'),
        ('Bihar', 'Bihar'),
        ('Jharkhand', 'Jharkhand'),
        ('Odisha', 'Odisha'),
        ('West Bengal', 'West Bengal'),
        ('Sikkim', 'Sikkim'),
        ('Arunachal Pradesh', 'Arunachal Pradesh'),
        ('Assam', 'Assam'),
        ('Manipur', 'Manipur'),
        ('Meghalaya', 'Meghalaya'),
        ('Mizoram', 'Mizoram'),
        ('Nagaland', 'Nagaland'),
        ('Tripura', 'Tripura')
    ]
    
    Name=models.CharField(max_length=20,null=True) 
    Gender=models.CharField(max_length=10,choices=GENDER_CHOICES, null=True)   
    State=models.CharField(max_length=20,choices=STATE_CHOICES,null=True)
    Email=models.EmailField(null=True)
    Dob=models.DateField(auto_now=False,auto_now_add=False,null=True)
    Pwd=models.CharField(max_length=255,default='NA',null=True)
    Household_ID=models.CharField(max_length=20, unique=True, null=True)
    
def __str__(self):
        return self.Email
# Create your models here.
