from django.urls import path
from . import views



urlpatterns = [
    path('', views.home, name='home'),                        # Home page
    path('register/', views.register, name='register'),
    path('createlogin/', views.createlogin, name='createlogin'),
    path('login/', views.login_view, name='custom_login'),
    path('predict_state/', views.predict, name='predict_page'),     # Original prediction page
    path('result/', views.result, name='result'),             # Original result page
    path('predict/', views.prediction_options, name='predict_options'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('logout/', views.logout, name='logout'),             # Logout view
    path('household_predict/', views.household_predict, name='household_powercut'),
    path('household_result/', views.household_result_view, name='household_result'),
]
# urlpatterns = [
#     path('', views.home, name='home'),               # Home page
#     path('register/', views.register, name='register'),
#     path('createlogin/', views.createlogin, name='createlogin'),
#     path('login/', views.login_view, name='custom_login'),
#     path('predict/', views.predict, name='predict_page'),   # Prediction page
#     path('result/', views.result, name='result'),    # Result page
#     path('dashboard/', views.dashboard, name='dashboard'),
   
#     path('logout/', views.logout, name='logout'),  # Logout view
    
# ]

