from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('start_detection/', views.start_detection, name='start_detection'),
    path('stop_detection/', views.stop_detection, name='stop_detection'),
    path('start_training/', views.start_training, name='start_training'),
    path('stop_training/', views.stop_training, name='stop_training'),
    path('check_auto_advance/', views.check_auto_advance, name='check_auto_advance'),
    path('get_random_letter/', views.get_random_letter, name='get_random_letter'),
    path('reference_image/<str:letter>/<str:filename>', views.serve_reference_image, name='serve_reference_image'),
]
