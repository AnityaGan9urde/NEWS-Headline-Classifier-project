from django.db import models

# Create your models here.

class newNews(models.Model):
    headline = models.CharField(max_length=100)
    topic = models.CharField(max_length=50)
    model = models.CharField(max_length=50)