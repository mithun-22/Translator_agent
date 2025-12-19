from django.db import models

# Create your models here.
class Translation(models.Model):
    source_text = models.TextField()
    source_lang_slug = models.CharField(max_length=5, default='')  
    target_lang_slug = models.CharField(max_length=5, default='')
    translated_text = models.TextField(blank = True, null = True)
