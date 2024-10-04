# from django.db import models

# # Create your models here.
# class React(models.Model):
#   employee = models.CharField(max_length=30)
#   department = models.CharField(max_length=200)
# models.py

from django.db import models

class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploads/')  # This saves the file in the 'uploads/' directory
    uploaded_at = models.DateTimeField(auto_now_add=True)  # Automatically sets the timestamp when uploaded

    def __str__(self):
        return f"{self.file.name} uploaded on {self.uploaded_at}"
