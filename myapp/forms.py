from django import forms


class imageUploadForm(forms.Form):
    # name = forms.CharField()
    Upload_Image = forms.ImageField()