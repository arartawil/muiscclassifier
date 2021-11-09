from django.http import HttpResponse
from django.shortcuts import render

from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage


import librosa
import math
import tensorflow as tf
import numpy as np
from .myfunc import load_file


def index(request):
    return render(request, 'index.html', {})


def simple_upload(request):
    error = -1
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        type= request.POST.get('lang')

        print(type)
        ext=myfile.name.split('.')[1]
        if(ext == "wav"):
            fs = FileSystemStorage()
            #filename = fs.save(myfile.name, myfile)
            data=load_file(myfile,type)
            #uploaded_file_url = fs.url(filename)
            error = 0
            return render(request, 'simple_upload.html', {
                'uploaded_file_url': data,
                'error':error
            })
        else:
            error = 1
            return render(request, 'simple_upload.html', {
                'error': error
            })

    return render(request, 'simple_upload.html',{
                'error': error
            })


