from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

from MachineLearning import ModelCombiner

def index(request):
    template = loader.get_template("index.html")

    context = {}
    if request.method == 'POST':

        model = ModelCombiner.joinmodel()
        result = model.getTrueFalse(request.POST, verbose=True)
        context['fact'] = 'result'

    return HttpResponse(template.render(context, request))