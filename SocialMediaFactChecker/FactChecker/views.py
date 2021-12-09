from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

from MachineLearning import ModelCombiner

def index(request):
    template = loader.get_template("index.html")

    context = {}
    if request.method == 'POST':

        model = ModelCombiner.joinmodel()
        result = model.getTrueFalse(request.POST['fact'], verbose=True)

        context['text'] = request.POST['fact'];

        if result['modelTweets_1'] != result['modelNews_2']:
            context['fact'] = 'Maybe'
        else:
            context['fact'] = result['combinedResult']

    return HttpResponse(template.render(context, request))