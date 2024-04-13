import torch.nn as nn
from torchvision import models
from py_apps.utils.classify_app import ClassifyApp
import networkx as nx
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import json
import neuro

def getMulticlassModel():
    resnet18 = models.resnet18()
    resnet18.fc = nn.Sequential(nn.Linear(resnet18.fc.in_features, 2), nn.Softmax(dim=1))# change to binary classification 
    resnet18.conv1 =  nn.Sequential(
                                nn.AvgPool2d((1, 32), stride=(1, 32)), # shrink input 
                                nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # change first conv layer to greyscale (for the spectrogram )
                                ) 
    return resnet18, nn.CrossEntropyLoss()
    
def getBindayClassification():
    resnet18 = models.resnet18()
    resnet18.fc = nn.Sequential(nn.Linear(resnet18.fc.in_features, 1), nn.Sigmoid())# change to binary classification 
    resnet18.conv1 =  nn.Sequential(
                                nn.AvgPool2d((1, 32), stride=(1, 32)), # shrink input 
                                nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # change first conv layer to greyscale (for the spectrogram )
                                ) 
    return resnet18, nn.BCELoss()

class ClassifyAudioApp(ClassifyApp):
    def __init__(self, config, X, y):
        super().__init__(config, X, y) 
    
    def fitness(self, net: neuro.Network, proc, id_for_printing=-1):
        
        net.prune()
        # missing_path_penalty = self._count_missing_input_output_paths(net)
        # if missing_path_penalty != 0:
        #     return missing_path_penalty
        y_predict = self.predict(proc, net, self.X_train)
        if (self.fitness_type == "accuracy"):
            ret = accuracy_score(self.y_train, y_predict) 
        elif (self.fitness_type == "f1"):
            ret = f1_score(self.y_train, y_predict, average="weighted")
        return ret
    
    def _count_missing_input_output_paths(self, net: neuro.Network):
            # Modified BFS to only visit a node when ALL of its incoming nodes have already been visited
            snnModel_raw = json.loads(json.dumps(net.as_json().to_python()))
            G = nx.Graph()

            for n in snnModel_raw['Nodes']:
                G.add_node(n['id'])

            for e in snnModel_raw['Edges']:
                G.add_edge(e['from'], e['to'], weight=round(e['values'][0],3))

            inputNodes = snnModel_raw["Inputs"]
            outputNodes = snnModel_raw["Outputs"]

            number_of_missing_paths = 0
            for subStation in inputNodes:
                for out in outputNodes:
                    if not nx.has_path(G,subStation, out):
                        number_of_missing_paths = number_of_missing_paths - 10
            return number_of_missing_paths