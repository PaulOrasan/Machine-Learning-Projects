from Classifier import Classifier
class ClassifierManager:
    def __init__(self):
        self._models = []

    def addModel(self, model: Classifier):
        self._models.append(model)

    def testModels(self, test_dataset):
        for model in self._models:
            model.test(test_dataset)

    def showEvaluations(self):
        for model in self._models:
            print(model)
            evaluation = model.getEvaluations()
            print("Training:")
            print("Accuracy: ", evaluation['training']['accuracy'])
            print("Precision: ", evaluation['training']['precision'])
            print("Recall: ", evaluation['training']['recall'])
            print()
            print("Testing:")
            print("Accuracy: ", evaluation['testing']['accuracy'])
            print("Precision: ", evaluation['testing']['precision'])
            print("Recall: ", evaluation['testing']['recall'])
            print()
