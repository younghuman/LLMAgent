class ExpertActionPredictor():
    def __init__(self, bart_model, bert_model, predict_func, softmax=True, rule=False):
        self.bart_model=bart_model
        self.bert_model=bert_model
        self.predict_func=predict_func
        self.softmax = softmax
        self.rule = rule
    
    def predict(self, obs, info):
        return self.predict_func(obs, info, self.bert_model, softmax=self.softmax, rule=self.rule, bart_model=self.bart_model)

