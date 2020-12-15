from importlib import import_module
import torch
from utils import build_iterator
import pickle as pkl


class PredWeather:
    def __init__(self):
        model_name = "TextCNN"  # bert
        x = import_module(model_name)
        self.config = x.Config()
        self.config.batch_size = 1
        self.vocab = pkl.load(open(self.config.vocab_path, 'rb'))
        self.config.n_vocab = len(self.vocab)
        self.model = x.Model(self.config).to("cpu")
        self.model.load_state_dict(torch.load(self.config.save_path))
        self.model.eval()

    @staticmethod
    def load_dataset(text, vocab, pad_size=32):
        tokenizer = lambda x: [y for y in x]  # char-level
        UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
        token = tokenizer(text)
        words_line = []
        if pad_size:
            if len(token) < pad_size:
                token.extend([PAD] * (pad_size - len(token)))
            else:
                token = token[:pad_size]

        for word in token:
            words_line.append(vocab.get(word, vocab.get(UNK)))

        return words_line, 0, len(text)

    def predict_text(self, input_text):
        label_dict = {0: "other", 1: "weather"}
        model_in = self.load_dataset(input_text, self.vocab)
        test_iter = build_iterator([model_in], self.config)

        with torch.no_grad():
            for texts, labels in test_iter:
                outputs = self.model(texts)
                label = torch.max(outputs.data, 1)[1].cpu().numpy()[0]
                return label_dict[label]


if __name__ == '__main__':
    TEXT = "今天吃什么"
    _s = PredWeather()
    print("模型加载完毕")
    while True:
        print(_s.predict_text(input()))
