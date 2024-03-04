
import re

class Tokenizer():
  def __init__(self, text):
    self.body = text

  def selfies_encode(self,body):

    #typical selfies regex
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    tokens = regex.findall(body)
    return tokens

  def train_bpe(self, body, steps, pairs=[], filename='vocab'):

    tokens = self.selfies_encode(body)

    pairs = sorted(pairs, key=len, reverse=True)
    answer = []
    skips = []
    threshold = 0
    for iter in range(steps):
        new_token = []

        i = 0
        while i < len(tokens):
            matched_pair = None
            for pair in pairs:

                if ''.join(tokens[i:i+len(pair)]) == ''.join(pair):

                    matched_pair = ''.join(pair)
                    break
            if matched_pair:
                new_token.append(matched_pair)
                i += len(pair)
            else:
                new_token.append(tokens[i])
                i += 1

        freq_dict = dict()

        for el in range(len(new_token) -1):
            pair = new_token[el]+new_token[el+1]
            if pair not in freq_dict:
                freq_dict[pair] =1
            else:
                freq_dict[pair] +=1
        d = sorted(freq_dict.items(),  key=lambda item: item[1], reverse=True)
        s = d[0][0].split(']')[:-1]

        l = [x+']' for x in s]
        if '.' in ''.join(l):
            parts = re.split(r'(\.)', ''.join(l))
            l = [part for part in parts if part]

        if iter >0:
            if answer[iter-1][1][0] == d[threshold][0]:
                threshold +=1
                print('Skip this token due to some technical issue')
                skips.append(answer[iter-1][1][0])

        pairs.append(l)
        pairs = sorted(pairs, key=len, reverse=True)

        print(f'Iteration: {iter+1}, most often pair: {d[threshold][0]}, occurs {d[threshold][1]} times, length of all tokenized data set: {len(new_token)}')

        answer.append([iter,d[threshold],len(new_token)])

        #saving pairs

        with open(filename+'.txt', 'w') as f:

          for pair in pairs:
            f.write(''.join(pair)+'\n')
          for token in list(set(tokens)):
            f.write(token+'\n')

    return answer

  def train_ngram(self):
    pass

  def encode(self, body, model):
    model = sorted(list(set(model)), key=len, reverse=True)

    tokens = self.selfies_encode(body)

    char_to_token = {char: idx for idx, char in enumerate(model)}
    encoded_text = [char_to_token[token] for token in tokens]
    return encoded_text

  def decode(self,encoded_text,model):
    model = sorted(list(set(model)), key=len, reverse=True)
    token_to_char = {char: idx for char, idx in enumerate(model)}
    decoded_text = [token_to_char[idx] for idx in encoded_text]
    return ''.join(decoded_text)

  def load(self, filename = 'vocab'):

    with open(filename+'.txt', 'r') as f:
      pairs = f.readlines()
    return [line.rstrip('\n') for line in pairs]


if __name__ == '__main__':
    t = Tokenizer('[^][In][Branch2][Ring1][=C][C][Branch1][=Branch2][Si][Branch1][C][C][Branch1][C][C][C][Branch1][=Branch2][Si]')

    t.train_bpe('[^][In][Branch2][Ring1][=C][C][Branch1][=Branch2][Si][Branch1][C][C][Branch1][C][C][C][Branch1][=Branch2][Si]', 4)

    loaded_vocab = t.load()
    testy = t.encode('[^][In][Branch2][Ring1][=C][C][Branch1][=Branch2][Si][Branch1][C][C][Branch1][C][C][C][Branch1][=Branch2][Si]', loaded_vocab)

    print(t.decode(testy, loaded_vocab))

