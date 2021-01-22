# char-RNN Generator

``` python
# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
```
저번 RNN과 같이 각 나라들의 txt에서 이름들을 불러들어 온다. readlines 함수를 정의해서 각 나라들의 이름을 lines에 저장하고 그 다음에 category_lines 에 딕셔너리 형태로 저장하게 된다.


Generator 의 형태는 이렇게 되어있다.

![Generator](https://i.imgur.com/jzVrf7f.png)

input과 category, hidden state를 입력받고 있다. input에는 문자가 들어가고 category에는 나라가 들어간다. 세개의 input을 combined 하고 i2o 와 i2h로 나뉘어서 hidden state로 다시 쓰이고 output하고 hidden 하고 다시 합쳐서 output을 만들어 낸다. o2o레이어를 만들고 난 뒤에 dropout layer을 만들어서 overfiting 방지한다. 그 다음에 softmax 함수를 통해 다음에 나올 문자열을 예상한다. 그런 형태로 되어 있다.

```python
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
```

__init__에서 i2o, i2h, o2o, dropout, sofmax를 정의해주고 있다. i2o, i2h, o2o는 전부 linear 구조로 되어 있다. 그 다음에 forward함수에서 각 레이어들의 구조를 연결하고 hidden state하고 output을 출력한다. 그 다음에 train 구조를 보면

```python
criterion = nn.NLLLoss()

learning_rate = 0.0005

def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)
```
 NLLLoss()로 loss 값을 계산해준다. 각 문자마다 rnn 에 넣어서 output과 hidden이 나오면 hidden state는 다음 문자할 때 들어가고 output하고 실제 다음에 들어가는 문자하고 비교해서 loss를 계산한다. 문자가 끝이 나면 이제 backward()로 역전파를 하고 모델을 training 해 준다. 이렇게 training 된 모델에 문자열을 넣어서 단어들을 predict할 수 있다.

 ```python

 max_length = 20

# Sample from a category and starting letter
def sample(category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))

```

맨 처음에 category와 시작 문자열을 입력으로 받고 그것을 tensor로 만든다. 그다음 hidden state를 초기화 하고 모델에 넣어줘서 문자가 EOS에 도달할 때까지, 단어가 끝날 때까지 돈다. 그 후에 예상 단어들을 출력해준다.

한번 테스트를 해보았는데

```python
samples('Korean', 'ABCDEFGHIJKLMNPQRSTUVWXYZ')
```

<details>
<summary> > 결과값 보기 </summary>
<div markdonw = "1">

```
Ang
Bou
Con
Don
Eon
Fon
Gon
Hon
Ion
Jon
Kon
Long
Man
Non
Pon
Qun
Ro
Shon
Thon
Uon
Von
Won
Xon
Yon
Zon
```

</div>
</details>
<br>

앞서서 했던 RNN 과 원리가 비슷해서 쉽게 이해가 된 것 같다. 이 RNN 원리를 이용해서 Nickname Generator를 만들 수 있지 않을까 생각이 들었다.  