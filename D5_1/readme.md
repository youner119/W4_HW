# Character-level RNN

문자를 연속으로 읽어드려서 각 단계의 예측과 hidden state를 출력하고 다음 단계에 이전 hidden state을 넘겨주고 단어의 끝이 오면 최종 예측을 한다.

그래서 이런 형태로 되어 있다

![구조](https://i.imgur.com/Z2xbySO.png)

단어에서 각 문자를 연속적으로 읽어와서 input으로 주고 output하고 hidden state가 나오는데 그 그래서 나온 hidden state하고 그 다음 문자를 combined 한다. 

----
먼저 데이터를 불러와야 한다. Korean.txt 이런 형식으로 그 안에는 이름들이 있다. 그렇게 18개국의 이름이 나와있다. 

```python
# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
```

코드를 보면 category에는 나라가 들어간다. all_categories에는 나라 이름이 들어가고 category_lines는 딕셔너리 형태로 각 나라의 이름들이 들어가 있다.

그다음에 단어를 컴퓨터가 사용할 수 있게 데이터화 시켜야 한다. Tensor로 변경해야 한다. 하나의 문자를 바꾸기 위해서 크기가 <1, n_letter>인 one-hot Vector를 이용한다. "c" = <0, 0, 1, ....> 이렇게 변환한다. 그리고 이제 단어를 그런식으로 <단어 길이, 1, n_letter>의 tesnsor 로 변환한다.

```python
import torch

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J'))

print(lineToTensor('Jones').size())
```

lineToTensor로 단어를 Tensor로 변환해주고 코드를 실행해보면

```
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0.]])
torch.Size([5, 1, 57])
```
이렇게 나오는 것을 알 수 있다.

그 다음에 Model을 보면

```python
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
```

이렇게 되어있다. 

![구조](https://i.imgur.com/Z2xbySO.png)

i2h, i2o다 linear구조로 되어있다. hidden layer와 input layer가 combined 하고 output과 hidden 값이 출력이 된다.

----
이제 training을 해보자. 순서는 이렇게 된다.

1. 단어를 넣고 그것을 tensor로 만든다
2. 0으로 초기화된 hidden state를 만들어서 처음에 동작할 수 있게 한다.
3. hidden state를 계속 넘겨가면서 문자를 계속 통과시킨다. 
4. 최종 나오는 output를 정답이랑 비교한다.
5. 역전파
6. output과 loss를 출력한다.

```python
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()
```
입력값으로 line_tensor와 category_tensor를 받아들이고 initHidden()으로 0으로 초기화된 hidden state를 만든다. 그 다음 output, hidden = rnn(line_tensor[i], hidden)을 for문을 돌리면서 hidden state 계속 돌리고 최종 output을 남기고 output과 실제 정답을 이용해서 loss를 계산한다. loss 계산하고 그 다음에 backward() 하고 모델을 수정한다.

그래서 train 하는 코드는 이렇게 된다.

```python
mport time
import math

n_iters = 100000
print_every = 5000
plot_every = 1000



# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
```

plt를 이용해서 loss를 보면

``` python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
```

![all_losses](https://pytorch.org/tutorials/_images/sphx_glr_char_rnn_classification_tutorial_001.png)

training이 많이 될 때 마다 loss 값이 줄어드는 게 보인다. training을 다 하고 나서 이제 직접 써볼 수 있다.

```python
def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')
```

모델에 input 값을 집어넣어 테스트 할 수 있다.